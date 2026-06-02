"""Main training entry point.

Usage:
    uv run python -m patholens.training.train --config configs/grounding_v2.yaml \\
        --run-name grounded_v2_run1

Architecture recap (relevant to this file):
    - Vision tokens: CONCHv1.5 patch embeddings (N_v, 768), capped at max_vision_tokens
    - LLM: Llama-3.2-3B-Instruct + LoRA r=16
    - Loss: CombinedLoss = caption + λ_g * grounding + λ_f * faithfulness
    - Grounding GT: BCSS masks (where available) or CONCH v1 pseudo-GT
"""

from __future__ import annotations

import logging
from typing import Literal

import h5py
import numpy as np
import torch
import typer
from rich.console import Console
from torch import Tensor
from transformers import Trainer, TrainingArguments

from patholens.data.conch_pseudo_gt import PseudoGTCache
from patholens.data.dataset import GroundingSlideDataset, grounding_collate_fn
from patholens.models.grounded_vlm import GroundedVLM
from patholens.training.grounding_targets import build_target
from patholens.training.losses import CombinedLoss
from patholens.utils.config import load_config
from patholens.utils.logging import setup_logger

app = typer.Typer()
console = Console()
logger = setup_logger(__name__)

# ─── Trainer ──────────────────────────────────────────────────────────────────

_BCSS_MASK_KEYS = ("mask_tumor", "mask_stroma", "mask_lymph", "mask_necrosis", "roi_coverage")


class PathoTrainer(Trainer):
    """HuggingFace Trainer subclass for PathoLens grounding-aware training.

    Overrides compute_loss to:
        1. Call GroundedVLM.forward with output_attentions when grounding is enabled
        2. Build per-sentence grounding targets via build_target() (BCSS or pseudo-GT)
        3. Compute CombinedLoss and log individual components to W&B

    Args:
        combined_loss: configured CombinedLoss instance
        pseudo_gt_cache: disk-backed cache for CONCH v1 pseudo-GT distributions
        conch_v1_encoder: loaded ConchV1Encoder (for pseudo-GT computation)
        grounding_source: 'conch_only' or 'bcss_hybrid'
        adapter_lr: learning rate for the linear adapter (default 1e-4)
        *args, **kwargs: forwarded to transformers.Trainer
    """

    def __init__(
        self,
        *args,
        combined_loss: CombinedLoss,
        pseudo_gt_cache: PseudoGTCache,
        conch_v1_encoder: object,
        grounding_source: Literal["conch_only", "bcss_hybrid"] = "bcss_hybrid",
        adapter_lr: float = 1e-4,
        grounding_warmup_epochs: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.combined_loss = combined_loss
        self.pseudo_gt_cache = pseudo_gt_cache
        self.conch_v1_encoder = conch_v1_encoder
        self.grounding_source = grounding_source
        self.adapter_lr = adapter_lr
        self.grounding_warmup_epochs = float(grounding_warmup_epochs)
        # Remember the user-supplied lambdas so warmup can restore them.
        self._target_lambda_g = float(combined_loss.lambda_g)
        self._target_lambda_f = float(combined_loss.lambda_f)

    # ── Loss ────────────────────────────────────────────────────────────────

    def compute_loss(
        self,
        model: GroundedVLM,
        inputs: dict,
        return_outputs: bool = False,
        **kwargs,
    ) -> Tensor | tuple[Tensor, dict]:
        """Compute CombinedLoss for one training step.

        Args:
            model: GroundedVLM instance (LoRA + adapter trainable, base frozen)
            inputs: collated batch from grounding_collate_fn — contains mixed
                    tensors (already on device) and list/numpy fields
            return_outputs: if True, also return the model output dict

        Returns:
            total loss scalar, or (total loss, model_output_dict) if return_outputs
        """
        slide_ids: list[str] = inputs["slide_ids"]
        vision_tokens_list: list[Tensor] = inputs["patch_embeddings_v15"]
        patch_v1_list: list[np.ndarray] = inputs["patch_embeddings_v1"]
        coords_list: list[np.ndarray] = inputs["coordinates"]
        input_ids: Tensor = inputs["input_ids"]
        attention_mask: Tensor = inputs["attention_mask"]
        labels: Tensor = inputs["labels"]
        sentences_list: list[list[str]] = inputs["sentences"]
        spans_list: list[list[tuple[int, int]]] = inputs["sentence_spans"]
        bcss_h5_paths: list[str | None] = inputs["bcss_h5_paths"]

        # Move vision tokens to adapter's device; other arrays stay on CPU/numpy
        adapter_device = next(model.adapter.parameters()).device
        vision_tokens = torch.stack(vision_tokens_list, dim=0).to(adapter_device)
        input_ids = input_ids.to(adapter_device)
        attention_mask = attention_mask.to(adapter_device)
        labels = labels.to(adapter_device)

        # Optional grounding warmup: caption-only for the first
        # `grounding_warmup_epochs` epochs so the adapter + LoRA learn the
        # language modelling task before grounding/faithfulness compete for
        # gradient budget. Lambdas are restored once the warmup window closes.
        warmup = float(getattr(self, "grounding_warmup_epochs", 0.0))
        if warmup > 0.0 and float(self.state.epoch or 0.0) < warmup:
            self.combined_loss.lambda_g = 0.0
            self.combined_loss.lambda_f = 0.0
        else:
            self.combined_loss.lambda_g = float(self._target_lambda_g)
            self.combined_loss.lambda_f = float(self._target_lambda_f)

        need_attn = self.combined_loss.lambda_g > 0
        out: dict = model(
            vision_tokens=vision_tokens,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_attentions=need_attn,
        )

        generated_attn: Tensor | None = None
        gt_attn: Tensor | None = None

        if need_attn and "text_to_vision_attn" in out:
            # out["text_to_vision_attn"]: (B, T, N_v)
            gen_list: list[Tensor] = []
            gt_list: list[Tensor] = []

            for b_idx in range(len(slide_ids)):
                slide_id = slide_ids[b_idx]
                patch_v1 = patch_v1_list[b_idx]  # (N_v, 512) numpy
                coords = coords_list[b_idx]  # (N_v, 2) numpy
                sentences = sentences_list[b_idx]
                spans = spans_list[b_idx]
                bcss_h5_path = bcss_h5_paths[b_idx]

                bcss_masks: dict | None = _load_bcss_masks(bcss_h5_path, len(patch_v1))

                for sentence, (span_start, span_end) in zip(sentences, spans, strict=False):
                    if span_start >= span_end or span_end > out["text_to_vision_attn"].shape[1]:
                        continue

                    # Average text-to-vision attention over sentence token positions → (N_v,)
                    sent_attn: Tensor = out["text_to_vision_attn"][
                        b_idx, span_start:span_end, :
                    ].mean(dim=0)

                    gt_dist, _ = build_target(
                        slide_id=slide_id,
                        sentence=sentence,
                        patch_embeddings_v1=patch_v1,
                        coordinates=coords,
                        conch_v1_encoder=self.conch_v1_encoder,
                        bcss_masks=bcss_masks,
                        pseudo_gt_cache=self.pseudo_gt_cache,
                        grounding_source=self.grounding_source,
                    )

                    gen_list.append(sent_attn)
                    gt_list.append(torch.from_numpy(gt_dist).to(adapter_device))

            if gen_list:
                generated_attn = torch.stack(gen_list, dim=0)  # (S, N_v)
                gt_attn = torch.stack(gt_list, dim=0)  # (S, N_v)

        # `out["logits"]` is over (N_v + T) positions (vision prefix + text);
        # `labels` is only the T text positions. Pad with -100 for the vision
        # prefix so cross-entropy sees matching sequence lengths.
        n_v = out["n_vision_tokens"]
        if n_v > 0:
            vis_lab = torch.full(
                (labels.size(0), n_v), -100, dtype=labels.dtype, device=labels.device
            )
            full_labels = torch.cat([vis_lab, labels], dim=1)
        else:
            full_labels = labels

        loss_dict = self.combined_loss(
            logits=out["logits"],
            labels=full_labels,
            generated_attn=generated_attn,
            gt_attn=gt_attn,
        )

        log_every = self.args.logging_steps
        if self.state.global_step % max(log_every, 1) == 0:
            self.log({f"train/{k}": v.item() for k, v in loss_dict.items()})

        total_loss: Tensor = loss_dict["total"]
        return (total_loss, out) if return_outputs else total_loss

    # ── Optimizer with per-group lr ──────────────────────────────────────────

    def create_optimizer(self) -> torch.optim.Optimizer:
        """Build AdamW with separate learning rates for adapter vs LoRA parameters.

        adapter_lr (default 1e-4) is typically 2× the LoRA lr (5e-5) to allow the
        projection head to adapt quickly while LoRA fine-tunes the LLM gradually.
        """
        adapter_params = list(self.model.adapter.parameters())
        adapter_ids = {id(p) for p in adapter_params}

        lora_params = [
            p for p in self.model.parameters() if id(p) not in adapter_ids and p.requires_grad
        ]

        param_groups = [
            {"params": adapter_params, "lr": self.adapter_lr},
            {"params": lora_params, "lr": self.args.learning_rate},
        ]

        self.optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=self.args.weight_decay,
            betas=(0.9, 0.999),
        )
        return self.optimizer


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _load_bcss_masks(bcss_h5_path: str | None, n_patches: int) -> dict | None:
    """Load precomputed BCSS patch masks from HDF5; truncate to n_patches.

    Args:
        bcss_h5_path: path to per-slide BCSS HDF5 written by BcssLoader.save_slide_h5
        n_patches: number of patches in the current slide (after max_vision_tokens cap)

    Returns:
        dict with keys mask_tumor / mask_stroma / mask_lymph / mask_necrosis /
        roi_coverage, each truncated to n_patches — or None if path is None / missing
    """
    if bcss_h5_path is None:
        return None
    try:
        with h5py.File(bcss_h5_path, "r") as f:
            masks = {k: f[k][:n_patches] for k in _BCSS_MASK_KEYS if k in f}
        return masks if masks else None
    except Exception as exc:  # corrupt / missing file
        logging.getLogger(__name__).warning(
            "Failed to load BCSS masks from %s: %s", bcss_h5_path, exc
        )
        return None


def _grounding_source_from_config(source_str: str) -> Literal["conch_only", "bcss_hybrid"]:
    """Map YAML grounding.source string to the build_target literal."""
    mapping = {
        "conch_only": "conch_only",
        "conch_pseudo": "conch_only",
        "bcss_hybrid": "bcss_hybrid",
        "hybrid": "bcss_hybrid",
    }
    return mapping.get(source_str, "conch_only")  # type: ignore[return-value]


# ─── CLI ──────────────────────────────────────────────────────────────────────


@app.command()
def main(
    config: str = typer.Option(..., help="Path to YAML config file"),
    run_name: str = typer.Option(..., help="W&B run name and checkpoint sub-directory"),
    resume: str | None = typer.Option(None, help="Checkpoint directory to resume from"),
    device: str = typer.Option("cuda", help="'cuda' or 'cpu'"),
) -> None:
    """Train PathoLens-VLM with grounding-aware loss.

    Example:
        uv run python -m patholens.training.train \\
            --config configs/grounding_v2.yaml \\
            --run-name grounded_v2_run1
    """
    from omegaconf import OmegaConf
    from transformers import AutoTokenizer

    cfg = load_config(config)
    logger.info("Config: %s", OmegaConf.to_yaml(cfg))

    # ── Tokenizer ────────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(cfg.llm_repo)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Model ────────────────────────────────────────────────────────────────
    if resume:
        logger.info("Resuming from checkpoint: %s", resume)
        model = GroundedVLM.from_pretrained(resume)
    else:
        model = GroundedVLM(
            llm_repo=cfg.llm_repo,
            vision_dim=cfg.vision_dim,
            llm_hidden_dim=cfg.llm_hidden_dim,
            lora_r=cfg.lora.r,
            lora_alpha=cfg.lora.alpha,
            lora_dropout=cfg.lora.dropout,
            lora_target_modules=list(cfg.lora.target_modules),
            load_in_4bit=cfg.get("load_in_4bit", True),
            attn_layer=cfg.get("attn_layer", 14),
        )
        model.load(device=device)

        if cfg.training.get("gradient_checkpointing", True) and hasattr(
            model.llm, "gradient_checkpointing_enable"
        ):
            model.llm.enable_input_require_grads()
            model.llm.gradient_checkpointing_enable()

    # ── CONCH v1 encoder for pseudo-GT (optional — only needed when grounding
    # falls back away from BCSS). Gracefully disabled when the HF account has
    # no access; build_target then returns a uniform fallback distribution,
    # which is fine as long as every training slide has BCSS coverage.
    from patholens.models.conch_encoder import ConchV1Encoder

    conch_v1 = None
    try:
        conch_v1 = ConchV1Encoder(device=device).load()
    except Exception as exc:
        logger.warning(
            "CONCH v1 unavailable (%s); proceeding without pseudo-GT "
            "(only viable when every slide has BCSS coverage).",
            exc,
        )

    # ── Pseudo-GT cache ──────────────────────────────────────────────────────
    cache_root: str = cfg.get("pseudo_gt_cache_dir", "data/processed")
    pseudo_gt_cache = PseudoGTCache(cache_dir=cache_root)

    # ── Datasets ─────────────────────────────────────────────────────────────
    bcss_masks_dir: str | None = cfg.get("bcss_masks_dir", None)
    max_vision_tokens: int = cfg.get("max_vision_tokens", 1024)

    train_dataset = GroundingSlideDataset(
        instruction_json=cfg.instruction_train,
        embeddings_dir=cfg.embeddings_dir,
        tokenizer=tokenizer,
        bcss_masks_dir=bcss_masks_dir,
        pseudo_gt_cache_dir=cache_root,
        max_vision_tokens=max_vision_tokens,
    )
    val_dataset = GroundingSlideDataset(
        instruction_json=cfg.instruction_val,
        embeddings_dir=cfg.embeddings_dir,
        tokenizer=tokenizer,
        bcss_masks_dir=bcss_masks_dir,
        pseudo_gt_cache_dir=cache_root,
        max_vision_tokens=max_vision_tokens,
    )
    logger.info("Train: %d slides, Val: %d slides", len(train_dataset), len(val_dataset))

    # ── Loss ─────────────────────────────────────────────────────────────────
    loss_type: str = cfg.loss.get("loss_type", "kl")
    combined_loss = CombinedLoss(
        lambda_grounding=float(cfg.loss.lambda_grounding),
        lambda_faithfulness=float(cfg.loss.lambda_faithfulness),
        grounding_loss_type=loss_type,  # type: ignore[arg-type]
        grounding_temperature=float(cfg.grounding.get("cosine_temperature", 0.1)),
    )
    grounding_source = _grounding_source_from_config(cfg.grounding.source)

    # ── Training arguments ───────────────────────────────────────────────────
    output_dir = f"{cfg.output_dir}/{run_name}"
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=int(cfg.training.epochs),
        per_device_train_batch_size=int(cfg.training.batch_size),
        gradient_accumulation_steps=int(cfg.training.gradient_accumulation),
        learning_rate=float(cfg.optimizer.lr_lora),
        weight_decay=float(cfg.optimizer.weight_decay),
        bf16=cfg.training.mixed_precision == "bf16",
        fp16=cfg.training.mixed_precision == "fp16",
        max_grad_norm=float(cfg.training.max_grad_norm),
        warmup_steps=int(cfg.scheduler.warmup_steps),
        eval_strategy="steps",
        eval_steps=int(cfg.training.eval_every),
        save_strategy="steps",
        save_steps=int(cfg.training.save_every),
        logging_steps=int(cfg.get("log_every", 10)),
        report_to="wandb",
        run_name=run_name,
        remove_unused_columns=False,
        dataloader_num_workers=0,
        gradient_checkpointing=False,  # managed manually above for 4-bit compat
        save_safetensors=False,  # Llama ties embed/lm_head; safetensors refuses
        save_total_limit=2,  # cap on-disk checkpoints during long runs
    )

    # ── Trainer ──────────────────────────────────────────────────────────────
    trainer = PathoTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=grounding_collate_fn,
        combined_loss=combined_loss,
        pseudo_gt_cache=pseudo_gt_cache,
        conch_v1_encoder=conch_v1,
        grounding_source=grounding_source,
        adapter_lr=float(cfg.optimizer.lr_adapter),
        grounding_warmup_epochs=float(cfg.loss.get("grounding_warmup_epochs", 0.0)),
    )

    trainer.train()

    final_dir = f"{output_dir}/final"
    model.save_pretrained(final_dir)
    logger.info("Training complete. Model saved to %s", final_dir)


if __name__ == "__main__":
    app()
