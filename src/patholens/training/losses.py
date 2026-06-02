"""Training losses: caption + grounding + faithfulness."""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class CaptionLoss(nn.Module):
    """Standard LM cross-entropy loss with mask for vision/padding tokens."""

    def __init__(self, ignore_index: int = -100) -> None:
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, logits: Tensor, labels: Tensor) -> Tensor:
        """
        Args:
            logits: (B, seq_len, vocab_size)
            labels: (B, seq_len), -100 for masked positions

        Returns:
            scalar loss
        """
        # Shift for causal LM: predict next token
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        return F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=self.ignore_index,
        )


class GroundingLoss(nn.Module):
    """Grounding loss between generated text→vision attention and GT attention.

    Ground truth attention comes from the hybrid router in grounding_targets.py:
        - BCSS pixel mask (where ROI covers the patch and concept matches)
        - CONCH v1 pseudo-GT (everywhere else)

    Two variants:
        "kl"     — KL(gt || generated), sharpened by temperature (baseline)
        "cosine" — 1 − cosine_similarity(gt, generated) (ablation; softer for noisy teacher)
    """

    def __init__(
        self,
        temperature: float = 0.1,
        loss_type: Literal["kl", "cosine"] = "kl",
    ) -> None:
        super().__init__()
        self.temperature = temperature
        self.loss_type = loss_type

    def forward(
        self,
        generated_attn: Tensor,  # (N_sentences, N_vision_tokens)
        gt_attn: Tensor,  # (N_sentences, N_vision_tokens)
    ) -> Tensor:
        """
        Args:
            generated_attn: model attention over vision tokens per sentence
            gt_attn: ground truth attention distribution per sentence

        Returns:
            scalar grounding loss
        """
        eps = 1e-8
        # Renormalise the generated attention slice to a proper distribution
        # over N_v patches (attention weights summed over heads do not sum to 1
        # because they share probability mass with text-token positions).
        gen_probs = generated_attn / generated_attn.sum(dim=-1, keepdim=True).clamp(min=eps)
        # gt is already normalised (sums to 1) by `build_target`; just clamp.
        gt_probs = gt_attn / gt_attn.sum(dim=-1, keepdim=True).clamp(min=eps)

        if self.loss_type == "kl":
            # KL(gt || generated). Temperature sharpens both distributions before
            # comparison (lower T -> peakier, harder match).
            log_gen = torch.log(gen_probs.clamp(min=eps))
            if self.temperature != 1.0:
                log_gen = log_gen / self.temperature
                log_gen = log_gen - log_gen.logsumexp(dim=-1, keepdim=True)
                gt_sharp = (gt_probs.clamp(min=eps).log() / self.temperature).softmax(dim=-1)
            else:
                gt_sharp = gt_probs
            return F.kl_div(log_gen, gt_sharp, reduction="batchmean")
        else:
            # 1 − cosine similarity over the (already non-negative) distributions
            gen_norm = F.normalize(gen_probs, dim=-1)
            gt_norm = F.normalize(gt_probs, dim=-1)
            return 1.0 - (gen_norm * gt_norm).sum(dim=-1).mean()


class FaithfulnessRegularizer(nn.Module):
    """Encourage attention concentration: low entropy per sentence.

    Input is the *raw* per-sentence attention slice (sum over patches is small
    because attention is shared between vision and text positions in the
    transformer's softmax). We renormalise to a proper probability distribution
    over the N_v patches before taking entropy — otherwise softmax of near-zero
    values collapses to uniform and entropy locks at log(N_v).

    Adapted from ACMIL branch concentration loss.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, sentence_attentions: Tensor) -> Tensor:
        """
        Args:
            sentence_attentions: (N_sentences, N_vision_tokens) — raw attention
                weights summed over heads, non-negative, row-sum < 1 (some
                attention went to text positions).

        Returns:
            mean entropy across sentences (to be minimized; lower = peakier).
        """
        eps = 1e-8
        row_sum = sentence_attentions.sum(dim=-1, keepdim=True).clamp(min=eps)
        probs = sentence_attentions / row_sum  # proper distribution over N_v
        entropy = -(probs * torch.log(probs + eps)).sum(dim=-1)
        return entropy.mean()


class CombinedLoss(nn.Module):
    """L = L_caption + λ_g · L_grounding + λ_f · L_faithfulness."""

    def __init__(
        self,
        lambda_grounding: float = 0.0,
        lambda_faithfulness: float = 0.0,
        grounding_loss_type: Literal["kl", "cosine"] = "kl",
        grounding_temperature: float = 0.1,
    ) -> None:
        super().__init__()
        self.caption_loss = CaptionLoss()
        self.grounding_loss = GroundingLoss(
            temperature=grounding_temperature, loss_type=grounding_loss_type
        )
        self.faithfulness_reg = FaithfulnessRegularizer()
        self.lambda_g = lambda_grounding
        self.lambda_f = lambda_faithfulness

    def forward(
        self,
        logits: Tensor,
        labels: Tensor,
        generated_attn: Tensor | None = None,
        gt_attn: Tensor | None = None,
    ) -> dict[str, Tensor]:
        loss_dict = {}
        loss_dict["caption"] = self.caption_loss(logits, labels)
        total = loss_dict["caption"]

        if self.lambda_g > 0 and generated_attn is not None and gt_attn is not None:
            loss_dict["grounding"] = self.grounding_loss(generated_attn, gt_attn)
            total = total + self.lambda_g * loss_dict["grounding"]

        if self.lambda_f > 0 and generated_attn is not None:
            # FaithfulnessRegularizer handles renormalisation internally —
            # pass the raw attention slice (do NOT softmax it first; that
            # collapses near-zero values to a uniform distribution and the
            # entropy locks at log(N_v)).
            loss_dict["faithfulness"] = self.faithfulness_reg(generated_attn)
            total = total + self.lambda_f * loss_dict["faithfulness"]

        loss_dict["total"] = total
        return loss_dict
