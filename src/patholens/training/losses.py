"""Training losses: caption + grounding + faithfulness."""

from __future__ import annotations

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
    """KL divergence between generated text→vision attention and ground truth attention.

    Ground truth attention can come from:
        (a) CONCHv1.5 zero-shot pseudo-supervision (cosine similarity)
        (b) CAMELYON16 pixel mask (explicit supervision)

    To be implemented in Hafta 4.
    """

    def __init__(self, temperature: float = 0.1) -> None:
        super().__init__()
        self.temperature = temperature

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
            scalar KL divergence loss
        """
        # KL(gt || generated)
        return F.kl_div(
            F.log_softmax(generated_attn / self.temperature, dim=-1),
            F.softmax(gt_attn / self.temperature, dim=-1),
            reduction="batchmean",
        )


class FaithfulnessRegularizer(nn.Module):
    """Encourage attention concentration: low entropy per sentence.

    Adapted from ACMIL branch concentration loss.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, sentence_attentions: Tensor) -> Tensor:
        """
        Args:
            sentence_attentions: (N_sentences, N_vision_tokens), already softmax'd

        Returns:
            mean entropy across sentences (to be minimized)
        """
        eps = 1e-8
        entropy = -(sentence_attentions * torch.log(sentence_attentions + eps)).sum(dim=-1)
        return entropy.mean()


class CombinedLoss(nn.Module):
    """L = L_caption + λ_g · L_grounding + λ_f · L_faithfulness."""

    def __init__(
        self,
        lambda_grounding: float = 0.0,
        lambda_faithfulness: float = 0.0,
    ) -> None:
        super().__init__()
        self.caption_loss = CaptionLoss()
        self.grounding_loss = GroundingLoss()
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
            sent_attn_softmax = torch.softmax(generated_attn, dim=-1)
            loss_dict["faithfulness"] = self.faithfulness_reg(sent_attn_softmax)
            total = total + self.lambda_f * loss_dict["faithfulness"]

        loss_dict["total"] = total
        return loss_dict
