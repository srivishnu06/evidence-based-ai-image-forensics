
import torch
import torch.nn as nn
import torch.nn.functional as F


class BCEWithRegionDistributionLoss(nn.Module):
    """BCE with spatial-distribution supervision."""

    def __init__(self, distribution_weight=0.1):
        super().__init__()

        if distribution_weight < 0:
            raise ValueError(
                "distribution_weight must be non-negative"
            )

        self.distribution_weight = float(
            distribution_weight
        )

    def forward(self, logits, targets):
        if logits.shape != targets.shape or logits.ndim != 4:
            raise ValueError(
                "Expected matching [B, C, H, W] tensors"
            )

        logits = logits.float()
        targets = targets.float()

        if not torch.isfinite(logits).all():
            raise ValueError("Non-finite logits")

        if not torch.isfinite(targets).all():
            raise ValueError("Non-finite targets")

        if torch.any(targets < 0) or torch.any(targets > 1):
            raise ValueError("Targets must be between 0 and 1")

        bce = F.binary_cross_entropy_with_logits(
            logits, targets
        )

        # Convert spatial targets into distributions.
        # Fractional polygon coverage is preserved.
        gt = targets.flatten(2)
        scores = logits.flatten(2)

        gt_mass = gt.sum(dim=-1)
        present = gt_mass > 0

        if not present.any():
            return bce

        gt_distribution = gt / gt_mass.clamp_min(1e-8).unsqueeze(-1)

        predicted_log_distribution = F.log_softmax(
            scores, dim=-1
        )

        # KL divergence between annotated region
        # distribution and predicted spatial distribution.
        kl = (
            gt_distribution
            * (
                gt_distribution.clamp_min(1e-8).log()
                - predicted_log_distribution
            )
        ).sum(dim=-1)

        # Give categories present in the batch equal
        # influence instead of allowing common labels
        # to dominate the auxiliary loss.
        class_counts = present.sum(dim=0)
        class_means = (
            (kl * present).sum(dim=0)
            / class_counts.clamp_min(1)
        )

        distribution_loss = class_means[
            class_counts > 0
        ].mean()

        return (
            bce
            + self.distribution_weight * distribution_loss
        )
