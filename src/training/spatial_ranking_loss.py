
import torch
import torch.nn as nn
import torch.nn.functional as F


class BCEWithSpatialRankingLoss(nn.Module):
    """BCE plus a foreground-versus-background peak-ranking loss."""

    def __init__(self, ranking_weight=0.1):
        super().__init__()
        if ranking_weight < 0:
            raise ValueError("ranking_weight must be non-negative")
        self.ranking_weight = float(ranking_weight)

    def forward(self, logits, targets):
        logits = logits.float()
        targets = targets.float()

        if logits.shape != targets.shape or logits.ndim != 4:
            raise ValueError("Expected matching [B,C,H,W] tensors")

        bce = F.binary_cross_entropy_with_logits(
            logits, targets
        )

        positive = targets > 0
        per_class_losses = []

        for c in range(logits.shape[1]):
            pos = positive[:, c].flatten(1)
            neg = ~pos

            valid = pos.any(dim=1) & neg.any(dim=1)
            if not valid.any():
                continue

            scores = logits[:, c][valid].flatten(1)
            pos = pos[valid]
            neg = neg[valid]

            positive_peak = scores.masked_fill(
                ~pos, -1e4
            ).amax(dim=1)

            negative_peak = scores.masked_fill(
                ~neg, -1e4
            ).amax(dim=1)

            per_class_losses.append(
                F.softplus(
                    negative_peak - positive_peak
                ).mean()
            )

        if per_class_losses:
            ranking = torch.stack(
                per_class_losses
            ).mean()
        else:
            ranking = logits.sum() * 0.0

        return bce + self.ranking_weight * ranking
