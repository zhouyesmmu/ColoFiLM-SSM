from __future__ import annotations

import torch
import torch.nn.functional as F


def discrete_time_survival_nll(
    hazards: torch.Tensor,
    time_bins: torch.Tensor,
    events: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    if hazards.ndim != 2:
        raise ValueError("hazards must have shape [B, T].")

    hazards = hazards.clamp(min=eps, max=1.0 - eps)
    time_bins = time_bins.long().clamp(min=0, max=hazards.size(1) - 1)
    events = events.float()

    survival = torch.cumprod(1.0 - hazards, dim=1)
    previous_survival = torch.cat(
        [torch.ones_like(survival[:, :1]), survival[:, :-1]],
        dim=1,
    )

    batch_index = torch.arange(hazards.size(0), device=hazards.device)
    event_likelihood = previous_survival[batch_index, time_bins] * hazards[batch_index, time_bins]
    censor_likelihood = survival[batch_index, time_bins]

    likelihood = torch.where(events > 0.5, event_likelihood, censor_likelihood)
    return -torch.log(likelihood.clamp_min(eps)).mean()


def mutation_bce_with_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    pos_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(
        logits,
        labels.float(),
        pos_weight=pos_weight,
    )


def transcriptome_mse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(prediction, target.float())
