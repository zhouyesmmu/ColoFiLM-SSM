from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import Dataset

from .config import ColoFiLMSSMConfig


class SyntheticColoDataset(Dataset):
    def __init__(
        self,
        config: ColoFiLMSSMConfig,
        num_samples: int = 16,
        num_tokens: int = 128,
        missing_context_prob: float = 0.25,
        seed: int = 31,
    ) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.config = config
        self.wsi_tokens = torch.randn(
            num_samples,
            num_tokens,
            config.wsi_dim,
            generator=generator,
        )
        self.pathway_scores = torch.randn(num_samples, config.num_pathways, generator=generator)
        self.pathway_available = torch.rand(num_samples, generator=generator) > missing_context_prob
        self.clinical_available = torch.rand(num_samples, generator=generator) > missing_context_prob

        signal = self.wsi_tokens.mean(dim=(1, 2)) + 0.25 * self.pathway_scores[:, :4].mean(dim=1)
        event_prob = torch.sigmoid(signal)
        self.events = torch.bernoulli(event_prob, generator=generator)
        self.time_bins = torch.randint(0, config.survival_bins, (num_samples,), generator=generator)

        mutation_logits = torch.randn(num_samples, config.mutation_genes, generator=generator)
        mutation_logits[:, 0] = mutation_logits[:, 0] + signal
        self.mutation_labels = torch.bernoulli(torch.sigmoid(mutation_logits), generator=generator)

        projection = torch.randn(
            config.num_pathways,
            config.transcriptome_genes,
            generator=generator,
        ) / config.num_pathways**0.5
        self.expression = self.pathway_scores @ projection
        self.expression = self.expression + 0.05 * torch.randn(
            self.expression.shape,
            generator=generator,
        )

    def __len__(self) -> int:
        return self.wsi_tokens.size(0)

    def __getitem__(self, index: int) -> dict[str, Any]:
        prompt = (
            f"age: {50 + index % 30} | sex: {'female' if index % 2 else 'male'} | "
            f"stage: {1 + index % 4} | status: Unknown"
        )
        return {
            "wsi_tokens": self.wsi_tokens[index],
            "clinical_prompt": prompt if self.clinical_available[index] else None,
            "pathway_scores": self.pathway_scores[index] if self.pathway_available[index] else None,
            "num_pathways": self.config.num_pathways,
            "time_bin": self.time_bins[index],
            "event": self.events[index],
            "mutation_labels": self.mutation_labels[index],
            "expression": self.expression[index],
        }


def collate_colofilm(batch: list[dict[str, Any]]) -> dict[str, Any]:
    wsi_tokens = torch.stack([item["wsi_tokens"] for item in batch], dim=0)
    clinical_prompts = [item["clinical_prompt"] for item in batch]

    pathway_mask = torch.tensor([item["pathway_scores"] is not None for item in batch], dtype=torch.bool)
    num_pathways = int(batch[0]["num_pathways"])

    pathway_scores = []
    for item in batch:
        if item["pathway_scores"] is None:
            pathway_scores.append(torch.zeros(num_pathways, dtype=wsi_tokens.dtype))
        else:
            pathway_scores.append(item["pathway_scores"])

    return {
        "wsi_tokens": wsi_tokens,
        "clinical_prompts": clinical_prompts,
        "pathway_scores": torch.stack(pathway_scores, dim=0),
        "pathway_mask": pathway_mask,
        "time_bins": torch.stack([item["time_bin"] for item in batch], dim=0),
        "events": torch.stack([item["event"] for item in batch], dim=0),
        "mutation_labels": torch.stack([item["mutation_labels"] for item in batch], dim=0),
        "expression": torch.stack([item["expression"] for item in batch], dim=0),
    }
