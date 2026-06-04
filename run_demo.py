from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from colofilm_ssm import ColoFiLMSSM, ColoFiLMSSMConfig
from colofilm_ssm.data import SyntheticColoDataset, collate_colofilm
from colofilm_ssm.losses import (
    discrete_time_survival_nll,
    mutation_bce_with_logits,
    transcriptome_mse,
)


def main() -> None:
    torch.manual_seed(7)
    config = ColoFiLMSSMConfig(
        wsi_dim=1024,
        model_dim=128,
        max_tokens=128,
        num_pathways=32,
        transcriptome_genes=64,
        num_ssm_blocks=2,
        survival_bins=8,
        dropout=0.05,
    )

    dataset = SyntheticColoDataset(config, num_samples=12, num_tokens=128)
    loader = DataLoader(dataset, batch_size=3, shuffle=True, collate_fn=collate_colofilm)

    model = ColoFiLMSSM(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    model.train()
    for step, batch in enumerate(loader):
        outputs = model(
            batch["wsi_tokens"],
            clinical_prompts=batch["clinical_prompts"],
            pathway_scores=batch["pathway_scores"],
            pathway_mask=batch["pathway_mask"],
            task="all",
            return_aux=True,
        )

        survival_loss = discrete_time_survival_nll(
            outputs["hazards"],
            batch["time_bins"],
            batch["events"],
        )
        mutation_loss = mutation_bce_with_logits(outputs["mutation_logits"], batch["mutation_labels"])
        expression_loss = transcriptome_mse(outputs["expression"], batch["expression"])
        loss = survival_loss + mutation_loss + expression_loss + outputs["auxiliary"]["total"]

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        print(f"step={step} loss={loss.item():.4f}")
        break

    model.eval()
    with torch.no_grad():
        batch = next(iter(loader))
        missing_context_outputs = model(
            batch["wsi_tokens"],
            clinical_prompts=None,
            pathway_scores=None,
            task="all",
        )
        print("hazards", tuple(missing_context_outputs["hazards"].shape))
        print("mutation_prob", tuple(missing_context_outputs["mutation_prob"].shape))
        print("expression", tuple(missing_context_outputs["expression"].shape))


if __name__ == "__main__":
    main()

