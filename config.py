from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ColoFiLMSSMConfig:
    wsi_dim: int = 1024
    model_dim: int = 256
    max_tokens: int = 4000

    prompt_buckets: int = 4096
    num_pathways: int = 128
    pathway_layers: int = 1

    num_ssm_blocks: int = 2
    ssm_backend: str = "fallback"
    ssm_conv_kernel: int = 4
    mamba_state_dim: int = 16
    mamba_expand: int = 2
    dropout: float = 0.1

    fusion_layers: int = 1
    fusion_heads: int = 4

    survival_bins: int = 8
    mutation_genes: int = 10
    transcriptome_genes: int = 256

    aux_weight: float = 0.05
