from __future__ import annotations

import hashlib
import math
from itertools import combinations
from typing import Dict, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from .config import ColoFiLMSSMConfig

try:
    from mamba_ssm import Mamba as _ExternalMamba
except Exception:
    _ExternalMamba = None


def _fixed_normal(shape: tuple[int, ...], seed: int, scale: float) -> torch.Tensor:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return torch.randn(shape, generator=generator) * scale


class FrozenPromptEncoder(nn.Module):
    def __init__(self, buckets: int, embed_dim: int, seed: int = 17) -> None:
        super().__init__()
        table = _fixed_normal((buckets, embed_dim), seed, 1.0 / math.sqrt(embed_dim))
        self.register_buffer("table", table)
        self.norm = nn.LayerNorm(embed_dim, elementwise_affine=False)

    @property
    def buckets(self) -> int:
        return self.table.size(0)

    @staticmethod
    def _tokens(prompt: str) -> list[str]:
        return [token for token in prompt.lower().replace("|", " ").split() if token]

    def _bucket(self, token: str) -> int:
        digest = hashlib.sha1(token.encode("utf-8")).hexdigest()
        return int(digest[:8], 16) % self.buckets

    def forward(self, prompts: Sequence[str]) -> torch.Tensor:
        device = self.table.device
        vectors = []
        for prompt in prompts:
            tokens = self._tokens(prompt)
            if not tokens:
                vectors.append(torch.zeros(self.table.size(1), device=device, dtype=self.table.dtype))
                continue
            indices = torch.tensor([self._bucket(token) for token in tokens], device=device)
            vectors.append(self.table.index_select(0, indices).mean(dim=0))
        return self.norm(torch.stack(vectors, dim=0))


class PathwayEncoder(nn.Module):
    def __init__(self, config: ColoFiLMSSMConfig) -> None:
        super().__init__()
        self.pathway_embedding = nn.Parameter(torch.empty(config.num_pathways, config.model_dim))
        self.value_projection = nn.Linear(1, config.model_dim)

        layer = nn.TransformerEncoderLayer(
            d_model=config.model_dim,
            nhead=config.fusion_heads,
            dim_feedforward=config.model_dim * 2,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=config.pathway_layers,
            enable_nested_tensor=False,
        )
        self.norm = nn.LayerNorm(config.model_dim)
        nn.init.trunc_normal_(self.pathway_embedding, std=0.02)

    def forward(self, pathway_scores: torch.Tensor) -> torch.Tensor:
        if pathway_scores.ndim != 2:
            raise ValueError("pathway_scores must have shape [B, P].")
        pathway_tokens = self.pathway_embedding.unsqueeze(0) + self.value_projection(
            pathway_scores.unsqueeze(-1)
        )
        encoded = self.encoder(pathway_tokens)
        return self.norm(encoded.mean(dim=1))


class FiLMConditioner(nn.Module):
    def __init__(self, config: ColoFiLMSSMConfig) -> None:
        super().__init__()
        dim = config.model_dim
        self.net = nn.Sequential(
            nn.LayerNorm(dim * 2),
            nn.Linear(dim * 2, dim * 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(dim * 2, dim * 2),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(
        self,
        clinical_embedding: torch.Tensor,
        pathway_embedding: torch.Tensor,
        clinical_mask: torch.Tensor,
        pathway_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        context = torch.cat([clinical_embedding, pathway_embedding], dim=-1)
        delta_gamma, beta = self.net(context).chunk(2, dim=-1)
        gamma = 1.0 + delta_gamma

        no_context = ~(clinical_mask | pathway_mask)
        if no_context.any():
            gamma = gamma.clone()
            beta = beta.clone()
            gamma[no_context] = 1.0
            beta[no_context] = 0.0
        return gamma, beta


class SSMBlock(nn.Module):
    def __init__(self, config: ColoFiLMSSMConfig) -> None:
        super().__init__()
        dim = config.model_dim
        self.backend = config.ssm_backend.lower()
        if self.backend not in {"fallback", "mamba"}:
            raise ValueError("ssm_backend must be either 'fallback' or 'mamba'.")
        if self.backend == "mamba" and _ExternalMamba is None:
            raise ImportError("Install mamba-ssm or set ssm_backend='fallback'.")

        self.norm = nn.LayerNorm(dim)
        if self.backend == "mamba":
            self.in_projection = None
            self.depthwise_conv = None
            self.decay_logit = None
            self.mixer = _ExternalMamba(
                d_model=dim,
                d_state=config.mamba_state_dim,
                d_conv=config.ssm_conv_kernel,
                expand=config.mamba_expand,
            )
        else:
            self.in_projection = nn.Linear(dim, dim * 2)
            self.depthwise_conv = nn.Conv1d(
                dim,
                dim,
                kernel_size=config.ssm_conv_kernel,
                padding=config.ssm_conv_kernel - 1,
                groups=dim,
            )
            self.decay_logit = nn.Parameter(torch.full((dim,), -1.0))
            self.mixer = None
        self.out_projection = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(config.dropout)
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        residual = tokens
        normalized = self.norm(tokens)
        if self.backend == "mamba":
            y = self.mixer(normalized)
            tokens = residual + self.dropout(self.out_projection(y))
            tokens = tokens + self.dropout(self.ffn(tokens))
            return tokens

        mixed, gate = self.in_projection(normalized).chunk(2, dim=-1)
        mixed = self.depthwise_conv(mixed.transpose(1, 2))
        mixed = mixed[:, :, : tokens.size(1)].transpose(1, 2)

        decay = torch.sigmoid(self.decay_logit).view(1, -1)
        input_scale = 1.0 - decay
        state = torch.zeros(tokens.size(0), tokens.size(2), device=tokens.device, dtype=tokens.dtype)
        outputs = []
        for step in mixed.unbind(dim=1):
            state = decay * state + input_scale * step
            outputs.append(state)

        y = torch.stack(outputs, dim=1) * torch.sigmoid(gate)
        tokens = residual + self.dropout(self.out_projection(y))
        tokens = tokens + self.dropout(self.ffn(tokens))
        return tokens


class AttentionPool(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Tanh(),
            nn.Linear(dim, 1),
        )

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scores = self.attention(tokens).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        pooled = torch.bmm(weights.unsqueeze(1), tokens).squeeze(1)
        return pooled, weights


class ContextFusion(nn.Module):
    def __init__(self, config: ColoFiLMSSMConfig) -> None:
        super().__init__()
        dim = config.model_dim
        self.type_embedding = nn.Parameter(torch.empty(3, dim))
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=config.fusion_heads,
            dim_feedforward=dim * 2,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=config.fusion_layers,
            enable_nested_tensor=False,
        )
        self.norm = nn.LayerNorm(dim)
        nn.init.trunc_normal_(self.type_embedding, std=0.02)

    def forward(
        self,
        wsi_embedding: torch.Tensor,
        clinical_embedding: torch.Tensor,
        pathway_embedding: torch.Tensor,
        clinical_mask: torch.Tensor,
        pathway_mask: torch.Tensor,
    ) -> torch.Tensor:
        tokens = torch.stack([wsi_embedding, clinical_embedding, pathway_embedding], dim=1)
        tokens = tokens + self.type_embedding.unsqueeze(0)

        key_padding_mask = torch.stack(
            [
                torch.zeros_like(clinical_mask, dtype=torch.bool),
                ~clinical_mask,
                ~pathway_mask,
            ],
            dim=1,
        )
        encoded = self.encoder(tokens, src_key_padding_mask=key_padding_mask)
        valid = (~key_padding_mask).to(encoded.dtype).unsqueeze(-1)
        fused = (encoded * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
        return self.norm(fused)


class InvariantSpecificRegularizer(nn.Module):
    def __init__(self, config: ColoFiLMSSMConfig) -> None:
        super().__init__()
        dim = config.model_dim
        latent_dim = max(32, dim // 2)
        self.weight = config.aux_weight
        self.modalities = ("wsi", "clinical", "pathway")
        self.invariant = nn.ModuleDict(
            {name: nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, latent_dim)) for name in self.modalities}
        )
        self.specific = nn.ModuleDict(
            {name: nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, latent_dim)) for name in self.modalities}
        )

    def forward(
        self,
        embeddings: Dict[str, torch.Tensor],
        available: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        invariant = {
            name: F.normalize(self.invariant[name](embeddings[name]), dim=-1)
            for name in self.modalities
        }
        specific = {
            name: F.normalize(self.specific[name](embeddings[name]), dim=-1)
            for name in self.modalities
        }

        zero = next(iter(embeddings.values())).new_tensor(0.0)
        sim_terms = []
        for left, right in combinations(self.modalities, 2):
            mask = available[left] & available[right]
            if mask.any():
                sim_terms.append(1.0 - F.cosine_similarity(invariant[left][mask], invariant[right][mask]).mean())

        diff_terms = []
        for name in self.modalities:
            mask = available[name]
            if mask.any():
                corr = F.cosine_similarity(invariant[name][mask], specific[name][mask])
                diff_terms.append((corr * corr).mean())

        similarity = torch.stack(sim_terms).mean() if sim_terms else zero
        difference = torch.stack(diff_terms).mean() if diff_terms else zero
        total = self.weight * (similarity + difference)
        return {"similarity": similarity, "difference": difference, "total": total}


class ColoFiLMSSM(nn.Module):
    def __init__(self, config: Optional[ColoFiLMSSMConfig] = None) -> None:
        super().__init__()
        self.config = config or ColoFiLMSSMConfig()
        dim = self.config.model_dim

        self.wsi_projection = nn.Sequential(
            nn.Linear(self.config.wsi_dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
        )
        self.clinical_encoder = FrozenPromptEncoder(self.config.prompt_buckets, dim)
        self.pathway_encoder = PathwayEncoder(self.config)
        self.film = FiLMConditioner(self.config)
        self.ssm_blocks = nn.ModuleList([SSMBlock(self.config) for _ in range(self.config.num_ssm_blocks)])
        self.pool = AttentionPool(dim)
        self.fusion = ContextFusion(self.config)
        self.regularizer = InvariantSpecificRegularizer(self.config)

        self.survival_head = nn.Linear(dim, self.config.survival_bins)
        self.mutation_head = nn.Linear(dim, self.config.mutation_genes)
        self.transcriptome_head = nn.Linear(dim, self.config.transcriptome_genes)

    def _cap_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        max_tokens = self.config.max_tokens
        if max_tokens <= 0 or tokens.size(1) <= max_tokens:
            return tokens
        indices = torch.linspace(0, tokens.size(1) - 1, max_tokens, device=tokens.device).long()
        return tokens.index_select(1, indices)

    def _encode_clinical(
        self,
        clinical_prompts: Optional[Sequence[Optional[str]]],
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if clinical_prompts is None:
            embedding = torch.zeros(batch_size, self.config.model_dim, device=device, dtype=dtype)
            mask = torch.zeros(batch_size, device=device, dtype=torch.bool)
            return embedding, mask
        if len(clinical_prompts) != batch_size:
            raise ValueError("clinical_prompts length must match batch size.")

        mask = torch.tensor([prompt is not None for prompt in clinical_prompts], device=device, dtype=torch.bool)
        safe_prompts = [prompt if prompt is not None else "" for prompt in clinical_prompts]
        embedding = self.clinical_encoder(safe_prompts).to(device=device, dtype=dtype)
        embedding = embedding * mask.to(dtype).unsqueeze(-1)
        return embedding, mask

    def _encode_pathway(
        self,
        pathway_scores: Optional[torch.Tensor],
        pathway_mask: Optional[torch.Tensor],
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if pathway_scores is None:
            embedding = torch.zeros(batch_size, self.config.model_dim, device=device, dtype=dtype)
            mask = torch.zeros(batch_size, device=device, dtype=torch.bool)
            return embedding, mask
        if pathway_scores.shape != (batch_size, self.config.num_pathways):
            raise ValueError("pathway_scores must have shape [B, num_pathways].")

        scores = pathway_scores.to(device=device, dtype=dtype)
        mask = (
            torch.ones(batch_size, device=device, dtype=torch.bool)
            if pathway_mask is None
            else pathway_mask.to(device=device, dtype=torch.bool)
        )
        embedding = self.pathway_encoder(scores).to(dtype=dtype)
        embedding = embedding * mask.to(dtype).unsqueeze(-1)
        return embedding, mask

    def forward(
        self,
        wsi_tokens: torch.Tensor,
        clinical_prompts: Optional[Sequence[Optional[str]]] = None,
        pathway_scores: Optional[torch.Tensor] = None,
        pathway_mask: Optional[torch.Tensor] = None,
        task: str = "survival",
        return_aux: bool = False,
    ) -> Dict[str, torch.Tensor | Dict[str, torch.Tensor]]:
        if wsi_tokens.ndim != 3:
            raise ValueError("wsi_tokens must have shape [B, N, D].")
        if wsi_tokens.size(-1) != self.config.wsi_dim:
            raise ValueError("Last dimension of wsi_tokens must match config.wsi_dim.")

        wsi_tokens = self._cap_tokens(wsi_tokens)
        batch_size = wsi_tokens.size(0)
        device = wsi_tokens.device
        dtype = wsi_tokens.dtype

        clinical_embedding, clinical_available = self._encode_clinical(
            clinical_prompts, batch_size, device, dtype
        )
        pathway_embedding, pathway_available = self._encode_pathway(
            pathway_scores, pathway_mask, batch_size, device, dtype
        )

        tokens = self.wsi_projection(wsi_tokens)
        gamma, beta = self.film(
            clinical_embedding,
            pathway_embedding,
            clinical_available,
            pathway_available,
        )
        tokens = tokens * gamma.unsqueeze(1) + beta.unsqueeze(1)

        for block in self.ssm_blocks:
            tokens = block(tokens)

        wsi_embedding, attention = self.pool(tokens)
        fused = self.fusion(
            wsi_embedding,
            clinical_embedding,
            pathway_embedding,
            clinical_available,
            pathway_available,
        )

        outputs: Dict[str, torch.Tensor | Dict[str, torch.Tensor]] = {
            "fused_embedding": fused,
            "wsi_embedding": wsi_embedding,
            "attention": attention,
        }

        if task in {"survival", "all"}:
            survival_logits = self.survival_head(fused)
            outputs["survival_logits"] = survival_logits
            outputs["hazards"] = torch.sigmoid(survival_logits)
        if task in {"mutation", "all"}:
            mutation_logits = self.mutation_head(fused)
            outputs["mutation_logits"] = mutation_logits
            outputs["mutation_prob"] = torch.sigmoid(mutation_logits)
        if task in {"transcriptome", "all"}:
            outputs["expression"] = self.transcriptome_head(fused)
        if task not in {"survival", "mutation", "transcriptome", "all"}:
            raise ValueError("task must be one of: survival, mutation, transcriptome, all.")

        if return_aux:
            available = {
                "wsi": torch.ones(batch_size, device=device, dtype=torch.bool),
                "clinical": clinical_available,
                "pathway": pathway_available,
            }
            embeddings = {
                "wsi": wsi_embedding,
                "clinical": clinical_embedding,
                "pathway": pathway_embedding,
            }
            outputs["auxiliary"] = self.regularizer(embeddings, available)
        return outputs
