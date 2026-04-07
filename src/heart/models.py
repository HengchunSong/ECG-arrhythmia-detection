from __future__ import annotations

import torch
from torch import nn


class DepthwiseSeparableConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.depthwise = nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=False,
        )
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(x))


class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Conv1d(channels, hidden, kernel_size=1),
            nn.SiLU(),
            nn.Conv1d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.fc(self.pool(x))
        return x * scale


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        pool_stride: int = 2,
        dropout: float = 0.1,
        use_se: bool = False,
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            DepthwiseSeparableConv1d(in_channels, out_channels, kernel_size=kernel_size),
            nn.BatchNorm1d(out_channels),
            nn.SiLU(),
            DepthwiseSeparableConv1d(out_channels, out_channels, kernel_size=kernel_size),
            nn.BatchNorm1d(out_channels),
            nn.SiLU(),
        )
        self.se = SEBlock(out_channels) if use_se else nn.Identity()
        self.dropout = nn.Dropout(dropout)
        self.pool = nn.MaxPool1d(kernel_size=pool_stride, stride=pool_stride)
        self.shortcut = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
            if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.block(x)
        out = self.se(out)
        out = out + residual[..., :out.shape[-1]]
        out = self.dropout(out)
        return self.pool(out)


class TemporalAttention(nn.Module):
    def __init__(self, channels: int, heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(
            nn.Linear(channels, channels * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels * 2, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq = x.transpose(1, 2)
        attn_in = self.norm1(seq)
        attn_out, _ = self.attn(attn_in, attn_in, attn_in, need_weights=False)
        seq = seq + attn_out
        seq = seq + self.ffn(self.norm2(seq))
        return seq.transpose(1, 2)


class InceptionBlock1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        branch_channels: int = 24,
        bottleneck_channels: int = 32,
        dropout: float = 0.1,
        use_se: bool = True,
    ) -> None:
        super().__init__()
        self.bottleneck = (
            nn.Conv1d(in_channels, bottleneck_channels, kernel_size=1, bias=False)
            if in_channels != bottleneck_channels else nn.Identity()
        )
        self.branch_k9 = nn.Conv1d(
            bottleneck_channels,
            branch_channels,
            kernel_size=9,
            padding=4,
            bias=False,
        )
        self.branch_k19 = nn.Conv1d(
            bottleneck_channels,
            branch_channels,
            kernel_size=19,
            padding=9,
            bias=False,
        )
        self.branch_k39 = nn.Conv1d(
            bottleneck_channels,
            branch_channels,
            kernel_size=39,
            padding=19,
            bias=False,
        )
        self.branch_pool = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, branch_channels, kernel_size=1, bias=False),
        )
        merged_channels = branch_channels * 4
        self.merge = nn.Sequential(
            nn.BatchNorm1d(merged_channels),
            nn.SiLU(),
            nn.Conv1d(merged_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.SiLU(),
        )
        self.se = SEBlock(out_channels) if use_se else nn.Identity()
        self.dropout = nn.Dropout(dropout)
        self.shortcut = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
            if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bottleneck = self.bottleneck(x)
        out = torch.cat(
            [
                self.branch_k9(bottleneck),
                self.branch_k19(bottleneck),
                self.branch_k39(bottleneck),
                self.branch_pool(x),
            ],
            dim=1,
        )
        out = self.merge(out)
        out = self.se(out)
        out = out + self.shortcut(x)
        return self.dropout(out)


class MorphologyFeatureExtractor(nn.Module):
    def __init__(self, in_channels: int = 1, dropout: float = 0.15) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=15, padding=7, bias=False),
            nn.BatchNorm1d(32),
            nn.SiLU(),
        )
        self.stage1 = nn.Sequential(
            InceptionBlock1d(32, 64, branch_channels=16, bottleneck_channels=24, dropout=dropout),
            InceptionBlock1d(64, 64, branch_channels=16, bottleneck_channels=32, dropout=dropout),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.stage2 = nn.Sequential(
            InceptionBlock1d(64, 96, branch_channels=24, bottleneck_channels=32, dropout=dropout),
            InceptionBlock1d(96, 96, branch_channels=24, bottleneck_channels=32, dropout=dropout),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.stage3 = nn.Sequential(
            InceptionBlock1d(96, 128, branch_channels=32, bottleneck_channels=48, dropout=dropout),
            InceptionBlock1d(128, 128, branch_channels=32, bottleneck_channels=48, dropout=dropout),
        )
        self.out_channels = 128

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        return x


class MorphologyECGNet(nn.Module):
    def __init__(self, in_channels: int = 1, num_classes: int = 2, dropout: float = 0.15) -> None:
        super().__init__()
        self.encoder = MorphologyFeatureExtractor(in_channels=in_channels, dropout=dropout)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(self.encoder.out_channels, 64),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        return self.head(x)


class ContextRhythmECGNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        dropout: float = 0.15,
        context_beats: int = 5,
    ) -> None:
        super().__init__()
        if context_beats < 3 or context_beats % 2 == 0:
            raise ValueError("context_beats must be an odd number >= 3.")
        self.context_beats = context_beats
        self.center_index = context_beats // 2
        self.encoder = MorphologyFeatureExtractor(in_channels=in_channels, dropout=dropout)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.rhythm = nn.GRU(
            input_size=self.encoder.out_channels,
            hidden_size=96,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        fused_dim = self.encoder.out_channels + 96 * 2
        self.head = nn.Sequential(
            nn.Linear(fused_dim, 128),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, context_beats, channels, length = x.shape
        if context_beats != self.context_beats:
            raise ValueError(
                f"Expected {self.context_beats} beats in context window, received {context_beats}."
            )

        beats = x.reshape(batch_size * context_beats, channels, length)
        encoded = self.encoder(beats)
        embeddings = self.pool(encoded).flatten(1)
        embeddings = embeddings.reshape(batch_size, context_beats, -1)
        rhythm_out, _ = self.rhythm(embeddings)

        center_embedding = embeddings[:, self.center_index, :]
        center_context = rhythm_out[:, self.center_index, :]
        fused = torch.cat([center_embedding, center_context], dim=1)
        return self.head(fused)


class RRContextRhythmECGNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        dropout: float = 0.15,
        context_beats: int = 5,
        rr_feature_dim: int = 5,
    ) -> None:
        super().__init__()
        if context_beats < 3 or context_beats % 2 == 0:
            raise ValueError("context_beats must be an odd number >= 3.")
        if rr_feature_dim < 1:
            raise ValueError("rr_feature_dim must be positive for RR-aware context models.")

        self.context_beats = context_beats
        self.center_index = context_beats // 2
        self.encoder = MorphologyFeatureExtractor(in_channels=in_channels, dropout=dropout)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.rr_encoder = nn.Sequential(
            nn.Linear(rr_feature_dim, 32),
            nn.SiLU(),
            nn.Linear(32, 32),
            nn.SiLU(),
        )
        self.rhythm = nn.GRU(
            input_size=self.encoder.out_channels + 32,
            hidden_size=96,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        fused_dim = self.encoder.out_channels * 2 + 96 * 2 + 32
        self.head = nn.Sequential(
            nn.Linear(fused_dim, 160),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(160, 64),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def encode_beats(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x)
        return self.pool(encoded).flatten(1)

    def forward(
        self,
        context: torch.Tensor,
        rr_features: torch.Tensor,
        normalized_center: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, context_beats, channels, length = context.shape
        if context_beats != self.context_beats:
            raise ValueError(
                f"Expected {self.context_beats} beats in context window, received {context_beats}."
            )
        if rr_features.shape[:2] != (batch_size, context_beats):
            raise ValueError("rr_features must align with the context window.")

        beats = context.reshape(batch_size * context_beats, channels, length)
        embeddings = self.encode_beats(beats).reshape(batch_size, context_beats, -1)
        rr_embeddings = self.rr_encoder(
            rr_features.reshape(batch_size * context_beats, -1)
        ).reshape(batch_size, context_beats, -1)
        rhythm_inputs = torch.cat([embeddings, rr_embeddings], dim=-1)
        rhythm_out, _ = self.rhythm(rhythm_inputs)

        center_embedding = embeddings[:, self.center_index, :]
        center_context = rhythm_out[:, self.center_index, :]
        center_rr = rr_embeddings[:, self.center_index, :]
        normalized_embedding = self.encode_beats(normalized_center)
        fused = torch.cat([center_embedding, normalized_embedding, center_context, center_rr], dim=1)
        return self.head(fused)


class PersonalizedRRContextECGNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        dropout: float = 0.15,
        context_beats: int = 3,
        history_beats: int = 8,
        rr_feature_dim: int = 5,
    ) -> None:
        super().__init__()
        if context_beats < 3 or context_beats % 2 == 0:
            raise ValueError("context_beats must be an odd number >= 3 for personalized models.")
        if history_beats < 1:
            raise ValueError("history_beats must be at least 1 for personalized models.")
        if rr_feature_dim < 1:
            raise ValueError("rr_feature_dim must be positive for personalized models.")

        self.context_beats = context_beats
        self.center_index = context_beats // 2
        self.history_beats = history_beats
        self.rr_feature_dim = rr_feature_dim
        self.encoder = MorphologyFeatureExtractor(in_channels=in_channels, dropout=dropout)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.rr_encoder = nn.Sequential(
            nn.Linear(rr_feature_dim, 32),
            nn.SiLU(),
            nn.Linear(32, 32),
            nn.SiLU(),
        )
        self.personal_rr_encoder = nn.Sequential(
            nn.Linear(rr_feature_dim * 4 + 1, 48),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(48, 32),
            nn.SiLU(),
        )
        self.context_gru = nn.GRU(
            input_size=self.encoder.out_channels + 32,
            hidden_size=96,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        fused_dim = self.encoder.out_channels * 4 + 96 * 2 + 32 + 32 + 1
        self.head = nn.Sequential(
            nn.Linear(fused_dim, 192),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(192, 64),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def encode_beats(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x)
        return self.pool(encoded).flatten(1)

    def forward(
        self,
        context: torch.Tensor,
        rr_features: torch.Tensor,
        history_beats: torch.Tensor,
        history_mask: torch.Tensor,
        history_rr: torch.Tensor,
        rr_baseline: torch.Tensor,
        normalized_center: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, context_steps, channels, length = context.shape
        if context_steps != self.context_beats:
            raise ValueError(
                f"Expected {self.context_beats} context beats, received {context_steps}."
            )
        if rr_features.shape[:2] != (batch_size, context_steps):
            raise ValueError("rr_features must align with the context window.")

        batch_size_history, history_steps, history_channels, history_length = history_beats.shape
        if batch_size_history != batch_size or history_channels != channels or history_length != length:
            raise ValueError("history_beats must match the batch, channel, and length dimensions of context.")
        if history_steps != self.history_beats:
            raise ValueError(
                f"Expected {self.history_beats} history beats, received {history_steps}."
            )

        context_embeddings = self.encode_beats(
            context.reshape(batch_size * context_steps, channels, length)
        ).reshape(batch_size, context_steps, -1)
        context_rr_embeddings = self.rr_encoder(
            rr_features.reshape(batch_size * context_steps, -1)
        ).reshape(batch_size, context_steps, -1)
        context_inputs = torch.cat([context_embeddings, context_rr_embeddings], dim=-1)
        context_outputs, _ = self.context_gru(context_inputs)

        current_embedding = context_embeddings[:, self.center_index, :]
        center_context = context_outputs[:, self.center_index, :]
        center_rr = context_rr_embeddings[:, self.center_index, :]
        current_rr = rr_features[:, self.center_index, :]
        normalized_embedding = self.encode_beats(normalized_center)
        history_embeddings = self.encode_beats(
            history_beats.reshape(batch_size * history_steps, channels, length)
        ).reshape(batch_size, history_steps, -1)
        history_mask_expanded = history_mask.unsqueeze(-1)
        history_embeddings = history_embeddings * history_mask_expanded

        history_rr_embeddings = self.rr_encoder(
            history_rr.reshape(batch_size * history_steps, -1)
        ).reshape(batch_size, history_steps, -1)
        history_rr_embeddings = history_rr_embeddings * history_mask_expanded

        valid_counts = history_mask.sum(dim=1, keepdim=True)
        valid_counts_clamped = valid_counts.clamp_min(1.0)

        prototype = (history_embeddings * history_mask_expanded).sum(dim=1) / valid_counts_clamped

        baseline_mean = rr_baseline[:, :self.rr_feature_dim]
        baseline_std = rr_baseline[:, self.rr_feature_dim:self.rr_feature_dim * 2]
        history_fraction = rr_baseline[:, -1:]
        rr_delta = current_rr - baseline_mean
        rr_z = rr_delta / (baseline_std + 1e-3)
        personal_rr = self.personal_rr_encoder(
            torch.cat([current_rr, baseline_mean, baseline_std, rr_z, history_fraction], dim=1)
        )

        prototype_delta = torch.abs(current_embedding - prototype)
        prototype_similarity = nn.functional.cosine_similarity(
            current_embedding,
            prototype,
            dim=1,
            eps=1e-6,
        ).unsqueeze(1)
        fused = torch.cat(
            [
                current_embedding,
                normalized_embedding,
                center_context,
                center_rr,
                prototype,
                prototype_delta,
                personal_rr,
                prototype_similarity,
            ],
            dim=1,
        )
        return self.head(fused)


class BaselineECGNet(nn.Module):
    def __init__(self, in_channels: int = 1, num_classes: int = 2, dropout: float = 0.15) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(32),
            nn.SiLU(),
        )
        self.features = nn.Sequential(
            ConvBlock(32, 48, kernel_size=7, dropout=dropout, use_se=False),
            ConvBlock(48, 72, kernel_size=5, dropout=dropout, use_se=False),
            ConvBlock(72, 96, kernel_size=3, dropout=dropout, use_se=False),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(96, 64),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.features(x)
        return self.head(x)


class AttentionECGNet(nn.Module):
    def __init__(self, in_channels: int = 1, num_classes: int = 2, dropout: float = 0.15) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=9, padding=4, bias=False),
            nn.BatchNorm1d(32),
            nn.SiLU(),
        )
        self.features = nn.Sequential(
            ConvBlock(32, 48, kernel_size=7, dropout=dropout, use_se=True),
            ConvBlock(48, 80, kernel_size=5, dropout=dropout, use_se=True),
            ConvBlock(80, 128, kernel_size=3, dropout=dropout, use_se=True),
        )
        self.temporal_attention = TemporalAttention(128, heads=4, dropout=dropout)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.features(x)
        x = self.temporal_attention(x)
        return self.head(x)


def build_model(
    name: str,
    dropout: float = 0.15,
    in_channels: int = 1,
    context_beats: int = 5,
    rr_feature_dim: int = 0,
    history_beats: int = 8,
) -> nn.Module:
    key = name.lower()
    if key == "baseline":
        return BaselineECGNet(dropout=dropout, in_channels=in_channels)
    if key == "attention":
        return AttentionECGNet(dropout=dropout, in_channels=in_channels)
    if key == "morph":
        return MorphologyECGNet(dropout=dropout, in_channels=in_channels)
    if key == "context":
        return ContextRhythmECGNet(dropout=dropout, in_channels=in_channels, context_beats=context_beats)
    if key == "rr-context":
        return RRContextRhythmECGNet(
            dropout=dropout,
            in_channels=in_channels,
            context_beats=context_beats,
            rr_feature_dim=rr_feature_dim,
        )
    if key == "personalized-rr-context":
        return PersonalizedRRContextECGNet(
            dropout=dropout,
            in_channels=in_channels,
            context_beats=context_beats,
            history_beats=history_beats,
            rr_feature_dim=rr_feature_dim,
        )
    raise ValueError(f"Unknown model: {name}")
