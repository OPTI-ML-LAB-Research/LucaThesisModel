"""1D-ResNet backbone for Raman spectra feature extraction.

Architecture (Plan X, T10):
    Input  (B, 1, 1024)                    raw / preprocessed spectrum
        |
    Stem:  Conv1d(1 -> 32, k=7, s=2, p=3)  -- BN -- ReLU -- MaxPool(k=3, s=2)
        |
    Stage 1: 2 x BasicBlock1D(32 -> 32)
    Stage 2: 2 x BasicBlock1D(32 -> 64,  stride 2 in first block)
    Stage 3: 2 x BasicBlock1D(64 -> 128, stride 2 in first block)
    Stage 4: 2 x BasicBlock1D(128 -> 256, stride 2 in first block)
        |
    GlobalAvgPool1d  ->  (B, 256)
    Output (B, 256)                        feature vector

Design notes:
    * Total params ~1.1M -- well below the 5M ceiling specified in T10.
    * No classification / regression head here. Output is a pure feature
      vector consumed by ``QuantificationHead`` and any future heads.
    * Dropout (configurable, default 0.2) applied AFTER the global-avg-pool
      so that MC-Dropout at inference time samples the feature distribution
      rather than disturbing convolutional features (cleaner uncertainty).
    * Architecture is a faithful 1D adaptation of ResNet-18, scaled down
      for our smaller spectrum length (1024) and limited training set
      (~3300 samples after Scheme A composition-OOD split).

Compatible with Custom Instructions hyperparameters:
    feature_dim: 256
    dropout_rate: 0.2
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn


class BasicBlock1D(nn.Module):
    """Two-conv residual block with optional channel/stride change.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels (= number of conv filters).
        stride: Stride for the first conv (used to downsample).

    Shape:
        Input:  (B, in_channels, L)
        Output: (B, out_channels, L_out)  where L_out = ceil(L / stride)
    """

    expansion: int = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=3, stride=stride, padding=1, bias=False,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(
            out_channels, out_channels,
            kernel_size=3, stride=1, padding=1, bias=False,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # Identity / projection shortcut.
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.relu(out + identity)
        return out


class ResNet1DBackbone(nn.Module):
    """1D-ResNet feature extractor for Raman spectra.

    Args:
        in_channels: Number of input channels (1 for raw spectrum).
        feature_dim: Output feature dimension (= channels of last stage).
        stage_channels: Channels per stage. Length controls depth (4 stages
            recommended for spectrum_length=1024).
        blocks_per_stage: Number of BasicBlock1D per stage.
        dropout_rate: Dropout applied to the pooled feature vector.

    Shape:
        Input:  (B, in_channels, spectrum_length)   e.g. (B, 1, 1024)
        Output: (B, feature_dim)                     e.g. (B, 256)
    """

    def __init__(
        self,
        in_channels: int = 1,
        feature_dim: int = 256,
        stage_channels: Optional[List[int]] = None,
        blocks_per_stage: int = 2,
        dropout_rate: float = 0.2,
    ) -> None:
        super().__init__()

        if stage_channels is None:
            stage_channels = [32, 64, 128, 256]
        if stage_channels[-1] != feature_dim:
            raise ValueError(
                f"Last stage must have {feature_dim} channels to match "
                f"feature_dim, got {stage_channels[-1]}."
            )

        self.feature_dim = feature_dim
        self.dropout_rate = dropout_rate

        # ----- Stem -----
        # Aggressive early downsampling: 1024 -> 512 (conv) -> 256 (maxpool).
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, stage_channels[0],
                      kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(stage_channels[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )

        # ----- Stages -----
        # First stage: stride 1 (already downsampled by stem).
        # Subsequent stages: stride 2 in first block.
        stages = []
        in_c = stage_channels[0]
        for i, out_c in enumerate(stage_channels):
            stride = 1 if i == 0 else 2
            stages.append(self._make_stage(in_c, out_c, blocks_per_stage, stride))
            in_c = out_c
        self.stages = nn.Sequential(*stages)

        # ----- Pool + dropout -----
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(p=dropout_rate)

        self._init_weights()

    @staticmethod
    def _make_stage(
        in_channels: int, out_channels: int, n_blocks: int, first_stride: int,
    ) -> nn.Sequential:
        """Stack of BasicBlock1D; first block may downsample / change channels."""
        blocks: List[nn.Module] = [
            BasicBlock1D(in_channels, out_channels, stride=first_stride)
        ]
        for _ in range(n_blocks - 1):
            blocks.append(BasicBlock1D(out_channels, out_channels, stride=1))
        return nn.Sequential(*blocks)

    def _init_weights(self) -> None:
        """Kaiming init for convs, ones/zeros for BN."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input spectra, shape (B, in_channels, spectrum_length).

        Returns:
            Feature tensor of shape (B, feature_dim).
        """
        if x.ndim != 3:
            raise ValueError(
                f"Expected 3D input (B, C, L), got shape {tuple(x.shape)}."
            )
        x = self.stem(x)            # (B, 32, ~256)
        x = self.stages(x)          # (B, 256, ~16)
        x = self.global_pool(x)     # (B, 256, 1)
        x = x.squeeze(-1)           # (B, 256)
        x = self.dropout(x)         # (B, 256)
        return x

    def count_parameters(self, trainable_only: bool = True) -> int:
        """Total parameter count, optionally only trainable ones."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)

    backbone = ResNet1DBackbone(
        in_channels=1, feature_dim=256, dropout_rate=0.2,
    )
    n_params = backbone.count_parameters()
    print(f"[T10] Trainable parameters: {n_params:,}")
    assert n_params < 5_000_000, f"Backbone too large: {n_params:,} > 5M"

    x = torch.randn(8, 1, 1024)
    backbone.eval()  # dropout off for shape check
    with torch.no_grad():
        z = backbone(x)
    print(f"[T10] Input shape : {tuple(x.shape)}")
    print(f"[T10] Output shape: {tuple(z.shape)}")
    assert z.shape == (8, 256), f"Bad output shape: {z.shape}"
    assert torch.isfinite(z).all(), "NaN/Inf in backbone output"
    print("[T10] Smoke test PASSED")
