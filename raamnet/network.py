"""RAAM-Net for hyperspectral image change detection.

Shared layers preserve reproducible initialization and checkpoint keys.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from . import _layers as original

MODEL_VERSION = "RAAM-Net"


class FeatureDirectionalDetail(original.BitemporalDirectionalDetailModule):
    """Reference-feature BDDM; inputs are Fr0 and the detached reference cue."""

    def forward(self, feature: torch.Tensor, reference_role: torch.Tensor) -> torch.Tensor:
        if reference_role.ndim == 3:
            reference_role = reference_role.unsqueeze(1)
        if feature.ndim != 4 or reference_role.ndim != 4 or reference_role.shape[1] != 1:
            raise ValueError("feature [B,D,H,W] and role [B,1,H,W] or [B,H,W] required")
        if feature.shape[0] != reference_role.shape[0]:
            raise ValueError("batch sizes differ")
        size = feature.shape[-2:]
        role = self._upsample(reference_role, size).to(device=feature.device, dtype=feature.dtype)
        evidence = self._directional_gradients(feature).abs()
        pooled = self._upsample(self._downsample(feature), size)
        edge_energy = (feature - pooled).abs().mean(dim=1, keepdim=True)
        evidence_energy = evidence.mean(dim=2)
        direction_weight = self.direction_gate(torch.cat((role, edge_energy, evidence_energy), dim=1)).softmax(dim=1)
        directional = (direction_weight.unsqueeze(2) * evidence).sum(dim=1)
        high = self.high_detail(self.high_in(directional))
        low = self.low_detail(self.low_in(self._downsample(directional)))
        high_exchange = high + self._upsample(self.low_to_high(low), size)
        low_exchange = low + self.high_to_low(self._downsample(high))
        high = self.high_out(self.high_refine(high_exchange))
        low = self._upsample(self.low_out(self.low_refine(low_exchange)), size)
        frequency_weight = self.frequency_gate(torch.cat((role, edge_energy), dim=1)).softmax(dim=1)
        octave = frequency_weight[:, :1] * high + frequency_weight[:, 1:] * low
        return feature + self.gamma_base * self.base(feature) + self.gamma_octave * octave


class ConvFeatureFusion(nn.Module):
    """Balanced concatenation projection with zero-initialized refinement."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = channels
        self.projection = nn.Conv2d(2 * channels, channels, 1, bias=False)
        self.refinement = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            original._group_norm(channels), nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, 1, bias=False),
        )
        with torch.no_grad():
            self.projection.weight.zero_()
            index = torch.arange(channels)
            self.projection.weight[index, index, 0, 0] = 0.5
            self.projection.weight[index, index + channels, 0, 0] = 0.5
            self.refinement[-1].weight.zero_()

    def forward(self, first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        if first.ndim != 4 or first.shape != second.shape or first.shape[1] != self.channels:
            raise ValueError("two equal [B,C,H,W] features with configured channels required")
        content = self.projection(torch.cat((first, second), dim=1))
        return content + self.refinement(content)


class ProgressiveCrossRoleFusion(original.ProgressiveCrossRoleFusion):
    """Role-guided residual correction with learned gating and selection."""

    def forward(self, bc, br, rc, rd, cc, cd, sr, sc):
        sr = self._score(sr, bc)
        sc = self._score(sc, bc)
        ar = 0.5 * (rc + rd)
        ac = 0.5 * (cc + cd)
        difference = (ac - ar).abs()
        agreement = ac * ar
        uncertainty = 4.0 * sr * sc
        weights = self.role_gate(torch.cat((sr, sc, difference.mean(1, keepdim=True), uncertainty), 1)).softmax(1)
        selector = self._selector(difference, sc, uncertainty)
        dc = self.change_correction(ac, difference)
        dr = self.reference_correction(ar, agreement)
        change = bc + weights[:, 1:2] * selector * dc
        reference = br + weights[:, :1] * selector * dr
        return torch.cat((change, reference), 1)


class RAAMNet(original.RAAMNet):
    """RAAM-Net with reference detail refinement and discrepancy context.

    Shared tensors are initialized in a fixed order for reproducibility.
    The two fusion blocks preserve the random-number state during
    construction so initialization remains checkpoint-compatible.
    """

    def __init__(self, bands: int, dim: int = 64, classes: int = 2, d_state: int = 16) -> None:
        super().__init__(bands=bands, dim=dim, classes=classes, d_state=d_state)
        self.reference_detail.__class__ = FeatureDirectionalDetail
        del self.reference_detail.temporal_projection
        state = torch.get_rng_state().clone()
        self.fuse_change = ConvFeatureFusion(dim)
        self.fuse_reference = ConvFeatureFusion(dim)
        torch.set_rng_state(state)
        self.prcf.__class__ = ProgressiveCrossRoleFusion

    def forward(self, date1: torch.Tensor, date2: torch.Tensor) -> torch.Tensor:
        if date1.shape != date2.shape or date1.ndim != 4:
            raise ValueError("equal [B,C,H,W] inputs required")
        gain = self.spectral_gate(date1, date2)
        first, second = date1 * gain, date2 * gain
        reference_role, change_role = self.role_estimator(first, second)
        reference = self.reference_reduction(torch.cat((first, second), dim=1))
        change = self.change_reduction((second - first).abs())
        cue = change_role + 0.0
        topology = self._change_topology_descriptor(first, second, cue)
        context = self.change_context(change, cue, topology)
        detail = self.reference_detail(reference, reference_role.detach())
        base_change = self.fuse_change(context, detail)
        base_reference = self.fuse_reference(change, reference)
        fused = self.prcf(
            base_change, base_reference, reference, detail, context, change,
            reference_role.detach(), change_role.detach(),
        )
        tokens = self.head_drop(self.head_norm(fused.permute(0, 2, 3, 1)))
        tokens = F.silu(self.head_1(tokens))
        tokens = F.silu(self.head_2(tokens))
        return self.classifier(tokens).permute(0, 3, 1, 2).contiguous()
