"""Publication implementation of RAAM-Net.

The model has four explicit stages:
    temporal--spectral calibration -> complementary role estimation ->
    reference BDDM + change role-conditioned dual-axis BiMamba -> PRCF.

Inputs use the standard PyTorch layout ``[B, C, H, W]`` and the output is
unnormalized two-class logits with shape ``[B, 2, H, W]``.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

try:
    from mamba_ssm import Mamba
except ImportError as error:  # pragma: no cover - dependency guidance
    raise ImportError(
        "RAAM-Net requires mamba-ssm. Install the requirements for the CUDA "
        "and PyTorch version on your system before importing raamnet."
    ) from error


def _group_norm(channels: int, max_groups: int = 8) -> nn.GroupNorm:
    """Create a valid GroupNorm layer for an arbitrary channel count."""
    groups = min(int(channels), int(max_groups))
    while channels % groups:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class BidirectionalMamba1D(nn.Module):
    """Independent forward/backward Mamba paths over a sequence [B, L, D]."""

    def __init__(self, dim: int, d_state: int = 16, d_conv: int = 4,
                 expand: int = 2, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.forward_mamba = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.backward_mamba = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.out_norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        normalized = self.norm(sequence)
        forward = self.forward_mamba(normalized)
        backward = self.backward_mamba(torch.flip(normalized, dims=(1,)))
        backward = torch.flip(backward, dims=(1,))
        return sequence + self.drop(self.out_norm(0.5 * (forward + backward)))


class TemporalAwareSpectralGate(nn.Module):
    """Shared bitemporal soft calibration over the ordered spectral axis.

    Each band token contains mean absolute difference, spatial correlation,
    and RMS difference. The zero-initialized prediction head makes the initial
    gate exactly one, so the module starts as an identity mapping.
    """

    def __init__(self, embed_dim: int = 32, d_state: int = 8, gate_scale: float = 0.5) -> None:
        super().__init__()
        if not 0.0 < gate_scale < 1.0:
            raise ValueError("gate_scale must be in (0, 1)")
        self.input_proj = nn.Sequential(nn.Linear(3, embed_dim), nn.SiLU())
        self.spectral_model = BidirectionalMamba1D(embed_dim, d_state=d_state, d_conv=3, dropout=0.0)
        self.gate_head = nn.Linear(embed_dim, 1)
        self.gate_scale = float(gate_scale)
        nn.init.zeros_(self.gate_head.weight)
        nn.init.zeros_(self.gate_head.bias)

    @staticmethod
    def _standardize(value: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        return (value - value.mean(dim=1, keepdim=True)) / (
            value.std(dim=1, keepdim=True, unbiased=False) + eps
        )

    def forward(self, date1: torch.Tensor, date2: torch.Tensor) -> torch.Tensor:
        if date1.shape != date2.shape or date1.ndim != 4:
            raise ValueError("date1 and date2 must be equal [B, C, H, W] tensors")
        # Statistics are accumulated in float32 for mixed-precision stability.
        first, second = date1.float(), date2.float()
        delta = second - first
        centered_first = first - first.mean(dim=(2, 3), keepdim=True)
        centered_second = second - second.mean(dim=(2, 3), keepdim=True)
        std_first = centered_first.square().mean(dim=(2, 3)).clamp_min(1e-6).sqrt()
        std_second = centered_second.square().mean(dim=(2, 3)).clamp_min(1e-6).sqrt()
        correlation = (
            (centered_first * centered_second).mean(dim=(2, 3)) /
            (std_first * std_second + 1e-6)
        ).clamp(-1.0, 1.0)
        descriptor = torch.stack((
            self._standardize(delta.abs().mean(dim=(2, 3))),
            self._standardize(correlation),
            self._standardize(delta.square().mean(dim=(2, 3)).clamp_min(1e-8).sqrt()),
        ), dim=-1)
        token = self.spectral_model(self.input_proj(descriptor))
        gate = 1.0 + self.gate_scale * self.gate_head(token).squeeze(-1).tanh()
        return gate[:, :, None, None].to(dtype=date1.dtype)


class ComplementaryRoleEstimator(nn.Module):
    """Physical spectral-angle prior with a learnable spatial correction."""

    def __init__(self) -> None:
        super().__init__()
        self.refiner = nn.Sequential(
            nn.Conv2d(2, 8, 3, padding=1, bias=False),
            nn.GroupNorm(1, 8),
            nn.SiLU(inplace=True),
            nn.Conv2d(8, 1, 1),
        )
        nn.init.zeros_(self.refiner[-1].weight)
        nn.init.zeros_(self.refiner[-1].bias)

    def forward(self, date1: torch.Tensor, date2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cosine = F.cosine_similarity(date1, date2, dim=1, eps=1e-6).clamp(-1.0 + 1e-5, 1.0 - 1e-5)
        angle = cosine.acos() / math.pi
        difference = (date2 - date1).abs().mean(dim=1)
        lower = difference.amin(dim=(1, 2), keepdim=True)
        upper = difference.amax(dim=(1, 2), keepdim=True)
        normalized_difference = (difference - lower) / (upper - lower + 1e-6)
        prior = (0.5 * angle + 0.5 * normalized_difference).clamp(1e-4, 1.0 - 1e-4)
        correction = self.refiner(torch.stack((angle, normalized_difference), dim=1)).squeeze(1)
        change_role = torch.sigmoid(torch.logit(prior) + correction)
        return 1.0 - change_role, change_role


def _horizontal_snake_order(height: int, width: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    grid = torch.arange(height * width, device=device).reshape(height, width)
    grid = grid.clone()
    grid[1::2] = grid[1::2].flip(1)
    order = grid.flatten()
    inverse = torch.empty_like(order)
    inverse[order] = torch.arange(order.numel(), device=device)
    return order, inverse


def _vertical_snake_order(height: int, width: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    grid = torch.arange(height * width, device=device).reshape(height, width).t().contiguous()
    grid[1::2] = grid[1::2].flip(1)
    order = grid.flatten()
    inverse = torch.empty_like(order)
    inverse[order] = torch.arange(order.numel(), device=device)
    return order, inverse


class RoleConditionedDualAxisBiMamba(nn.Module):
    """Dual-axis BiMamba whose residual context is routed by the change role.

    Two topology-preserving Snake paths are evaluated with shared Mamba
    parameters. Bitemporal change evidence softly selects the horizontal and
    vertical contexts; no pixel reordering by a role score is used.
    """

    temperature = 0.20
    routing_floor = 0.20

    def __init__(self, dim: int = 64, d_state: int = 16, role_hidden: int = 16,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.sequence_model = BidirectionalMamba1D(dim, d_state=d_state, d_conv=4, dropout=dropout)
        self.output_gate = nn.Sequential(nn.Linear(dim, dim), nn.Sigmoid())
        self.role_encoder = nn.Sequential(nn.Linear(3, role_hidden), nn.SiLU())
        self.role_input = nn.Linear(role_hidden, dim)
        self.role_gain = nn.Linear(role_hidden, dim)
        self.role_scale = nn.Parameter(torch.zeros(1))
        nn.init.zeros_(self.role_input.weight)
        nn.init.zeros_(self.role_input.bias)
        nn.init.zeros_(self.role_gain.weight)
        nn.init.zeros_(self.role_gain.bias)

    @staticmethod
    def _role_descriptors(score: torch.Tensor) -> torch.Tensor:
        batch, height, width = score.shape
        uncertainty = 4.0 * score * (1.0 - score)
        dx = torch.zeros_like(score)
        dy = torch.zeros_like(score)
        if width > 1:
            dx[:, :, :-1] = (score[:, :, 1:] - score[:, :, :-1]).abs()
        if height > 1:
            dy[:, :-1, :] = (score[:, 1:, :] - score[:, :-1, :]).abs()
        boundary = 0.5 * (dx + dy)
        return torch.stack((score, uncertainty, boundary), dim=-1).reshape(batch, height * width, 3)

    @staticmethod
    def _transition_cost(descriptor: torch.Tensor, order: torch.Tensor) -> torch.Tensor:
        ordered = descriptor.index_select(1, order)
        if ordered.shape[1] < 2:
            return torch.zeros(ordered.shape[0], dtype=ordered.dtype, device=ordered.device)
        return (ordered[:, 1:] - ordered[:, :-1]).square().mean(dim=(1, 2))

    def _path_weights(self, descriptor: torch.Tensor, horizontal_order: torch.Tensor,
                      vertical_order: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        horizontal_cost = self._transition_cost(descriptor, horizontal_order)
        vertical_cost = self._transition_cost(descriptor, vertical_order)
        weights = (-torch.stack((horizontal_cost, vertical_cost), dim=-1) / self.temperature).softmax(dim=-1)
        return weights[:, 0, None, None], weights[:, 1, None, None]

    def _run_path(self, feature: torch.Tensor, role: torch.Tensor, descriptors: torch.Tensor,
                  order: torch.Tensor, inverse: torch.Tensor) -> torch.Tensor:
        sequence = feature.index_select(1, order)
        ordered_role = role.index_select(1, order)
        encoded_role = self.role_encoder(descriptors.index_select(1, order).to(sequence.dtype))
        conditioning = self.role_input(encoded_role)
        modeled = self.sequence_model(sequence + conditioning)
        delta = modeled - (sequence + conditioning)
        channel_gain = 1.0 + 0.25 * self.role_gain(encoded_role).tanh()
        neutral = self.routing_floor + 0.5 * (1.0 - self.routing_floor)
        routing = (self.routing_floor + (1.0 - self.routing_floor) * ordered_role) / neutral
        output = sequence + routing * channel_gain * self.output_gate(modeled) * delta
        return output.index_select(1, inverse)

    def forward(self, feature: torch.Tensor, change_role: torch.Tensor,
                topology_descriptor: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = feature.shape
        if change_role.shape != (batch, height, width):
            raise ValueError("change_role must have shape [B, H, W]")
        if topology_descriptor.shape != (batch, 3, height, width):
            raise ValueError("topology_descriptor must have shape [B, 3, H, W]")
        sequence = feature.flatten(2).transpose(1, 2)
        role = change_role.reshape(batch, height * width, 1)
        descriptor = topology_descriptor.flatten(2).transpose(1, 2)
        descriptors = self._role_descriptors(change_role)
        horizontal_order, horizontal_inverse = _horizontal_snake_order(height, width, feature.device)
        vertical_order, vertical_inverse = _vertical_snake_order(height, width, feature.device)
        horizontal = self._run_path(sequence, role, descriptors, horizontal_order, horizontal_inverse)
        vertical = self._run_path(sequence, role, descriptors, vertical_order, vertical_inverse)
        horizontal_weight, vertical_weight = self._path_weights(descriptor, horizontal_order, vertical_order)
        output = horizontal_weight * horizontal + vertical_weight * vertical
        return output.transpose(1, 2).reshape(batch, channels, height, width).contiguous()


class BitemporalDirectionalDetailModule(nn.Module):
    """Reference-side directional and octave detail refinement (BDDM)."""

    def __init__(self, dim: int, input_bands: int, octave_ratio: float = 0.25,
                 gamma_init: float = 0.1) -> None:
        super().__init__()
        low_channels = max(1, int(round(dim * octave_ratio)))
        high_channels = dim - low_channels
        if high_channels < 1:
            raise ValueError("octave_ratio leaves no high-frequency channels")
        self.temporal_projection = nn.Sequential(
            nn.Conv2d(input_bands, dim, 1, bias=False), _group_norm(dim), nn.SiLU(inplace=True)
        )
        kernels = torch.tensor([
            [[1., 0., -1.], [2., 0., -2.], [1., 0., -1.]],
            [[1., 2., 1.], [0., 0., 0.], [-1., -2., -1.]],
            [[2., 1., 0.], [1., 0., -1.], [0., -1., -2.]],
            [[0., 1., 2.], [-1., 0., 1.], [-2., -1., 0.]],
        ], dtype=torch.float32).div_(4.0).unsqueeze(1)
        self.register_buffer("direction_kernels", kernels, persistent=False)
        self.base = nn.Sequential(nn.Conv2d(dim, dim, 3, padding=1, bias=False), _group_norm(dim), nn.SiLU(inplace=True))
        self.high_in = nn.Sequential(nn.Conv2d(dim, high_channels, 1, bias=False), _group_norm(high_channels), nn.SiLU(inplace=True))
        self.low_in = nn.Sequential(nn.Conv2d(dim, low_channels, 1, bias=False), _group_norm(low_channels), nn.SiLU(inplace=True))
        self.high_detail = nn.Sequential(nn.Conv2d(high_channels, high_channels, 3, padding=1, groups=high_channels, bias=False), _group_norm(high_channels), nn.SiLU(inplace=True))
        self.low_detail = nn.Sequential(nn.Conv2d(low_channels, low_channels, 3, padding=1, groups=low_channels, bias=False), _group_norm(low_channels), nn.SiLU(inplace=True))
        self.low_to_high = nn.Conv2d(low_channels, high_channels, 1, bias=False)
        self.high_to_low = nn.Conv2d(high_channels, low_channels, 1, bias=False)
        self.high_refine = nn.Sequential(nn.Conv2d(high_channels, high_channels, 3, padding=1, groups=high_channels, bias=False), _group_norm(high_channels), nn.SiLU(inplace=True))
        self.low_refine = nn.Sequential(nn.Conv2d(low_channels, low_channels, 3, padding=1, groups=low_channels, bias=False), _group_norm(low_channels), nn.SiLU(inplace=True))
        self.high_out = nn.Conv2d(high_channels, dim, 1, bias=False)
        self.low_out = nn.Conv2d(low_channels, dim, 1, bias=False)
        hidden = max(dim // 8, 8)
        self.direction_gate = nn.Sequential(nn.Conv2d(6, hidden, 3, padding=1), nn.SiLU(inplace=True), nn.Conv2d(hidden, 4, 1))
        self.frequency_gate = nn.Sequential(nn.Conv2d(2, hidden, 3, padding=1), nn.SiLU(inplace=True), nn.Conv2d(hidden, 2, 1))
        for gate in (self.direction_gate, self.frequency_gate):
            nn.init.zeros_(gate[-1].weight)
            nn.init.zeros_(gate[-1].bias)
        self.gamma_base = nn.Parameter(torch.tensor(float(gamma_init)))
        self.gamma_octave = nn.Parameter(torch.tensor(float(gamma_init)))

    @staticmethod
    def _downsample(value: torch.Tensor) -> torch.Tensor:
        return F.avg_pool2d(value, kernel_size=2, stride=2, ceil_mode=True)

    @staticmethod
    def _upsample(value: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        return F.interpolate(value, size=size, mode="bilinear", align_corners=False)

    def _directional_gradients(self, value: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = value.shape
        kernels = self.direction_kernels.to(dtype=value.dtype).unsqueeze(0).expand(
            channels, -1, -1, -1, -1
        ).reshape(channels * 4, 1, 3, 3)
        response = F.conv2d(value, kernels, padding=1, groups=channels)
        return response.reshape(batch, channels, 4, height, width).permute(0, 2, 1, 3, 4)

    def forward(self, feature: torch.Tensor, date1: torch.Tensor, date2: torch.Tensor,
                reference_role: torch.Tensor) -> torch.Tensor:
        if reference_role.ndim == 3:
            reference_role = reference_role.unsqueeze(1)
        if reference_role.shape[1] != 1:
            raise ValueError("reference_role must be [B, H, W] or [B, 1, H, W]")
        size = feature.shape[-2:]
        if date1.shape != date2.shape or date1.shape[-2:] != size:
            raise ValueError("dates must share the BDDM feature-map spatial size")
        role = self._upsample(reference_role, size).to(dtype=feature.dtype)
        projected1 = self.temporal_projection(date1)
        projected2 = self.temporal_projection(date2)
        gradients1 = self._directional_gradients(projected1)
        gradients2 = self._directional_gradients(projected2)
        discrepancy = (gradients2 - gradients1).abs()
        evidence = 0.5 * (gradients1.abs() + gradients2.abs()) * torch.exp(-discrepancy)
        pooled = self._upsample(self._downsample(feature), size)
        edge_energy = (feature - pooled).abs().mean(dim=1, keepdim=True)
        evidence_energy = evidence.abs().mean(dim=2)
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


class FeatureFusionModule(nn.Module):
    """Difference-guided base interaction for two complementary features."""

    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.diff_conv = nn.Sequential(nn.Conv2d(channels, hidden, 3, padding=1, bias=False), nn.ReLU(inplace=True), nn.Conv2d(hidden, channels, 3, padding=1, bias=False))
        self.global_avg = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, hidden, 1, bias=False), nn.ReLU(inplace=True), nn.Conv2d(hidden, channels, 1, bias=False))
        self.global_max = nn.Sequential(nn.AdaptiveMaxPool2d(1), nn.Conv2d(channels, hidden, 1, bias=False), nn.ReLU(inplace=True), nn.Conv2d(hidden, channels, 1, bias=False))

    def forward(self, first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        difference = second - first
        weight = torch.sigmoid(self.diff_conv(difference) + self.global_avg(difference) + self.global_max(difference))
        return first * weight + second * (1.0 - weight)


class _ZeroInitCorrection(nn.Module):
    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.block = nn.Sequential(
            nn.Conv2d(2 * channels, hidden, 1, bias=False), _group_norm(hidden), nn.SiLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden, bias=False), _group_norm(hidden), nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
        )
        nn.init.zeros_(self.block[-1].weight)
        nn.init.zeros_(self.block[-1].bias)

    def forward(self, first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        return self.block(torch.cat((first, second), dim=1))


class ProgressiveCrossRoleFusion(nn.Module):
    """FFM bases followed by role-guided, identity-initialized correction."""

    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        hidden = max(channels // 8, 8)
        self.change_correction = _ZeroInitCorrection(channels, reduction)
        self.reference_correction = _ZeroInitCorrection(channels, reduction)
        self.role_gate = nn.Sequential(nn.Conv2d(4, hidden, 3, padding=1), nn.SiLU(inplace=True), nn.Conv2d(hidden, 2, 1))
        nn.init.zeros_(self.role_gate[-1].weight)
        nn.init.zeros_(self.role_gate[-1].bias)
        selector_hidden = max(channels // 8, 8)
        self.channel_reduce = nn.Conv2d(channels, hidden, 1)
        self.channel_expand = nn.Conv2d(hidden, channels, 1)
        self.spatial_reduce = nn.Conv2d(4, selector_hidden, 3, padding=1)
        self.spatial_expand = nn.Conv2d(selector_hidden, 1, 1)
        nn.init.zeros_(self.channel_expand.weight)
        nn.init.zeros_(self.channel_expand.bias)
        nn.init.zeros_(self.spatial_expand.weight)
        nn.init.zeros_(self.spatial_expand.bias)

    @staticmethod
    def _score(score: torch.Tensor, feature: torch.Tensor) -> torch.Tensor:
        if score.ndim == 3:
            score = score[:, None]
        if score.shape[1] != 1:
            raise ValueError("role score must be a single-channel map")
        if score.shape[-2:] != feature.shape[-2:]:
            score = F.interpolate(score, size=feature.shape[-2:], mode="bilinear", align_corners=False)
        return score.to(feature.dtype)

    def _selector(self, difference: torch.Tensor, change_role: torch.Tensor,
                  uncertainty: torch.Tensor) -> torch.Tensor:
        channel_logits = self.channel_expand(F.silu(self.channel_reduce(F.adaptive_avg_pool2d(difference, 1))))
        channel_logits = channel_logits + self.channel_expand(F.silu(self.channel_reduce(F.adaptive_max_pool2d(difference, 1))))
        channel = 2.0 * torch.sigmoid(channel_logits)
        spatial_input = torch.cat((difference.mean(dim=1, keepdim=True), difference.amax(dim=1, keepdim=True), change_role, uncertainty), dim=1)
        spatial = 2.0 * torch.sigmoid(self.spatial_expand(F.silu(self.spatial_reduce(spatial_input))))
        return 0.5 * (channel + spatial)

    def forward(self, base_change: torch.Tensor, base_reference: torch.Tensor,
                reference_context: torch.Tensor, reference_detail: torch.Tensor,
                change_context: torch.Tensor, change_detail: torch.Tensor,
                reference_role: torch.Tensor, change_role: torch.Tensor) -> torch.Tensor:
        reference_role = self._score(reference_role, base_change)
        change_role = self._score(change_role, base_change)
        aligned_reference = 0.5 * (reference_context + reference_detail)
        aligned_change = 0.5 * (change_context + change_detail)
        difference = (aligned_change - aligned_reference).abs()
        agreement = aligned_change * aligned_reference
        uncertainty = 4.0 * reference_role * change_role
        role_weight = self.role_gate(torch.cat((reference_role, change_role, difference.mean(dim=1, keepdim=True), uncertainty), dim=1)).softmax(dim=1)
        selector = self._selector(difference, change_role, uncertainty)
        change = base_change + role_weight[:, 1:2] * selector * self.change_correction(aligned_change, difference)
        reference = base_reference + role_weight[:, :1] * selector * self.reference_correction(aligned_reference, agreement)
        return torch.cat((change, reference), dim=1)


class RAAMNet(nn.Module):
    """Final RAAM-Net with reference BDDM and change dual-axis BiMamba."""

    def __init__(self, bands: int, dim: int = 64, classes: int = 2, d_state: int = 16) -> None:
        super().__init__()
        if classes != 2:
            raise ValueError("RAAM-Net is implemented here for binary change detection (classes=2)")
        self.spectral_gate = TemporalAwareSpectralGate()
        self.role_estimator = ComplementaryRoleEstimator()
        self.reference_reduction = nn.Sequential(nn.Conv2d(2 * bands, dim, 1, bias=False), _group_norm(dim), nn.SiLU(inplace=True), nn.Dropout2d(0.1))
        self.change_reduction = nn.Sequential(nn.Conv2d(bands, dim, 1, bias=False), _group_norm(dim), nn.SiLU(inplace=True), nn.Dropout2d(0.1))
        self.change_context = RoleConditionedDualAxisBiMamba(dim, d_state=d_state)
        self.reference_detail = BitemporalDirectionalDetailModule(dim, bands)
        self.fuse_change = FeatureFusionModule(dim)
        self.fuse_reference = FeatureFusionModule(dim)
        self.prcf = ProgressiveCrossRoleFusion(dim)
        self.head_norm = nn.LayerNorm(2 * dim)
        self.head_drop = nn.Dropout(0.1)
        self.head_1 = nn.Linear(2 * dim, dim)
        self.head_2 = nn.Linear(dim, dim // 2)
        self.classifier = nn.Linear(dim // 2, classes)

    @staticmethod
    def _standardize_map(value: torch.Tensor) -> torch.Tensor:
        return (value - value.mean(dim=(2, 3), keepdim=True)) / (
            value.std(dim=(2, 3), keepdim=True, unbiased=False) + 1e-6
        )

    def _change_topology_descriptor(self, date1: torch.Tensor, date2: torch.Tensor,
                                    change_role: torch.Tensor) -> torch.Tensor:
        dot = (date1 * date2).sum(dim=1, keepdim=True)
        norm1 = date1.square().sum(dim=1, keepdim=True).clamp_min(1e-8).sqrt()
        norm2 = date2.square().sum(dim=1, keepdim=True).clamp_min(1e-8).sqrt()
        angular = 1.0 - (dot / (norm1 * norm2 + 1e-6)).clamp(-1.0, 1.0)
        relative = (date2 - date1).square().sum(dim=1, keepdim=True).sqrt() / (norm1 + norm2 + 1e-6)
        role = change_role[:, None]
        return torch.cat((
            self._standardize_map(role * relative),
            self._standardize_map(role * angular),
            self._standardize_map(role),
        ), dim=1)

    def forward(self, date1: torch.Tensor, date2: torch.Tensor) -> torch.Tensor:
        if date1.shape != date2.shape or date1.ndim != 4:
            raise ValueError("date1 and date2 must be equal [B, C, H, W] tensors")
        gate = self.spectral_gate(date1, date2)
        calibrated1, calibrated2 = date1 * gate, date2 * gate
        reference_role, change_role = self.role_estimator(calibrated1, calibrated2)
        reference = self.reference_reduction(torch.cat((calibrated1, calibrated2), dim=1))
        change = self.change_reduction((calibrated2 - calibrated1).abs())
        topology = self._change_topology_descriptor(calibrated1, calibrated2, change_role)
        change_context = self.change_context(change, change_role, topology)

        # The final validated configuration treats roles as stable downstream
        # conditions; the role estimator remains trained through BiMamba.
        reference_for_fusion, change_for_fusion = reference_role.detach(), change_role.detach()
        reference_detail = self.reference_detail(reference, calibrated1, calibrated2, reference_for_fusion)
        base_change = self.fuse_change(change_context, reference_detail)
        base_reference = self.fuse_reference(change, reference)
        fused = self.prcf(
            base_change, base_reference, reference, reference_detail,
            change_context, change, reference_for_fusion, change_for_fusion,
        )
        tokens = fused.permute(0, 2, 3, 1)
        tokens = self.head_drop(self.head_norm(tokens))
        tokens = F.silu(self.head_1(tokens))
        tokens = F.silu(self.head_2(tokens))
        return self.classifier(tokens).permute(0, 3, 1, 2).contiguous()

    @torch.no_grad()
    def predict(self, date1: torch.Tensor, date2: torch.Tensor) -> torch.Tensor:
        """Return the binary change map [B, H, W]."""
        return self(date1, date2).argmax(dim=1)
