"""Analytical complexity profile for the final RAAM-Net.

The report follows the convention used in the paper: convolutional and
linear-projection MACs are counted, including Mamba's input/output
projections and depth-wise sequence convolution. The implementation-specific
selective state-scan recurrence is deliberately excluded, because its kernel
cost is not reported consistently by THOP/fvcore. FLOPs are defined as
``2 x MACs``.

Unlike tracer-based profilers, this script does not execute an HxW Mamba
forward pass. It is therefore reproducible on CPU-only environments and
cannot accidentally include a hardware-specific scan kernel.
"""

from __future__ import annotations

import argparse

from raamnet import RAAMNet


DEFAULT_SCENES = {
    "Farmland": (450, 140, 155),
    "Hermiston": (390, 200, 242),
    "River": (463, 241, 198),
    "USA": (307, 241, 154),
}


def readable(value: float) -> str:
    for suffix, scale in (("T", 1e12), ("G", 1e9), ("M", 1e6), ("K", 1e3)):
        if abs(value) >= scale:
            return f"{value / scale:.2f}{suffix}"
    return f"{value:.0f}"


def conv_macs(height: int, width: int, in_channels: int, out_channels: int,
              kernel: int = 1, groups: int = 1) -> int:
    """MACs of a same-resolution Conv2d (bias/normalization excluded)."""
    return int(height * width * out_channels * (in_channels // groups) * kernel * kernel)


def linear_macs(tokens: int, in_features: int, out_features: int) -> int:
    return int(tokens * in_features * out_features)


def bidirectional_mamba_macs(length: int, dim: int, d_state: int,
                              d_conv: int, expand: int = 2) -> int:
    """MACs of ``BidirectionalMamba1D`` excluding selective state scan."""
    inner = expand * dim
    dt_rank = (dim + 15) // 16  # mamba_ssm default: ceil(d_model / 16)
    one_direction = linear_macs(length, dim, 2 * inner)
    one_direction += length * (
        inner * d_conv +                   # causal depth-wise Conv1d
        inner * (dt_rank + 2 * d_state) +  # x projection
        inner * dt_rank +                   # dt projection
        inner * dim                         # output projection
    )
    return 2 * one_direction


def bddm_macs(height: int, width: int, bands: int, dim: int) -> int:
    """MACs of the reference Bitemporal Directional Detail Module."""
    low = round(dim * 0.25)
    high = dim - low
    low_h, low_w = (height + 1) // 2, (width + 1) // 2
    full, half = 0, 0

    # Temporal projection and four fixed directional depth-wise filters.
    full += 2 * conv_macs(height, width, bands, dim)
    full += 2 * height * width * dim * 4 * 3 * 3
    full += conv_macs(height, width, dim, dim, kernel=3)  # base path

    full += conv_macs(height, width, dim, high)
    half += conv_macs(low_h, low_w, dim, low)
    full += conv_macs(height, width, high, high, kernel=3, groups=high)
    half += conv_macs(low_h, low_w, low, low, kernel=3, groups=low)
    full += conv_macs(height, width, low, high)  # low -> high
    half += conv_macs(low_h, low_w, high, low)   # high -> low
    full += conv_macs(height, width, high, high, kernel=3, groups=high)
    half += conv_macs(low_h, low_w, low, low, kernel=3, groups=low)
    full += conv_macs(height, width, high, dim)
    half += conv_macs(low_h, low_w, low, dim)

    hidden = max(dim // 8, 8)
    full += conv_macs(height, width, 6, hidden, kernel=3)
    full += conv_macs(height, width, hidden, 4)
    full += conv_macs(height, width, 2, hidden, kernel=3)
    full += conv_macs(height, width, hidden, 2)
    return full + half


def feature_fusion_macs(height: int, width: int, dim: int) -> int:
    """MACs of one Feature Fusion Module (two such modules are used)."""
    hidden = max(dim // 4, 1)
    spatial = conv_macs(height, width, dim, hidden, kernel=3)
    spatial += conv_macs(height, width, hidden, dim, kernel=3)
    pooled = 2 * (dim * hidden + hidden * dim)  # avg and max branches
    return spatial + pooled


def prcf_macs(height: int, width: int, dim: int) -> int:
    """MACs of Progressive Cross-Role Fusion, including two corrections."""
    correction_hidden = max(dim // 4, 8)
    one_correction = conv_macs(height, width, 2 * dim, correction_hidden)
    one_correction += conv_macs(height, width, correction_hidden, correction_hidden,
                                kernel=3, groups=correction_hidden)
    one_correction += conv_macs(height, width, correction_hidden, dim)

    role_hidden = max(dim // 8, 8)
    total = 2 * one_correction
    total += conv_macs(height, width, 4, role_hidden, kernel=3)
    total += conv_macs(height, width, role_hidden, 2)
    total += 2 * (dim * role_hidden + role_hidden * dim)  # channel selector
    total += conv_macs(height, width, 4, role_hidden, kernel=3)
    total += conv_macs(height, width, role_hidden, 1)
    return total


def profile_scene(name: str, height: int, width: int, bands: int, dim: int = 64) -> tuple[int, int]:
    """Return exact parameter count and analytical MACs for one full scene."""
    # Instantiation keeps parameter counting valid if the model changes.
    parameters = sum(parameter.numel() for parameter in RAAMNet(bands=bands, dim=dim).parameters())
    pixels = height * width
    macs = 0

    # TASG: descriptor embedding/head + bidirectional spectral Mamba.
    macs += linear_macs(bands, 3, 32) + linear_macs(bands, 32, 1)
    macs += bidirectional_mamba_macs(bands, 32, d_state=8, d_conv=3)
    # Complementary role refiner.
    macs += conv_macs(height, width, 2, 8, kernel=3)
    macs += conv_macs(height, width, 8, 1)
    # Reference and change reductions.
    macs += conv_macs(height, width, 2 * bands, dim)
    macs += conv_macs(height, width, bands, dim)

    # Horizontal/vertical topology paths; each is bidirectional.
    macs += 2 * bidirectional_mamba_macs(pixels, dim, d_state=16, d_conv=4)
    role_hidden = 16
    per_path_role = (linear_macs(pixels, 3, role_hidden) +
                     linear_macs(pixels, role_hidden, dim) +
                     linear_macs(pixels, role_hidden, dim) +
                     linear_macs(pixels, dim, dim))
    macs += 2 * per_path_role

    macs += bddm_macs(height, width, bands, dim)
    macs += 2 * feature_fusion_macs(height, width, dim)
    macs += prcf_macs(height, width, dim)
    # Classifier: 128 -> 64 -> 32 -> 2.
    macs += linear_macs(pixels, 2 * dim, dim)
    macs += linear_macs(pixels, dim, dim // 2)
    macs += linear_macs(pixels, dim // 2, 2)
    return int(parameters), int(macs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analytical complexity profile for final RAAM-Net")
    parser.add_argument("--scenes", nargs="*", choices=DEFAULT_SCENES, default=list(DEFAULT_SCENES))
    args = parser.parse_args()
    print("| Dataset | Params | MACs (scan excluded) | FLOPs (2xMACs) |")
    print("|---|---:|---:|---:|")
    for name in args.scenes:
        height, width, bands = DEFAULT_SCENES[name]
        parameters, macs = profile_scene(name, height, width, bands)
        print(f"| {name} | {readable(parameters)} | {readable(macs)} | {readable(2.0 * macs)}", flush=True)


if __name__ == "__main__":
    main()
