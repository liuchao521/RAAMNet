"""Legacy-THOP complexity profile used by the original comparison table.

This is intentionally different from :mod:`profile_complexity`.  It keeps
THOP's native operator hooks and therefore reports only operations observed by
the original THOP protocol.  A shape-only Mamba fused function permits a CPU
trace; its selective scan and fused internal operations are not registered by
THOP and are consequently not counted.  This behaviour matches the legacy
baseline table and should only be used when all compared methods use the same
THOP convention.
"""

from __future__ import annotations

import argparse
import json

import torch
from thop import profile

from raamnet import RAAMNet


def main() -> None:
    parser = argparse.ArgumentParser(description="Legacy THOP profile for RAAM-Net")
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--bands", type=int, required=True)
    args = parser.parse_args()

    # Mamba's CUDA-only fused function is replaced only for this profiling
    # process. THOP then retains its normal Conv2d/Linear accounting and,
    # consistently with the legacy comparison table, does not charge the
    # unregistered fused selective-scan body.
    from mamba_ssm.ops.selective_scan_interface import MambaInnerFn

    def shape_only_inner(ctx, xz, conv_weight, conv_bias, x_proj_weight,
                         dt_proj_weight, out_proj_weight, out_proj_bias, *unused):
        return xz.new_zeros((xz.shape[0], xz.shape[2], out_proj_weight.shape[0]))

    MambaInnerFn.forward = staticmethod(shape_only_inner)
    torch.set_num_threads(1)
    model = RAAMNet(bands=args.bands).eval()
    first = torch.randn(1, args.bands, args.height, args.width)
    second = torch.randn_like(first)
    with torch.no_grad():
        macs, _ = profile(model, inputs=(first, second), verbose=False)
    # THOP's returned ``params`` omits Parameters held by custom Mamba
    # modules. The legacy baseline table used this explicit count as well.
    parameters = sum(parameter.numel() for parameter in model.parameters())
    print("LEGACY_THOP_RESULT " + json.dumps({
        "shape_per_date": [args.height, args.width, args.bands],
        "parameters": int(parameters),
        "macs": float(macs),
        "flops": float(2.0 * macs),
        "selective_scan_counted": False,
        "convention": "native THOP hooks; 2 FLOPs per MAC",
    }), flush=True)


if __name__ == "__main__":
    main()
