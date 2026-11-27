"""Compare the released full model with an immutable original Stage39 tree.

Run on an idle CUDA device with its validated mamba-ssm environment:
python tests/verify_stage39_reference.py --reference-root /path/to/Stage39
"""
import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from raamnet import RAAMNet
from raamnet.engine import seed_everything


def run(reference_root):
    sys.path.insert(0, str(Path(reference_root).resolve()))
    from fusion_models import build_model
    cases = []
    for bands, parameters in ((155, 213000), (154, 212808), (198, 221256)):
        seed_everything(42)
        reference = build_model(bands, "P_full")
        expected_rng = torch.get_rng_state().clone()
        seed_everything(42)
        released = RAAMNet(bands)
        assert torch.equal(torch.get_rng_state(), expected_rng)
        old, new = reference.state_dict(), released.state_dict()
        assert old.keys() == new.keys()
        assert all(torch.equal(old[k], new[k]) for k in old)
        assert sum(p.numel() for p in released.parameters()) == parameters
        cases.append(dict(bands=bands, parameters=parameters, initial_tensors_and_rng_identical=True))
    reference = reference.cuda()
    released = released.cuda()
    for h, w in ((8, 10), (9, 11)):
        first = torch.randn(1, 198, h, w, device="cuda")
        second = torch.randn_like(first)
        target = torch.randint(0, 2, (1, h, w), device="cuda")
        reference.train()
        released.train()
        seed_everything(91)
        old_logits = reference(first, second)
        old_loss = torch.nn.functional.cross_entropy(old_logits, target)
        old_loss.backward()
        seed_everything(91)
        new_logits = released(first, second)
        new_loss = torch.nn.functional.cross_entropy(new_logits, target)
        new_loss.backward()
        assert torch.equal(old_logits, new_logits)
        assert torch.equal(old_loss, new_loss)
        old_params = dict(reference.named_parameters())
        # Native Mamba CUDA reductions can vary at the last bits even when
        # repeating the same original model. Record that control explicitly.
        control = build_model(198, "P_full").cuda().train()
        control.load_state_dict(reference.state_dict(), strict=True)
        seed_everything(91)
        control_logits = control(first, second)
        torch.nn.functional.cross_entropy(control_logits, target).backward()
        assert torch.equal(control_logits, old_logits)
        control_params = dict(control.named_parameters())
        max_difference = 0.0
        max_repeat_difference = 0.0
        for name, parameter in released.named_parameters():
            expected = old_params[name].grad
            assert (expected is None) == (parameter.grad is None), name
            if expected is not None:
                assert torch.isfinite(parameter.grad).all(), name
                difference = float((parameter.grad - expected).abs().max().item())
                repeat = float((control_params[name].grad - expected).abs().max().item())
                max_difference = max(max_difference, difference)
                max_repeat_difference = max(max_repeat_difference, repeat)
                torch.testing.assert_close(parameter.grad, expected, rtol=1e-5, atol=1e-7, msg=name)
        reference.zero_grad(set_to_none=True)
        released.zero_grad(set_to_none=True)
        cases.append(dict(shape=[h, w], train_logits_and_loss_bit_identical=True,
                          gradients_close=True, gradient_rtol=1e-5, gradient_atol=1e-7,
                          gradient_max_abs_difference=max_difference,
                          original_repeat_gradient_max_abs_difference=max_repeat_difference))
        del control
    result = dict(passed=True, gpu=torch.cuda.get_device_name(), torch=str(torch.__version__), cases=cases)
    print(json.dumps(result, indent=2), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-root", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    result = run(args.reference_root)
    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2))
