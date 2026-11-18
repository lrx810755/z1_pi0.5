"""Exercise the real forward method without downloading a VLM checkpoint.

Only dependency-heavy imports are bypassed; the production method is compiled
unchanged and receives differentiable synthetic per-element losses.
"""
import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


def forward_method():
    source = Path(__file__).resolve().parents[2] / "src/lerobot/policies/smolvla/modeling_smolvla.py"
    tree = ast.parse(source.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "SmolVLAPolicy")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "forward")
    method.decorator_list = []
    namespace = {"torch": torch, "Tensor": torch.Tensor, "ACTION": "action", "OBS_STATE": "observation.state"}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
    return namespace["forward"]


def run_loss(padding):
    values = torch.ones(2, 3, 32)
    values[:, :, 7:] = 1000
    values[0, 1:, :7] = 50
    values.requires_grad_()
    policy = SimpleNamespace(
        config=SimpleNamespace(adapt_to_pi_aloha=False, max_action_dim=32),
        normalize_inputs=lambda batch: batch,
        normalize_targets=lambda batch: batch,
        prepare_images=lambda batch: (None, None),
        prepare_state=lambda batch: None,
        prepare_language=lambda batch: (None, None),
        prepare_action=lambda batch: batch["action"],
        model=SimpleNamespace(forward=lambda *args: values),
    )
    batch = {"action": torch.zeros(2, 3, 7)}
    if padding is not None:
        batch["action_is_pad"] = padding
    loss, _ = forward_method()(policy, batch)
    loss.backward()
    return loss.item(), values.grad


def test_padding_does_not_affect_loss_or_gradient():
    loss, grad = run_loss(torch.tensor([[False, True, True], [False, False, False]]))
    assert loss == pytest.approx(1.0)
    assert torch.count_nonzero(grad[:, :, 7:]) == 0
    assert torch.count_nonzero(grad[0, 1:]) == 0
    assert grad[1, 0, 0].item() == pytest.approx(1 / 28)


def test_fully_padded_batch_has_finite_zero_loss():
    loss, grad = run_loss(torch.ones(2, 3, dtype=torch.bool))
    assert loss == 0
    assert torch.count_nonzero(grad) == 0


def test_missing_optional_mask_keeps_real_dimensions():
    loss, grad = run_loss(None)
    assert loss == pytest.approx((4 + 2 * 50) / 6)
    assert torch.count_nonzero(grad[:, :, 7:]) == 0


def test_invalid_mask_shape_rejected():
    with pytest.raises(ValueError, match="action_is_pad"):
        run_loss(torch.zeros(2, 2, dtype=torch.bool))
