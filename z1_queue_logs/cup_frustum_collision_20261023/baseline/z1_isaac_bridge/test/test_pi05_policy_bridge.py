import base64
import numpy as np
import torch
import pytest

from z1_isaac_bridge.pi05_policy_bridge import encode_batch, decode_batch, load_base_policy


def test_seeded_reset_replays_inference_noise():
    from z1_isaac_bridge.pi05_policy_bridge import reset_policy
    from types import SimpleNamespace
    resets = []
    policy = SimpleNamespace(reset=lambda: resets.append(True))
    reset_policy(policy, 123)
    first = torch.randn(50, 7)
    reset_policy(policy, 123)
    assert torch.equal(first, torch.randn(50, 7))
    reset_policy(policy, 456)
    assert not torch.equal(first, torch.randn(50, 7))
    assert len(resets) == 3


def test_rgb_state_and_task_survive_process_boundary():
    image = torch.arange(18, dtype=torch.float32).reshape(1, 3, 2, 3) / 255
    original = {'observation.images.global': image, 'observation.state': torch.arange(7).reshape(1, 7),
                'task': ['Physically grasp the bottle, lift it clear, and place it in the box.']}
    encoded = encode_batch(original)
    restored = decode_batch(encoded)
    assert torch.allclose(restored['observation.images.global'], image)
    assert restored['observation.state'].tolist() == original['observation.state'].tolist()
    assert restored['task'] == original['task']
    assert len(base64.b64decode(encoded['observation.images.global']['data'])) == 18


def test_silent_base_loading_failure_is_fatal():
    class BrokenLoader:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            print('Warning: Could not load state dict: broken weights')
            return object()
    with pytest.raises(RuntimeError, match='refusing inference'):
        load_base_policy(BrokenLoader, 'unused', None)


def test_successful_strict_base_loading_is_accepted():
    model = object()
    class Loader:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            assert kwargs['strict'] and kwargs['local_files_only']
            print('All keys loaded successfully!')
            return model
    assert load_base_policy(Loader, 'unused', None) is model
