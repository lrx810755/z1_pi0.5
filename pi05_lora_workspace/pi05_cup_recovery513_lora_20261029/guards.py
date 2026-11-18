"""Fail closed around native PI05 silent weight-load fallback and validation logs."""
import math
import re


def strict_base_load(factory, policy_class):
    """Require successful strict state loading even if native loader swallows errors.

    Scope the hook to this factory call; never edit the shared LeRobot source.
    Used for both fresh training and adapter reload's underlying base load.
    """
    inherited = 'load_state_dict' not in policy_class.__dict__
    original = policy_class.load_state_dict
    evidence = {}

    def checked(policy, state_dict, strict=True, assign=False):
        result = original(policy, state_dict, strict=True, assign=assign)
        if result.missing_keys or result.unexpected_keys:
            raise RuntimeError('Native strict base loading left unmatched keys')
        evidence.update(strict_base_load=True, loaded_keys=len(state_dict))
        return result

    policy_class.load_state_dict = checked
    try:
        policy = factory()
    finally:
        if inherited:
            delattr(policy_class, 'load_state_dict')
        else:
            policy_class.load_state_dict = original
    if not evidence:
        raise RuntimeError('No successful strict base weight load; native PI05 may have returned random weights')
    return policy, evidence


def validation_evidence(text):
    if 'eval_loss=' not in text:
        return None
    match = re.search(r'step (\d+): eval_loss=(\S*)', text)
    if not match:
        raise ValueError('Unrecognized validation result')
    microsteps, loss = int(match.group(1)), float(match.group(2))
    if not math.isfinite(loss):
        raise RuntimeError('Nonfinite validation loss')
    return dict(microsteps=microsteps, optimizer_updates=microsteps//8, heldout_loss=loss)
