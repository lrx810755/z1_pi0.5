"""Frame-anchor sampling weights; no action labels or losses are modified."""
import numpy as np


def build_phase_weights(rows, close, opening, fps=10, stable_frames=5):
    if not (0 < close < opening < len(rows)) or stable_frames < 1:
        raise ValueError('Invalid close/release boundaries')
    consecutive = 0
    stable = None
    for i in range(close, opening):
        row = rows[i]
        good = row.get('physical_grasp_ok') is True and row.get('object_touching_table') is False
        consecutive = consecutive + 1 if good else 0
        if consecutive >= stable_frames:
            stable = i
            break
    if stable is None:
        raise ValueError('No confirmed stable lift before release')
    weights = np.ones(len(rows), dtype=np.float64)
    start = max(0, close - int(fps))
    weights[start:close] = 1.5
    weights[close:stable+1] = 2.
    return weights, dict(preclose_start_frame=start, close_frame=close,
                         stable_lift_confirmed_frame=stable, open_frame=opening,
                         stable_frames=stable_frames, stable_rule='physical_grasp_ok AND not object_touching_table')


def probabilities(correction, phase_weights):
    correction = np.asarray(correction, dtype=bool)
    weights = np.array(phase_weights, dtype=np.float64, copy=True)
    if weights.ndim != 1 or correction.shape != weights.shape or not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError('Invalid sampling weights')
    for group, mass in [(False, .7), (True, .3)]:
        mask = correction == group
        if not mask.any():
            raise ValueError('Both source pools must contain training windows')
        weights[mask] *= mass / weights[mask].sum()
    return weights
