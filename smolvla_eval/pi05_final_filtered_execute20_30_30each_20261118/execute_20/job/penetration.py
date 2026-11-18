"""Summarize tabletop depth after the policy starts, excluding reset samples."""
import json

def measure_penetration(prediction_path, depth_path):
    if not prediction_path.is_file():
        return 0, None
    with prediction_path.open() as predictions:
        first = next(predictions, None)
    if first is None:
        return 0, None
    start = json.loads(first)['sim_stamp_s']
    count = 0
    left, right = 0.0, 0.0
    with depth_path.open() as depths:
        for line in depths:
            sample = json.loads(line)
            if sample['simulation_stamp_s'] < start:
                continue
            count += 1
            left = max(left, sample['left_depth_m'])
            right = max(right, sample['right_depth_m'])
    return count, [left, right] if count else None
