"""Preserve requested XY errors while preparing outside the cup's solid body."""
import copy
import numpy as np
from scipy.spatial.transform import Rotation


def elevated_approach_routes(original, reference, orientation):
    reference = np.asarray(reference, dtype=float)
    bad_original = np.asarray(original['bad_position'], dtype=float)
    forward = Rotation.from_quat(orientation).apply([1., 0., 0.])
    forward[2] = 0.
    forward /= np.linalg.norm(forward)
    for clearance in (.055, .04, .025):
        for retreat in (.06, .045, .03, .015):
            route = copy.deepcopy(original)
            bad = bad_original.copy()
            bad[2] += clearance
            above = bad.copy()
            above[2] += .02
            route.update(strategy='elevated_approach_' + str(clearance) + '_retreat_' + str(retreat),
                additional_preparation_height_m=clearance, outside_retreat_m=retreat,
                horizontal_deviation_preserved=True, bad_position=bad.tolist())
            route['preparation'] = [
                dict(stage='prepare_above_deviation', position=above.tolist(), orientation=orientation, linear=False),
                dict(stage='prepare_elevated_deviation', position=bad.tolist(),
                     orientation=original['bad_orientation'], linear=True),
            ]
            route['recovery'] = [
                dict(stage='recover_retreat_above_rim',
                     position=[float(reference[0] - retreat * forward[0]),
                               float(reference[1] - retreat * forward[1]), float(bad[2])],
                     orientation=orientation, linear=True),
                dict(stage='recover_lower_outside_rim', position=(reference - retreat * forward).tolist(),
                     orientation=orientation, linear=True),
                dict(stage='recover_final_reference', position=reference.tolist(), orientation=orientation, linear=True),
            ]
            yield route
            # Some inner-workspace poses have no IK at a constant raised Z.
            # Let the full collision planner find a route to the low outside
            # waypoint; retain the original final horizontal entry and checks.
            planned_return = copy.deepcopy(route)
            planned_return['strategy'] += '_planned_return'
            planned_return['recovery'] = planned_return['recovery'][1:]
            planned_return['recovery'][0].update(stage='recover_planned_outside_rim', linear=False)
            yield planned_return
