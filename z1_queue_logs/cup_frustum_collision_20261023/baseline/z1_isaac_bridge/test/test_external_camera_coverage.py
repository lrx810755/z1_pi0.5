"""Keep the complete bottle sampling volume inside both external cameras."""
import itertools
import json
from pathlib import Path
import numpy as np


def test_bottle_workspace_and_height_fit_both_external_views():
    config = Path(__file__).parents[1] / 'config'
    cameras = json.loads((config / 'external_cameras.json').read_text())
    physics = json.loads((config / 'grasp_physics.json').read_text())
    workspace = physics['scene_randomization']
    radius = workspace['object_footprint_radii_m']['beverage_bottle']
    x0, x1 = workspace['workspace_x_m']
    y0, y1 = workspace['workspace_y_m']
    for camera in cameras.values():
        pos = np.array(camera['position_m'])
        forward = np.array(camera['look_at_m']) - pos
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, [0., 0., 1.])
        right /= np.linalg.norm(right)
        down = np.cross(forward, right)
        rotation = np.column_stack((right, down, forward))
        qx, qy, qz, qw = camera['orientation_xyzw']
        quaternion_rotation = np.array([
            [1-2*(qy*qy+qz*qz), 2*(qx*qy-qz*qw), 2*(qx*qz+qy*qw)],
            [2*(qx*qy+qz*qw), 1-2*(qx*qx+qz*qz), 2*(qy*qz-qx*qw)],
            [2*(qx*qz-qy*qw), 2*(qy*qz+qx*qw), 1-2*(qx*qx+qy*qy)]])
        np.testing.assert_allclose(quaternion_rotation, rotation, atol=1e-10)
        # Includes bottle radius and 20 cm of height above the tabletop.
        for point in itertools.product((x0-radius, x1+radius), (y0-radius, y1+radius), (0., .20)):
            x, y, z = rotation.T @ (np.array(point) - pos)
            assert z > .05
            u = camera['fx'] * x / z + camera['cx']
            v = camera['fy'] * y / z + camera['cy']
            assert 10 < u < camera['width'] - 10
            assert 10 < v < camera['height'] - 10
