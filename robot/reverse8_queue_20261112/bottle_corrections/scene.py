#!/usr/bin/env python3
"""Task-local scene extension: move a bottle only during Home/setup, before recording."""
from pathlib import Path
import sys
source=Path('/home/lp531/ros2_ws/src/z1_ros2/isaac_assets/z1/scripts/run_z1_ros2_scene.py')
sys.path.insert(0,str(source.parent))
code=source.read_text()
reset_needle='        self._clear_frozen_object(publish=False)\n        self._clear_contact_state()\n        sampled = self._sample_randomized_object_poses(seed, level=level)'
assert code.count(reset_needle)==1
code=code.replace(reset_needle,reset_needle.replace('        sampled =', '        self._bottle_shift_attempt = None\n        sampled ='))
needle='            if kind == "freeze":\n'
replacement='''            if kind == "prepare_bottle_shift":
                with open(os.environ['Z1_REVERSE8_CORRECTION_CURRENT']) as stream:
                    ctx=json.load(stream)
                if ctx['task']['kind']!='shifted_target':
                    raise RuntimeError('Prepared bottle shift is restricted to shift tasks')
                if getattr(self,'_bottle_shift_attempt',None)!=ctx['attempt_id']:
                    names=list(self._robot.dof_names)
                    actual=np.asarray(self._robot.get_joint_positions())
                    for name in ['joint1','joint2','joint3','joint4','joint5','joint6']:
                        if abs(float(actual[names.index(name)])-SAFE_HOME_BY_JOINT[name])>.05:
                            raise RuntimeError('Bottle setup relocation requires arm at Home')
                    self._clear_frozen_object(publish=False)
                    path=OBJECT_NAME_TO_PRIM['beverage_bottle'];rigid=self._objects[path]
                    position,orientation=rigid.get_world_pose()
                    delta=np.asarray(ctx['task']['requested_shift_world_m'],dtype=np.float32)
                    if delta.shape!=(3,) or not np.isfinite(delta).all() or abs(float(delta[2]))>1e-8 or np.linalg.norm(delta)>.016:
                        raise ValueError('Invalid bottle setup shift')
                    target=np.asarray(position)+delta
                    bounds=grid_cell_bounds(RANDOMIZATION_CONFIG['workspace_x_m'],RANDOMIZATION_CONFIG['workspace_y_m'],RANDOMIZATION_CONFIG['levels']['full']['target_grid_shape'],ctx['task']['grid'])
                    # grid_cell_bounds returns x/y pairs in the shared sampler.
                    if not (bounds[0][0]<=target[0]<=bounds[0][1] and bounds[1][0]<=target[1]<=bounds[1][1]):
                        raise RuntimeError('Prepared bottle shift would cross the designated dataset grid')
                    rigid.set_world_pose(position=target,orientation=orientation)
                    rigid.set_linear_velocity(np.zeros(3,dtype=np.float32));rigid.set_angular_velocity(np.zeros(3,dtype=np.float32))
                    self._set_object_collisions(path,True);self._set_object_gravity(path,disabled=False);self._set_object_kinematic(path,False)
                    self._bottle_shift_attempt=ctx['attempt_id']
                    self._clear_contact_state()
                self._publish_grasp_status('BOTTLE_SHIFT_READY','Bottle setup shift applied; dynamic physics enabled',target='beverage_bottle')
            elif kind == "freeze":
'''
assert code.count(needle)==1
code=code.replace(needle,replacement)
exec(compile(code,str(source),'exec'),{'__name__':'__main__','__file__':str(source)})
