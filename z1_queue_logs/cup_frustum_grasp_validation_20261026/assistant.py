#!/usr/bin/python3
"""Experiment-only target sweep using the unchanged production grasp/lift pipeline."""
import os,copy,json,time
from pathlib import Path
R=Path(__file__).parent
os.environ['Z1_ISAAC_GRASP_CONFIG']=str(R/'grasp_physics.json')
import z1_isaac_bridge.grasp_assistant as m
BASE_PROFILE=copy.deepcopy(m.GRASP_PROFILES['cup']);BASE_PHYSICAL=copy.deepcopy(m.OBJECT_GRASP_PROFILES['cup'])

def evidence():
 case=json.loads((R/'current_case.json').read_text());d=R/'cases'/case['id'];d.mkdir(parents=True,exist_ok=True);return case,d

def write(name,value):
 _,d=evidence();p=d/name;t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(value,ensure_ascii=False,indent=2));t.replace(p)

class ProbeAssistant(m.GraspAssistant):
 def _generate_plan(self):
  case,_=evidence();physical=copy.deepcopy(BASE_PHYSICAL);profile=copy.deepcopy(BASE_PROFILE);overrides=case.get('relative_target_overrides',{})
  if 'grasp_height_target_m' in overrides:physical['grasp_height_target_m']=float(overrides['grasp_height_target_m'])
  if case.get('fixed_reference'):
   physical['approach_pitch_min_deg']=-6.5;physical['approach_pitch_max_deg']=-6.5
   profile.update(candidate_azimuth_offsets_deg=(45.,),extra_candidate_azimuths_deg=(),candidate_local_offsets_yz_m=((-0.004,0.),),candidate_count=1)
  profile.update(case.get('profile_overrides',{}))
  for axis in ('x','z'):
   if axis+'_m' in overrides:physical['calibrated_grasp_reference']['object_in_gripper_local_m']['xyz'.index(axis)]=float(overrides[axis+'_m'])
  if 'preload_deg' in overrides:
   import math
   physical['single_contact_preload_rad']=math.radians(float(overrides['preload_deg']));profile['single_contact_preload_rad']=physical['single_contact_preload_rad']
  profile.update(height=physical['grasp_height_target_m'],calibrated_grasp_reference=m._cylinder_reference_at_grasp_height(physical))
  ref=physical['calibrated_grasp_reference']['object_in_gripper_local_m']
  for axis,value in zip('xyz',ref):profile['desired_center_'+axis+'_m']=value
  profile.update(insertion_center_x_m=ref[0],dynamic_insertion_stop_x_m=ref[0],entry_center_z_m=ref[2])
  m.GRASP_PROFILES['cup']=profile;m.OBJECT_GRASP_PROFILES['cup']=physical
  write('configured.json',dict(case=case,profile=profile,physical=physical))
  if case.get('reuse_approach'):
   from moveit_msgs.msg import RobotTrajectory,RobotState
   from moveit_msgs.srv import GetStateValidity
   from rosidl_runtime_py.set_message import set_message_fields
   import numpy as np
   reference=json.loads((R/'safe_reference.json').read_text());trajectory=RobotTrajectory();set_message_fields(trajectory,reference['display']['trajectory'][0]);candidate=reference['candidate'];joint=trajectory.joint_trajectory
   actual=self._current_object_position('cup')
   if np.linalg.norm(np.array(actual)-reference['cup_position_m'])>.002:raise RuntimeError('Cached path requires the identical cup position')
   with self._lock:start=np.array([self._joint_positions[n] for n in joint.joint_names])
   if np.max(np.abs(start-np.array(joint.points[0].positions)))>.02:raise RuntimeError('Cached path requires the original Home start')
   self._set_bottle_finger_collision(False)
   client=self.create_client(GetStateValidity,'/check_state_validity')
   if not client.wait_for_service(timeout_sec=5):raise RuntimeError('MoveIt path validity service unavailable')
   checked=0;previous=start
   for point in joint.points:
    goal=np.array(point.positions)
    for a in np.linspace(0,1,max(2,int(np.ceil(np.max(np.abs(goal-previous))/.02))+1)):
     q=previous+(goal-previous)*a;request=GetStateValidity.Request();request.group_name='arm';request.robot_state=RobotState();request.robot_state.is_diff=True;request.robot_state.joint_state.name=list(joint.joint_names)+['jointGripper'];request.robot_state.joint_state.position=q.tolist()+[self.gripper_open]
     answer=self._wait_future(client.call_async(request),4.)
     if answer is None or not answer.valid:raise RuntimeError('Cached approach collision check rejected at state '+str(checked))
     checked+=1
    previous=goal
   with self._lock:self._approved_trajectory=trajectory;self._approved_target='cup';self._grasp_candidate=copy.deepcopy(candidate)
   self._set_status('PLAN_READY','实验：相同参考轨迹已重新通过逐状态碰撞检查',True)
   write('candidate.json',dict(candidate=candidate,message='Reused identical safe trajectory',collision_checked_states=checked,source=reference['source']))
   return 'Identical approach revalidated: '+str(checked)+' states'
  result=super()._generate_plan()
  with self._lock:write('candidate.json',dict(candidate=self._grasp_candidate,message=result))
  return result
 def _execute_gripper_close_until_obstructed(self,*args,**kwargs):
  angle=super()._execute_gripper_close_until_obstructed(*args,**kwargs);write('contact_angle.json',dict(angle_rad=angle,time=time.time()));return angle
 def _hold_gripper(self,target,*args,**kwargs):
  _,d=evidence()
  with (d/'hold_targets.jsonl').open('a') as f:f.write(json.dumps(dict(target_rad=target,time=time.time()))+'\n')
  return super()._hold_gripper(target,*args,**kwargs)
 def _guided_grasp(self):
  with self._lock:write('before_grasp.json',dict(task_status=self._task_status,candidate=self._grasp_candidate))
  try:
   result=super()._guided_grasp()
   with self._lock:write('grasp_result.json',dict(success=True,message=result,task_status=self._task_status,scene_status=self._scene_grasp_status))
   return result
  except Exception as exc:
   with self._lock:write('grasp_result.json',dict(success=False,error=str(exc),task_status=self._task_status,scene_status=self._scene_grasp_status))
   raise
m.GraspAssistant=ProbeAssistant
m.main()
