from pathlib import Path
import json
R=Path(__file__).parent;source=R/'cases/reference_capture2';display=json.loads((source/'approach_reference.json').read_text());candidate=json.loads((source/'candidate.json').read_text())['candidate'];assert max(abs(x) for x in display['trajectory'][0]['joint_trajectory']['points'][0]['positions'])<.02
(R/'safe_reference.json').write_text(json.dumps(dict(display=display,candidate=candidate,cup_position_m=[.52,-.10,0],source=str(source)),indent=2))
p=R/'assistant.py';s=p.read_text();needle="  result=super()._generate_plan()"
branch='''  if case.get('reuse_approach'):
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
'''
assert needle in s;s=s.replace(needle,branch+needle);p.write_text(s)
rows=[dict(id='paired_preload_'+str(deg).replace('.','p'),cup_position_m=[.52,-.10,0],target_grid=[2,1],fixed_reference=True,reuse_approach=True,relative_target_overrides={'preload_deg':deg}) for deg in [3.,2.,1.,.5]];(R/'paired_preload.json').write_text(json.dumps(rows,indent=2))
