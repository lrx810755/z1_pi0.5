import copy,json,os,time,sys
from pathlib import Path
import numpy as np

def ctx():return json.loads(Path(os.environ['Z1_PUSH_RECOVERY_CURRENT']).read_text())
def path(c):return Path(os.environ['Z1_PUSH_RECOVERY_CURRENT']).parent/f"evidence_{c['attempt_id']}.json"
def write(p,d):
 tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(d,ensure_ascii=False,indent=2));tmp.replace(p)
def main():
 import z1_isaac_bridge.grasp_assistant as m
 sys.path.append('/home/lp531/ros2_ws/src/z1_ros2/scripts/cup_correction_fragments_20261019')
 from geometry import Geometry
 m.GRASP_PROFILES=copy.deepcopy(m.GRASP_PROFILES);m.OBJECT_GRASP_PROFILES=copy.deepcopy(m.OBJECT_GRASP_PROFILES)
 m.GRASP_PROFILES['cup'].update(roll_deg=-90.,pregrasp_target_command='unfreeze',freeze_before_approach=False,single_pass_box_alignment=False,max_joint_delta_rad=3.)
 m.OBJECT_GRASP_PROFILES['cup'].update(mirror_wrist_roll_candidates=False,loaded_transport_velocity_scale=.05,loaded_transport_acceleration_scale=.02)
 m.BOX_TRANSPORT_XY_TOLERANCE_M=dict(m.BOX_TRANSPORT_XY_TOLERANCE_M,cup=.020)
 m.GRASP_CONFIG=copy.deepcopy(m.GRASP_CONFIG);m.GRASP_CONFIG['physical_test_lift']['goal_directed_max_attempts']=10
 Base=m.GraspAssistant
 class Assistant(Base):
  def __init__(self):
   self.meta=None;super().__init__();self.geometry=Geometry()
  def event(self,event,**kw):
   if self.meta is None:return
   self.meta['events'].append(dict(event=event,time=self.get_clock().now().nanoseconds*1e-9,**kw));write(path(self.ctx),self.meta)
  def interval(self,stage,start,end):
   self.meta['recovery_stages'].append(dict(stage=stage,start=start,end=end));self.meta.setdefault('recovery_intervals_s',[]).append(dict(start=start,end=end));write(path(self.ctx),self.meta)
  def _command_scene_grasp(self,command,*args,**kw):
   if command in ('freeze','prepare_cup_shift'):raise RuntimeError('Freeze/relocation forbidden in live policy recovery')
   return super()._command_scene_grasp(command,*args,**kw)
  def _generate_plan(self):
   self.ctx=ctx();self.meta=json.loads(path(self.ctx).read_text());self.event('expert_recovery_start')
   self.meta['recovery_start_sim_s']=self.get_clock().now().nanoseconds*1e-9
   # First expert motion lifts the open fingers away from the already-pushed cup.
   before,orientation=self._current_ee_pose();candidate=list(before);candidate[2]+=.045
   self.event('retreat_start',before=list(before),target=candidate)
   start=self.get_clock().now().nanoseconds*1e-9
   for fraction in [1/3,2/3,1.]:
    waypoint=list(before);waypoint[2]+=.045*fraction
    self._execute_manual_pose(tuple(waypoint),orientation,'推杯后抬高退让分段')
   self.interval('push_retreat',start,self.get_clock().now().nanoseconds*1e-9)
   self.event('retreat_complete',position=list(self._current_ee_pose()[0]))
   result=super()._generate_plan();self.event('relocalized_plan_ready',cup_position=list(self._current_object_position('cup')))
   return result
  def _execute_plan(self):
   start=self.get_clock().now().nanoseconds*1e-9;result=super()._execute_plan();self.interval('push_realign',start,self.get_clock().now().nanoseconds*1e-9);return result
  def _guided_grasp(self):
   start=self.get_clock().now().nanoseconds*1e-9;result=super()._guided_grasp();end=self.get_clock().now().nanoseconds*1e-9
   self.interval('push_stable_grasp',start,end);self.meta.update(recovery_end_sim_s=end,stable_grasp_end_sim_s=end,recovery_complete=True);self.event('stable_grasp_complete');return result
  def _execute_gripper_target(self,state,target):
   self.event('gripper_intent',state=state,target=float(target));return super()._execute_gripper_target(state,target)
  def _execute_gripper_close_until_obstructed(self,target,target_object,**kw):
   self.event('gripper_intent',state='close',target=float(target));return super()._execute_gripper_close_until_obstructed(target,target_object,**kw)
  def _execute_manual_pose(self,position,orientation,description,*args,**kw):
   if description!='释放后垂直撤离':return super()._execute_manual_pose(position,orientation,description,*args,**kw)
   before=self._current_ee_pose()[0]
   try:
    for f in [1/3,2/3,1.]:result=super()._execute_manual_pose(tuple(float(before[i]+f*(position[i]-before[i])) for i in range(3)),orientation,description+'分段',*args,**kw)
    return result
   finally:
    after=self._current_ee_pose()[0];self.meta['withdrawal_rise_m']=float(after[2]-before[2]);self.event('withdrawal_measured',before=list(before),after=list(after))
 m.GraspAssistant=Assistant;m.main()
