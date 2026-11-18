"""Explicit v2 migration: preserve successful clips and record every changed task."""
import copy
from core import digest

def revised_plan(plan,done):
 new=copy.deepcopy(plan)
 if plan.get('version',1)>=2:raise ValueError('plan already revised; refusing repeated shrink')
 for task in new['tasks']:
  if task['id'] in done:continue
  original=copy.deepcopy(task)
  task['offset_approach_m']=[v*.5 for v in task['offset_approach_m']]
  task['rotation_tool_deg']=[v*.5 for v in task['rotation_tool_deg']]
  # The blue box blocks layout generation here, before any offset is applied.
  col,row=task['grid']
  if row==4 and col<3:task['grid']=[[2,2],[2,0],[3,3]][col]
  if original['id']=='apple_left_tilt_002':task['grid']=[3,2]
  if original['id']=='apple_left_002':task['grid']=[3,3]
  if original['id']=='apple_close_ready_002':task['grid']=[3,3]
  task['original_task']=original
  task['revision']=2
  task['preparation_retreat_m']=.004
  task['revision_reason']='Reduced perturbation after repeated failures; blocked/edge cells remapped to successful train cells.'
 new.update(version=2,parent_plan_hash=plan['hash'],accepted_plan_hashes=[plan['hash']]+plan.get('accepted_plan_hashes',[]),
   amplitudes=dict(lateral_m=[.0025,.005],shallow_m=[.0035,.007],deep_m=[.001,.002],world_z_m=[.001,.002],tilt_deg=[1.5,2.5]),
   note='v2 modifies only unfinished tasks. Original successful clips retain their original metadata and hash. Spatial quotas are no longer three per original cell.')
 new.pop('hash',None);new['hash']=digest(new)
 return new
