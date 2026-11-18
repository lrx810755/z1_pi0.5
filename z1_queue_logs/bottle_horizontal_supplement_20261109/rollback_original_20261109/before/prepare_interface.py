from pathlib import Path
import json,shutil,hashlib,datetime
R=Path(__file__).resolve().parent;C=Path('/home/lp531/robot/cup_cylinder_horizontal200_20261108');P=Path('/home/lp531/ros2_ws/src/z1_ros2/z1_isaac_bridge')
old=C.name;new=R.name
originals=[Path('/home/lp531/SH/start_smolvla_bottle_three_camera_scene.sh'),Path('/home/lp531/SH/start_cup_cylinder_horizontal200_20261108.sh'),C/'assistant.py',C/'collection.launch.py',C/'grasp_physics.json',P/'z1_isaac_bridge/grasp_assistant.py',P/'config/grasp_physics.json']
(R/'original_interface_hashes.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in originals},indent=2))
for name in ['assistant.py','core.py','geometry.py','move_group.launch.py','collection.launch.py','start_collection.sh','collect.sh','startup.py','set_collection_column.py','failure_recorder.py','test_core.py','test_v2_runtime.py','test_column_setter.py','test_startup.py','test_failure_archive.py']:
 s=(C/name).read_text().replace(old,new)
 if name in ['assistant.py','set_collection_column.py','test_column_setter.py']:
  s=s.replace('cup_collection_column','bottle_collection_column')
 if name=='assistant.py':
  s=s.replace("'cup'","'beverage_bottle'").replace('requires cup and link06','requires beverage_bottle and link06').replace('Cup must','Bottle must').replace('Cup moved','Bottle moved')
  s=s.replace('Settings(speed_base=self.grasp_alignment_velocity_scale)','Settings(speed_base=self.grasp_alignment_velocity_scale,maximum_axis_pitch_deg=8.)')
 if name=='test_v2_runtime.py':s=s.replace("n.target_object='cup'","n.target_object='beverage_bottle'")
 if name=='collect.sh':
  s=s.replace('target_selection="${5:-all}"','target_selection="${5:-beverage_bottle}"')
  s=s.replace('if [[ "$target" == cup ]]; then','if [[ "$target" == beverage_bottle ]]; then')
  s=s.replace('case "$target_selection" in','[[ "$target_selection" == beverage_bottle ]] || { echo "独立入口仅支持 beverage_bottle" >&2; exit 2; }\ncase "$target_selection" in',1)
 if name=='start_collection.sh':
  s=s.replace('target_selection="${7:-all}"','target_selection="${7:-beverage_bottle}"')
  s=s.replace('case "$target_selection" in','[[ "$target_selection" == beverage_bottle ]] || { echo "独立入口仅支持 beverage_bottle" >&2; exit 2; }\ncase "$target_selection" in',1)
  s=s.replace('task_gui_port:=8837','task_gui_port:=8839').replace('scene=(ros2 launch "$log_dir/collection.launch.py"','scene=(ros2 launch "'+str(R/'collection.launch.py')+'"')
 if name=='startup.py':s=s.replace('127.0.0.1:8837','127.0.0.1:8839')
 if name=='collect.sh':s=s.replace('--port 8837','--port 8839')
 if name=='collection.launch.py':s=s.replace('default_value="apple"','default_value="beverage_bottle"').replace('default_value="8767"','default_value="8839"')
 (R/name).write_text(s)
 if name.endswith('.sh'):(R/name).chmod(0o755)
shutil.copy2(P/'z1_isaac_bridge/grasp_assistant.py',R/'base_assistant_snapshot.py')
shutil.copy2(P/'config/grasp_physics.json',R/'grasp_physics.json')
manifest=dict(created_at=datetime.datetime.now().astimezone().isoformat(),target='beverage_bottle',columns=list(range(5)),rows=list(range(4)),quota_per_cell_default=10,total_target_default=200,frames_per_second=10,ros_domain_id=136,gui_port=8839,headless=False,auto_started=False,first_column_policy='no_retreat_only',other_columns_policy=['direct_low_all_candidates','via_high','no_retreat_fallback'],retreat_candidates_mm=[50,60,80,100],high_offsets_mm=[80,120],max_axis_pitch_deg=8,final_velocity_scale=.064,grasp_profile='unchanged beverage_bottle profile',bottle_release_policy='original descent to115mm; no cup-specific direct-release override',dataset_root='/home/lp531/smolvla_data/'+new,failure_root='/home/lp531/smolvla_eval/'+new+'_failures')
(R/'collection_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
print('Private bottle interface files prepared')
