from pathlib import Path
import json,hashlib,subprocess,datetime
R=Path(__file__).parent
reg=[]
for n in [2,5,8]:
 cmd=['/usr/bin/python3',str(R/'assert_occlusion.py'),str(R/f'fixed_trial{n:02d}'),'original'];out=subprocess.check_output(cmd,text=True,env={'PYTHONNOUSERSITE':'1','PATH':'/usr/bin:/bin'});row=json.loads(out);row['trial']=n;reg.append(row)
scene=Path('/home/lp531/ros2_ws/src/z1_ros2/isaac_assets/z1/generated_dual_camera/smolvla_cup_only_frustum_20261025.usd');sha=hashlib.sha256(scene.read_bytes()).hexdigest();assert sha=='547b0902f99ea4aa537282fdbcfb57c6576f7697b3f1e3bb36fe434319b1509a'
for name in ['actual_runtime_smoke.log','default_runtime_smoke.log']:
 log=(R/name).read_text();assert "Close-range glass occlusion: {'/rtx/translucency/worldEps': 0.0001}" in log,name;assert '[Error]' not in log,name
replay=json.loads((R/'dynamic_replay/manifest.json').read_text());assert sum(x['frames'] for x in replay['clips'])==1040
report=dict(verified_at=datetime.datetime.now().astimezone().isoformat(),renderer_fix={'/rtx/translucency/worldEps':.0001},static_rgb_occlusion_regression=reg,dynamic_replay=dict(paired_poses=520,views=3,duration_s=104,trials=[2,5,8,12],is_act_evaluation=False,is_physics_replay=False),source_scene_sha256=sha,source_scene_unchanged=True,actual_entrypoint_smoke_checks=['default scene','cup-only scene'],scope='Verified renderer occlusion correction; physical contact penetrations are audited separately and were not modified.')
(R/'修复验证结果.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));print(json.dumps(report,ensure_ascii=False,indent=2))
old=R/'遮挡核查结论.md';text=old.read_text();notice='> 后续核查已确认 Trial 05、08 等存在真实的 RGB 渲染遮挡错误，并已修复。本文只解释第18次的特定桌面画面，不能用于否定其他片段的问题。完整结论见《穿透问题_根因与修复报告.md》。\n\n'
if not text.startswith('> 后续'):old.write_text(notice+text)
record=dict(renderer_fix=report['renderer_fix'],historic_30_trial_results_used_default_worldEps=1.,historic_results_and_videos_unchanged=True,fixed_runtime_entrypoints=json.loads((R/'fixed_entrypoints.json').read_text()),helper='/home/lp531/ros2_ws/src/z1_ros2/isaac_assets/z1/scripts/render_occlusion.py',evidence_report=str(R/'穿透问题_根因与修复报告.md'))
Path('/home/lp531/smolvla_eval/act_cup_only_frustum_steps15_paired30_20261025/render_fix_20261026.json').write_text(json.dumps(record,ensure_ascii=False,indent=2))
