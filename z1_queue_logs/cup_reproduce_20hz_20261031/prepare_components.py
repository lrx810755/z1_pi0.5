import json,hashlib,ast
from pathlib import Path
R=Path(__file__).resolve().parent
C=R/'correction';P=R/'recovery'
CORR_DATA='/home/lp531/smolvla_data/z1_cup_correction100_20hz_20261031'
PUSH_DATA='/home/lp531/smolvla_data/z1_cup_push_recovery13_20hz_20261031'
def edit(path,old,new):
 s=path.read_text();assert old in s,(path,old[:80]);path.write_text(s.replace(old,new))
# Work only on the copies created for this experiment.
for p in (C/'collector').glob('*.py'):
 edit(p,'/home/lp531/smolvla_data/z1_cup_correction100_full_renderfix_20261027',CORR_DATA) if '/home/lp531/smolvla_data/z1_cup_correction100_full_renderfix_20261027' in p.read_text() else None
base=C/'collector/base.launch.py';s=base.read_text();s=s.replace('/home/lp531/z1_queue_logs/cup_correction100_full_renderfix_20261027/grasp_physics_minus90.json',str(C/'grasp_physics_minus90.json'));base.write_text(s)
p=C/'collector/nodes.py';edit(p,'    Base = module.EpisodeRecorder',f'    sys.path.insert(0,{str(R)!r})\n    from common import guarded_recorder\n    Base = guarded_recorder(module.EpisodeRecorder,"correction")');edit(p,"get('frame_count', 0) >= 3","get('frame_count', 0) >= 6")
p=C/'collector/protocol.py';edit(p,'if tail < 16:','if tail < 32:');edit(p,'At least16 release-tail frames required','At least32 release-tail frames required at20Hz')
p=C/'collector/run.py';edit(p,'PORT = 8807','PORT = 8821');edit(p,"'recording_fps:=10.0'","'recording_fps:=20.0'");edit(p,"'recording_sync_tolerance_s:=0.045'","'recording_sync_tolerance_s:=0.020'");edit(p,"'recording_action_age_s:=0.40'","'recording_action_age_s:=0.050'")
# Recovery only replays the13 exact source candidates accepted into previous PI05 training.
records=json.loads((R/'original513_sources.json').read_text())['sources']['actual_policy_push_recovery']
candidates=[]
for row in records:
 meta=json.loads((Path(row['source_path'])/'correction.json').read_text());c=meta['source_candidate'];assert meta['source_identity']==c['candidate_id']+f"_takeover_{c['takeover_simulation_elapsed_s']:.6f}";candidates.append(c)
assert len(candidates)==len({c['candidate_id'] for c in candidates})==13
(P/'candidates').mkdir(exist_ok=True);(P/'candidates/candidates_manifest.json').write_text(json.dumps(dict(candidates=candidates),indent=2))
for p in (P/'collector').glob('*.py'):
 s=p.read_text().replace('/home/lp531/smolvla_data/z1_cup_policy_push_recovery20_20261028',PUSH_DATA)
 s=s.replace('/home/lp531/z1_queue_logs/cup_correction100_full_renderfix_20261027/collector',str(C/'collector'))
 p.write_text(s)
base=P/'collector/base.launch.py';s=base.read_text().replace('/home/lp531/z1_queue_logs/cup_policy_push_recovery20_20261028_134320/grasp_physics_minus90.json',str(P/'grasp_physics_minus90.json'));base.write_text(s)
p=P/'collector/recorder.py';edit(p,'if len(rows)-release < 16:','if len(rows)-release < 32:');edit(p,'at least16 release-tail frames required','at least32 release-tail frames required at20Hz');edit(p,'production.EpisodeRecorder=make_recorder_class(production.EpisodeRecorder)',f'sys.path.insert(0,{str(R)!r})\n    from common import guarded_recorder\n    production.EpisodeRecorder=make_recorder_class(guarded_recorder(production.EpisodeRecorder,"recovery"))')
p=P/'collector/audit_saved.py';edit(p,'EXPECTED_COUNT=19','EXPECTED_COUNT=13');edit(p,"float(summary.get('fps',0))==10","float(summary.get('fps',0))==20");edit(p,'index*.1','index*.05');edit(p,'continuous10Hz','continuous20Hz');edit(p,'len(rows)-release>=16','len(rows)-release>=32')
p=P/'collector/supervise.py';s=p.read_text().replace('PORT=8814','PORT=8822').replace('recording_fps:=10.0','recording_fps:=20.0').replace('recording_sync_tolerance_s:=0.045','recording_sync_tolerance_s:=0.020').replace('recording_action_age_s:=0.40','recording_action_age_s:=0.050')
cut=s.index('  # Training is launched only');end=s.index("if __name__=='__main__':",cut);s=s[:cut]+" write('workflow_status.json',dict(stage='completed',total=13))\n"+s[end:]
s=s.replace('def main():\n',"def main():\n signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(KeyboardInterrupt()))\n")
p.write_text(s)
for p in R.rglob('*.py'):ast.parse(p.read_text(),filename=str(p))
print('Configured20Hz correction100 and exact recovery13; no training handoff')
