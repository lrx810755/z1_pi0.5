"""Verify saved recordings without changing their frames or quota."""
import json,time,os,signal,collections
from pathlib import Path
r=Path(__file__).resolve().parent
plan=json.loads((r/'collection_plan.json').read_text());data=Path(plan['raw_root'])
seen=set()
while True:
    for path in sorted(data.glob('episode_*/episode_summary.json')):
        if path.parent.name in seen:continue
        s=json.loads(path.read_text())
        if not s.get('task_success'):continue
        frames=[json.loads(x) for x in (path.parent/'frames.jsonl').read_text().splitlines()]
        assert len(frames)==s['frame_count'] and len(frames)>0
        for f in frames:
            for camera in ['wrist','global','global_right']:
                assert (path.parent/f['observation.images.'+camera]).is_file()
        info=dict(episode=str(path.parent),frames=len(frames),images_checked=len(frames)*3,
                  task_success=s['task_success'],physical_grasp_ok=s['physical_grasp_ok'],
                  grid=s['scene_randomization']['target_grid'],verified_at=time.time())
        with (r/'saved_verified.jsonl').open('a') as f:f.write(json.dumps(info)+'\n')
        if not (r/'first_saved_verified.json').exists():
            (r/'first_saved_verified.json').write_text(json.dumps(info,indent=2))
        seen.add(path.parent.name)
    completed=r/'completed.json'
    if completed.exists():
        final=json.loads(completed.read_text());final['recordings_verified']=len(seen)
        counts=collections.Counter()
        for path in data.glob('episode_*/episode_summary.json'):
            s=json.loads(path.read_text())
            if s.get('task_success'):counts[tuple(s['scene_randomization']['target_grid'])]+=1
        final['grid_counts']={f'{c},{row}':counts[c,row] for row in range(4) for c in range(5)}
        final['quota_verified']=all(counts[c,row]==8 for row in range(4) for c in range(5))
        (r/'final_verified.json').write_text(json.dumps(final,indent=2))
        try:
            pid=int((r/'audit_pid').read_text())
            if b'audit_physics.py' in Path(f'/proc/{pid}/cmdline').read_bytes():os.kill(pid,signal.SIGTERM)
        except (FileNotFoundError,ProcessLookupError):pass
        break
    time.sleep(5)
