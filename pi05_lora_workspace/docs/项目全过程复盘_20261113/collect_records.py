"""Read local project notes and public dialogue messages for a sourced retrospective."""
import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HOME = Path('/home/lp531')
OUT = Path(__file__).resolve().parent
MATERIAL = OUT/'核对材料'
MATERIAL.mkdir(exist_ok=True)
roots = [HOME/p for p in ['ros2_ws/src/z1_ros2/docs', 'ros2_ws/monitoring',
    'ros2_ws/src/z1_ros2/z1_isaac_bridge', 'ros2_ws/src/z1_ros2/scripts',
    'robot', 'pi05_lora_workspace', 'z1_queue_logs', 'z1-smolvla/docs',
    'smolvla_eval', 'act_eval', 'smolvla_training', 'docs']]
skip = {'lerobot', 'openpi', '.venv', '__pycache__', 'checkpoints', 'smoke_output',
    'node_modules', '.git', 'baseline', 'before', 'backup_two_segment', 'weights',
    '.pytest_cache', 'build', 'sources'}
paths = set()
for root in roots:
    for p in root.rglob('*'):
        if not p.is_file() or p.suffix.lower() not in {'.md', '.txt'}:
            continue
        if p.is_relative_to(OUT) or any(x in skip for x in p.relative_to(root).parts[:-1]):
            continue
        if p.name in {'requirements.txt','CMakeLists.txt','launch_arguments.txt','launch_args.txt','concat.txt','before_concat.txt','after_concat.txt'} or any(x in p.name for x in ['concat','player_pid','playback_pid','player_window']):
            continue
        if p.stat().st_size > 1_000_000:
            continue
        paths.add(p)
for p in [HOME/'ros2_ws/src/z1_ros2/README.md', HOME/'ros2_ws/src/z1_ros2/REPRODUCE.md',
    HOME/'z1-smolvla/README.md', HOME/'z1-smolvla/ARCHITECTURE.md',
    HOME/'z1-smolvla/CURRENT_GRASP_HANDOFF.md',
    HOME/'isaac_assets/z1/scripts/z1_grasp_experiment_notes.md',
    HOME/'isaac_assets/z1/scanned_objects/GRASPABILITY_REPORT.md']:
    if p.exists(): paths.add(p)
seen = {}
index = []
texts = []
for p in sorted(paths):
    raw = p.read_bytes()
    content = raw.decode('utf-8', errors='replace')
    digest = hashlib.sha256(raw).hexdigest()
    duplicate = seen.get(digest, '')
    seen.setdefault(digest, str(p))
    index.append(dict(path=str(p),bytes=len(raw),sha256=digest,duplicate_of=duplicate,
        title=next((s.lstrip('# ') for s in content.splitlines() if s.strip()),'')))
    if not duplicate:
        texts.append('SOURCE: '+str(p)+'\n'+content+'\n')
(MATERIAL/'项目工作记录全文.txt').write_text('\n\n'.join(texts))
with (OUT/'读取来源索引.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(index[0])); w.writeheader();w.writerows(index)
history = [json.loads(s) for s in (HOME/'.codex/history.jsonl').read_text().splitlines()]
daily=defaultdict(list)
for r in history:
    date=datetime.fromtimestamp(r['ts'],ZoneInfo('Asia/Shanghai')).isoformat()
    daily[date[:10]].append(dict(timestamp=date,text=r['text'],session_id=r['session_id']))
for date, rows in daily.items():
    (MATERIAL/('用户决策_'+date+'.txt')).write_text('\n\n'.join(r['timestamp']+'\n'+r['text'] for r in rows))
sessions = sorted((HOME/'.codex/sessions').rglob('*.jsonl'))
public=[]; session_counts=[]; seen_messages=set()
for path in sessions:
    counts=Counter()
    with path.open() as stream:
        for line in stream:
            try:d=json.loads(line)
            except json.JSONDecodeError:continue
            payload=d.get('payload',{})
            if d.get('type')!='response_item' or payload.get('type')!='message':continue
            role=payload.get('role');channel=payload.get('channel') or payload.get('phase')
            if role not in ('user','assistant') or (role=='assistant' and channel not in ('final','final_answer','commentary')):continue
            content='\n'.join(x.get('text','') for x in payload.get('content',[]) if x.get('type') in ('input_text','output_text','text'))
            if not content or (role=='user' and ('<environment_context>' in content or '<permissions instructions>' in content)):continue
            # Only user-visible dialogue is retained. No hidden analysis/tool payloads.
            key=(role,d.get('timestamp'),content)
            if key in seen_messages:continue
            seen_messages.add(key);counts[role]+=1
            public.append(dict(timestamp=d.get('timestamp'),role=role,channel=channel,text=content,source=str(path)))
    session_counts.append(dict(path=str(path),bytes=path.stat().st_size,messages=dict(counts)))
public.sort(key=lambda x:x['timestamp'] or '')
(MATERIAL/'历史公开对话.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in public))
summary=dict(created_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
    scope='Local project work notes and user-visible conversation only; simulation mainline, real robot background only',
    documents_read=len(index),unique_documents=len(texts),document_bytes=sum(r['bytes'] for r in index),
    history_messages=len(history),history_days={k:len(v) for k,v in sorted(daily.items())},
    sessions_read=len(sessions),session_public_messages=len(public),sessions=session_counts)
(OUT/'读取范围.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k!='sessions'},ensure_ascii=False,indent=2))
