"""Manual stages: freeze reviewed sources, convert, and explicitly start training."""
import argparse
import fcntl
import json
import subprocess
import sys
from pathlib import Path
from config import RUN
from dataset_prepare import sha, write
from quality_decisions import accepted_reverse

def read(p):
    return json.loads(Path(p).read_text())

def resolve(path, edges):
    path=str(Path(path).resolve());seen=set()
    while path in edges:
        if path in seen:raise ValueError('Replacement cycle: '+path)
        seen.add(path);path=edges[path]
    return path

def inventory(exclusions=()):
    d=read(RUN/'data_sources.json');q=d['replacement_queue'];blockers=[];edges={};pins={}
    audits=[Path(q['manifest']).parent/'completion_verified.json']
    audits += [Path(s['logs'])/'completion_verified.json' for s in q['stages']]
    audits += [Path(e['completion_audit']) for e in q.get('extensions',[])]
    audits += [Path(source['completion_audit']) for source in d['additions'] if source.get('completion_audit')]
    maps=[Path(s['logs'])/'replacement_map.json' for s in q['stages'] if s['name']!='base160']
    maps += [Path(e['replacement_map']) for e in q.get('extensions',[])]
    for p in audits:
        if not p.exists() or not read(p).get('verified'):blockers.append('录制审计未完成: '+str(p))
        else:pins[str(p)]=sha(p)
    for p in maps:
        if not p.exists():blockers.append('重录替换表尚未生成: '+str(p));continue
        pins[str(p)]=sha(p)
        for r in read(p):
            a,b=str(Path(r['old']).resolve()),str(Path(r['new']).resolve())
            if a in edges and edges[a]!=b:raise ValueError('Conflicting replacement: '+a)
            edges[a]=b
    waiver=RUN/'quality_waivers.json'
    if waiver.exists():pins[str(waiver)]=sha(waiver)
    excluded={str(Path(p).resolve()) for p in exclusions}
    roots=d['base_sources']+d['additions']+[dict(root=s['data'],episodes=s['target']) for s in q['stages'] if s['name']=='base160']
    pending=set(d['quality_control']['reverse_excursion_ge_8cm'].get('saved_replacement_episode_marks',[]))
    candidates=[]
    for source in roots:
        root=Path(source['root']);episodes=sorted(root.glob('episode_*/frames.jsonl'))
        expected=source.get('episodes',source.get('saved_episodes_at_registration'))
        if expected is not None and len(episodes)!=expected:blockers.append(f'来源数量不符: {root}: {len(episodes)}/{expected}')
        candidates.extend(str(p.parent.resolve()) for p in episodes)
        pending.update(source.get('pending_rerecord_episodes',[]));pending.update(source.get('exclude_episodes',[]))
    for p in pending:
        final=resolve(p,edges)
        if final==str(Path(p).resolve()) and final not in excluded:
            frames=Path(final)/'frames.jsonl'
            if not frames.exists() or not accepted_reverse(RUN,final,sha(frames)):
                blockers.append('待重录/待明确排除: '+str(p))
    selected=[];seen=set();digests=set()
    for p in candidates:
        final=resolve(p,edges)
        if final in excluded or str(Path(p).resolve()) in excluded:continue
        if final in seen:continue
        seen.add(final);frames=Path(final)/'frames.jsonl'
        if not frames.exists():blockers.append('缺少替换数据: '+final);continue
        h=sha(frames)
        if h in digests:continue
        digests.add(h);selected.append(dict(path=final,frames_sha256=h))
    pins[str(RUN/'data_sources.json')]=sha(RUN/'data_sources.json')
    return dict(ready=not blockers,episodes=len(selected),blockers=blockers,sources=selected,
                input_pins=pins,explicit_exclusions=sorted(excluded),replacement_edges=edges)

def verify_frozen():
    frozen=read(RUN/'sources_frozen.json')
    for p,h in frozen['input_pins'].items():
        if sha(p)!=h:raise ValueError('冻结后输入发生变化，请重新整理转换: '+p)
    for r in frozen['sources']:
        if sha(Path(r['path'])/'frames.jsonl')!=r['frames_sha256']:raise ValueError('Frozen frames changed: '+r['path'])
    return frozen

def require_ready_for_training():
    frozen=verify_frozen()
    state=inventory(frozen['explicit_exclusions'])
    if not state['ready']:raise RuntimeError('\n'.join(state['blockers']))
    if state['sources']!=frozen['sources']:raise RuntimeError('数据来源已变化，请重新整理转换')
    if subprocess.run(['pgrep','-f','[/]run_z1_ros2_scene.py'],stdout=subprocess.DEVNULL).returncode==0:
        raise RuntimeError('仿真录制仍在运行；请完成录制后再训练')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['status','prepare','convert','train','smoke'])
    p.add_argument('--exclude-list',type=Path,help='JSON list of episode paths explicitly excluded by you')
    a=p.parse_args()
    with (RUN/'.interface.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if a.stage in ('status','prepare'):
            exclusions=read(a.exclude_list) if a.exclude_list else []
            if not isinstance(exclusions,list) or not all(isinstance(x,str) for x in exclusions):raise ValueError('Exclusion file must be a JSON string list')
            state=inventory(exclusions);write('readiness.json',state)
            print(json.dumps({k:v for k,v in state.items() if k not in ('sources','input_pins','replacement_edges')},ensure_ascii=False,indent=2))
            if a.stage=='prepare':
                if not state['ready']:raise RuntimeError('尚未具备最终冻结条件，未生成训练数据')
                write('sources_frozen.json',state)
        elif a.stage=='convert':
            verify_frozen()
            from dataset_prepare import main as convert
            convert()
            subprocess.run([sys.executable,str(RUN/'verify.py')],check=True)
        else:
            require_ready_for_training()
            if a.stage=='smoke':
                subprocess.run([sys.executable,str(RUN/'train.py'),'--smoke'],check=True)
                subprocess.run([sys.executable,str(RUN/'verify.py'),'--reload-smoke'],check=True)
            else:
                # GPU smoke is executed only by the user's explicit training command.
                if not (RUN/'checkpoint_reload_verified.json').exists():
                    subprocess.run([sys.executable,str(RUN/'train.py'),'--smoke'],check=True)
                    subprocess.run([sys.executable,str(RUN/'verify.py'),'--reload-smoke'],check=True)
                subprocess.run([sys.executable,str(RUN/'train.py'),'--train'],check=True)

if __name__=='__main__':
    try:main()
    except (ValueError,RuntimeError,FileNotFoundError,BlockingIOError) as e:
        print('未启动/未完成：'+str(e),file=sys.stderr);sys.exit(2)
