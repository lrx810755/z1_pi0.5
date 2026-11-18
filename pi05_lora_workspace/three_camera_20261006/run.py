"""Convert all successful three-camera episodes and launch an isolated PI05 LoRA run."""
import argparse
import collections
import json
import math
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import time

RUN=Path(__file__).resolve().parent
WORK=RUN.parent
RAW=Path('/home/lp531/smolvla_data/z1_tabletop_raw_left_wrist_three_camera_v1')
DATA=Path('/home/lp531/smolvla_data/z1_bottle_three_camera_pi05_20261006')
OUTPUT=Path('/home/lp531/smolvla_training/pi05_bottle_three_camera_20261006')
REPO='lp531/z1_bottle_three_camera_pi05_20261006'
CAMERAS=('observation.images.wrist','observation.images.global','observation.images.global_right')
ASSETS=Path('/home/lp531/ros2_ws/src/z1_ros2/isaac_assets/z1/scripts')


def write(name,obj):
    path=RUN/name
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=str)+'\n')
    tmp.replace(path)


def status(phase,**extra):
    write('status.json',dict(phase=phase,time=time.time(),**extra))
    print(phase,extra,flush=True)


def split_entries(entries):
    if len(entries)<2:raise ValueError('Need at least two complete episodes')
    seeds=[e['scene_randomization']['seed'] for e in entries]
    if len(set(seeds))!=len(seeds):raise ValueError('Repeated scene seed; refusing train/validation leakage')
    rng=random.Random(20261006)
    groups=collections.defaultdict(list)
    for e in entries:groups[tuple(e['scene_randomization'].get('target_grid',()))].append(e)
    train,val=[],[]
    for grid in sorted(groups):
        group=groups[grid][:];rng.shuffle(group)
        n=min(len(group)-1,max(1,round(len(group)*.1))) if len(group)>1 else 0
        val.extend(group[:n]);train.extend(group[n:])
    if not val:val.append(train.pop())
    rng.shuffle(train);rng.shuffle(val)
    return train,val


def plan():
    entries=[]
    for p in sorted(RAW.glob('episode_[0-9][0-9][0-9][0-9][0-9][0-9]')):
        summary=json.loads((p/'episode_summary.json').read_text())
        if summary.get('target_object')!='beverage_bottle' or not summary.get('task_success'):continue
        rows=[json.loads(l) for l in (p/'frames.jsonl').read_text().splitlines() if l.strip()]
        if len(rows)!=summary['frame_count'] or not rows:raise ValueError(f'Incomplete saved episode {p}')
        for i,r in enumerate(rows):
            if r['frame_index']!=i or not all(k in r for k in CAMERAS):raise ValueError(f'Frame metadata invalid: {p}:{i}')
            if not all(math.isfinite(v) for k in ('observation.state','action') for v in r[k]):raise ValueError(f'Nonfinite labels: {p}:{i}')
        entries.append(dict(summary,source=str(p)))
    train,val=split_entries(entries)
    manifest=dict(raw_root=str(RAW),dataset_root=str(DATA),train_episodes=len(train),validation_episodes=len(val),
                  total_episodes=len(entries),total_frames=sum(e['frame_count'] for e in entries),
                  seed=20261006,middle_frames_removed=0,cameras=list(CAMERAS),episodes=[dict(e,split='train' if i<len(train) else 'validation',episode_index=i) for i,e in enumerate(train+val)])
    write('split_manifest.json',manifest)
    return manifest


def prepare():
    import numpy as np
    import pyarrow.parquet as pq
    sys.path.insert(0,str(ASSETS))
    import export_z1_lerobot_dataset as export
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    if DATA.exists():raise FileExistsError(f'Preserve existing output; inspect before reuse: {DATA}')
    manifest=plan();info=json.loads((RAW/'dataset_info.json').read_text())
    shapes=export.inspect_first_frame(Path(manifest['episodes'][0]['source']),CAMERAS)
    ds=LeRobotDataset.create(repo_id=REPO,fps=10,root=DATA,robot_type='unitree_z1',
        features=export.features(info['joint_names'],shapes,CAMERAS,True),use_videos=True,image_writer_threads=4)
    train_arrays={'observation.state':[],'action':[]}
    expected={k:[] for k in train_arrays}
    for e in manifest['episodes']:
        p=Path(e['source']);rows=list(export.read_frames(p/'frames.jsonl'))
        arrays={k:np.asarray([r[k] for r in rows],dtype=np.float32) for k in expected}
        for k,a in arrays.items():
            expected[k].append(a)
            if e['split']=='train':train_arrays[k].append(a)
        for r in rows:
            frame={k:np.asarray(r[k],dtype=np.float32) for k in expected}
            frame.update({k:export.load_rgb(p/r[k]) for k in CAMERAS})
            frame['task']=e['task']  # LeRobot computes timestamp from frame index and fps.
            ds.add_frame(frame)
        ds.save_episode()
        print('CONVERTED',e['episode_index'],p.name,len(rows),flush=True)
    ds.finalize()
    table=pq.read_table(DATA/'data').sort_by([('index','ascending')])
    assert len(table)==manifest['total_frames']
    for k,parts in expected.items():
        actual=np.asarray(table[k].to_pylist(),dtype=np.float32)
        assert np.array_equal(actual,np.concatenate(parts)),f'Label mismatch: {k}'
    meta=json.loads((DATA/'meta/info.json').read_text())
    assert meta['total_episodes']==manifest['total_episodes']
    assert {k for k,v in meta['features'].items() if v['dtype']=='video'}==set(CAMERAS)
    packet_counts={}
    for camera in CAMERAS:
        total=0
        for video in (DATA/'videos'/camera).rglob('*.mp4'):
            result=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_packets','-select_streams','v:0','-show_entries','stream=nb_read_packets','-of','json',str(video)]))
            total+=int(result['streams'][0]['nb_read_packets'])
        assert total==manifest['total_frames'],(camera,total)
        packet_counts[camera]=total
    stats=json.loads((DATA/'meta/stats.json').read_text())
    for k,parts in train_arrays.items():
        a=np.concatenate(parts)
        stats[k]={name:func(a,axis=0).tolist() for name,func in [('min',np.min),('max',np.max),('mean',np.mean),('std',np.std)]}
        stats[k].update(q01=np.quantile(a,.01,axis=0).tolist(),q99=np.quantile(a,.99,axis=0).tolist(),count=[len(a)])
    (DATA/'meta/stats.json').write_text(json.dumps(stats,indent=2)+'\n')
    write('setup_verification.json',dict(episodes=manifest['total_episodes'],frames=manifest['total_frames'],all_state_action_frames_exact=True,normalization_train_only=True,video_packets=packet_counts,middle_frames_removed=0))


def config(smoke=False):
    sys.path.insert(0,str(WORK))
    from train_lora import make_config
    manifest=json.loads((RUN/'split_manifest.json').read_text())
    cfg=make_config(microbatch=1,smoke=smoke)
    cfg.dataset.root=str(DATA);cfg.dataset.repo_id=REPO
    cfg.dataset.eval_split=manifest['validation_episodes']/manifest['total_episodes']-1e-12
    cfg.output_dir=RUN/'smoke_batch1' if smoke else OUTPUT
    cfg.job_name='pi05_bottle_three_camera';cfg.seed=20261006
    cfg.policy.scheduler_warmup_steps=500*8
    cfg.policy.scheduler_decay_steps=10000*8
    if not smoke:
        cfg.steps=10000*8;cfg.save_freq=1000*8;cfg.eval_steps=1000*8
    cfg.validate()
    write('smoke_config.json' if smoke else 'resolved_train_config.json',cfg.to_dict())
    return cfg


def train(smoke=False):
    import io
    from contextlib import redirect_stdout
    from lerobot.scripts import lerobot_train as trainer
    cfg=config(smoke)
    sys.argv=[sys.argv[0]]
    original=trainer.make_policy
    def checked_policy(**kwargs):
        stream=io.StringIO()
        with redirect_stdout(stream):policy=original(**kwargs)
        print(stream.getvalue(),flush=True)
        if 'All keys loaded successfully!' not in stream.getvalue():raise RuntimeError('Official base loading was not verified')
        if set(policy.config.image_features)!=set(CAMERAS):raise RuntimeError(f'Camera features mismatch: {policy.config.image_features}')
        write('smoke_initialization.json' if smoke else 'training_initialization.json',dict(base=str(WORK/'weights/pi05_base'),cameras=list(policy.config.image_features),base_loaded=True,trainable_parameters=sum(p.numel() for p in policy.parameters() if p.requires_grad)))
        return policy
    trainer.make_policy=checked_policy
    trainer.train(cfg)
    if not smoke:
        matches=re.findall(r'step (\d+): eval_loss=([\d.eE+-]+)',(RUN/'training.log').read_text())
        candidates=sorted((float(loss),int(step)) for step,loss in matches if math.isfinite(float(loss)))
        for loss,step in candidates:
            model=OUTPUT/'checkpoints'/f'{step:06d}'/'pretrained_model'
            if model.exists():
                (OUTPUT/'best').symlink_to(model,target_is_directory=True)
                write('complete.json',dict(optimizer_updates=10000,best_checkpoint=str(model),validation_loss=loss,physical_success_evaluated=False));break
        else:raise RuntimeError('No evaluated checkpoint available')


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',action='store_true');p.add_argument('--train',action='store_true');p.add_argument('--smoke',action='store_true');args=p.parse_args()
    if args.plan:print(json.dumps(plan(),ensure_ascii=False,indent=2));return
    if args.train:train(args.smoke);return
    try:
        status('converting');prepare()
        for smoke,log in [(True,'smoke.log'),(False,'training.log')]:
            status('preflight' if smoke else 'training')
            cmd=[sys.executable,str(Path(__file__).resolve()),'--train']+(['--smoke'] if smoke else [])
            with (RUN/log).open('w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True,cwd=RUN)
        status('complete')
    except Exception as e:
        status('failed',error=str(e));raise

if __name__=='__main__':main()
