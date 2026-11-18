"""Generate audited sampling restrictions without rewriting data or starting training."""
from pathlib import Path
import csv, hashlib, json, sys
import numpy as np
import pyarrow.parquet as pq
from windows import build_window, anchor_mask
W=Path(__file__).resolve().parent;WORK=W.parent
RAW=Path('/home/lp531/smolvla_data/z1_bottle_correction_fragments_20261012')
BASE=Path('/home/lp531/smolvla_data/z1_bottle_gentle_correction_20261010')
sys.path.insert(0,str(WORK/'binary_gripper_20261007'))
from convert import label_episode
sys.path.insert(0,'/home/lp531/ros2_ws/src/z1_ros2/scripts/correction_fragments_20261012')
from fragment_core import partial_labels,valid_hold

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def load(p):return json.loads(p.read_text())

def main():
    old=load(WORK/'gentle_correction_20261010/snapshot.json')
    original=load(WORK/'gentle_correction_20261010/phase_manifest.json')[:160]
    new=[]
    for p in sorted(RAW.glob('episode_[0-9]*')):
        m=load(p/'correction.json');s=load(p/'episode_summary.json')
        assert m['accepted'] and m['recovery_complete'] and not m['table_contact'] and valid_hold(m['final_status'])
        assert s['fragment_success'] and s['dropped_frames']==0
        new.append(dict(source=str(p),task_id=m['task_id'],split=m['split'],kind=m['task']['kind']))
    assert len(old)==32 and len(new)==19
    records=[];hashes={};labels_by_id={}
    for group,entries in [('old_correction',old),('new_fragment',new)]:
        for i,e in enumerate(entries):
            root=Path(e['source']);p=root/'frames.jsonl';hashes[str(p)]=digest(p)
            rows=[json.loads(x) for x in p.read_text().splitlines()];m=load(root/'correction.json')
            if group=='old_correction':
                labels,event=label_episode(rows);close=event['close_frame'];opening=event['open_frame'];index=e['episode_index']
                assert digest(p)==e['frames_sha256']
            else:
                labels,close,_=partial_labels(rows);opening=None;index=192+i
            r=build_window(rows,m,close,opening)
            r.update(episode_index=index,episode_id_scope='existing_192' if index<192 else 'planned_append_order',
                     source=group,source_path=str(root),source_frames_sha256=digest(p),
                     task_id=e['task_id'],kind=e['kind'],split=e['split'],
                     start_time_s=rows[r['frame_start']]['timestamp'],end_time_s=rows[r['frame_end_inclusive']]['timestamp'])
            for start in range(r['anchor_start'],r['anchor_end_inclusive']+1):
                chunk=labels[start:start+15];assert len(chunk)==15
                # After the first close, no label in any allowed chunk may reopen.
                for j,label in enumerate(chunk,start):
                    if j>=close:assert label==1
            records.append(r);labels_by_id[index]=labels
    all_records=[]
    for r in original:
        all_records.append(dict(episode_index=r['episode_index'],source='original',split=r['split'],
                                frame_count=r['frames'],anchor_start=0,anchor_end_inclusive=r['frames']-1))
    all_records+=records
    table=pq.read_table(BASE/'data',columns=['index','episode_index','frame_index','action']).sort_by([('index','ascending')])
    ep=table['episode_index'].to_numpy();fi=table['frame_index'].to_numpy()
    assert np.array_equal(table['index'].to_numpy(),np.arange(len(table)))
    for r in all_records[:192]:
        assert np.array_equal(fi[ep==r['episode_index']],np.arange(r['frame_count']))
        if r['episode_index']>=160:
            actions=np.asarray(table.filter(__import__('pyarrow').compute.equal(table['episode_index'],r['episode_index']))['action'].to_pylist())
            np.testing.assert_array_equal(actions[:,-1],labels_by_id[r['episode_index']])
    current=anchor_mask(ep,fi,all_records[:192]);phase=np.load(WORK/'grip_stage_base_20261011/sampling_arrays.npz')
    np.testing.assert_array_equal(ep,phase['episode'])
    np.savez(W/'existing_192_sampling_arrays.npz',episode=ep,frame=fi,eligible_train_anchor=current,
             correction=phase['correction'],phase_weight=phase['phase_weight'])
    ndir=W/'new_fragment_masks';ndir.mkdir(exist_ok=True)
    for r in records[32:]:
        n=r['frame_count'];e=np.full(n,r['episode_index']);f=np.arange(n)
        np.savez(ndir/(r['task_id']+'.npz'),frame=f,
                 eligible_train_anchor=anchor_mask(e,f,[r]),
                 eligible_evaluation_anchor=anchor_mask(e,f,[r],training=False))
    assert all(digest(Path(p))==h for p,h in hashes.items())
    held=[r['episode_index'] for r in all_records if r['split']=='validation']
    assert len(held)==8 and not np.isin(ep[current],held).any()
    summary={}
    for group in ['old_correction','new_fragment']:
        rs=[r for r in records if r['source']==group]
        summary[group]=dict(episodes=len(rs),train_episodes=sum(r['split']=='train' for r in rs),
                           validation_episodes=sum(r['split']=='validation' for r in rs),
                           recorded_frames=sum(r['frame_count'] for r in rs),kept_frames=sum(r['kept_frames'] for r in rs),
                           eligible_train_anchors=sum(r['eligible_anchor_count'] for r in rs if r['split']=='train'),
                           hold_truncated_tasks=[r['task_id'] for r in rs if r['hold_truncated']])
    save(W/'windows.json',dict(fps=10,chunk_size=15,interval_convention='inclusive',
         rule='Start 0.5s before first recovery stage; no-stage close_ready starts 1s before close. End 1.5s after 5-frame physical grasp/lift confirmation, capped before release and recording end.',
         old_dataset=str(BASE),new_raw_root=str(RAW),records=all_records))
    with (W/'windows.csv').open('w') as out:
        keys=['source','task_id','split','frame_count','frame_start','frame_end_inclusive','anchor_start','anchor_end_inclusive','close_frame','stable_confirmed_frame','open_frame','start_time_s','end_time_s','hold_truncated']
        writer=csv.DictWriter(out,fieldnames=keys,extrasaction='ignore');writer.writeheader();writer.writerows(records)
    audit=dict(passed=True,chunk_size=15,all_correction_chunks_complete=True,no_reopening_label_in_correction_chunks=True,
               original_160_sampling_intervals_unchanged=True,validation_leak=False,source_frame_hashes_unchanged=True,
               existing_dataset_episodes=192,existing_dataset_eligible_train_anchors=int(current.sum()),
               new_fragments_converted_or_merged=False,new_episode_ids_are_planned_only=True,
               combined_future_episodes=211,combined_future_train_episodes=203,combined_future_validation_episodes=8,
               training_started=False,groups=summary)
    save(W/'audit.json',audit);print(json.dumps(audit,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
