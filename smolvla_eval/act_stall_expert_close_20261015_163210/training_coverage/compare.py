"""Read-only training coverage and action-target audit of six tested stall poses."""
import json,sys,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation
import cv2

OUT=Path(__file__).resolve().parent
RUN=Path('/home/lp531/pi05_lora_workspace/act_gentle_cached_20261015')
DATA=Path('/home/lp531/smolvla_data/z1_bottle_with_fragments_20261012')
EVAL=Path('/home/lp531/smolvla_eval/act_gentle_15vs8_repaired_20261015_133937')
CACHE=Path('/home/lp531/smolvla_data/act_rgb224_float32_cache_20261015')
CP=Path('/home/lp531/smolvla_training/act_gentle_cached_chunk16_20261015/checkpoints/050000/pretrained_model')
split=json.loads((RUN/'split_manifest.json').read_text());records=split['source_records']
for r in records:r['split']='train' if r['episode_index'] in split['train_episodes'] else 'validation'
raw=[]
for p in DATA.glob('data/*/*.parquet'):raw.extend(pq.read_table(p).to_pylist())
raw.sort(key=lambda r:r['index'])
states=np.array([r['observation.state'] for r in raw]);actions=np.array([r['action'] for r in raw])
ep=np.array([r['episode_index'] for r in raw]);frame=np.array([r['frame_index'] for r in raw])
arr=np.load(RUN/'sampling_arrays.npz');assert np.array_equal(ep,arr['episode']) and np.array_equal(frame,arr['frame'])
train=arr['eligible_train'];weight=arr['weights'];key=arr['key'];offset=frame-np.array([records[e]['close_frame'] for e in ep])
preopen=(offset<=0)&(states[:,-1]<=-.88)
indices=np.flatnonzero(train & preopen)
byep={e:np.flatnonzero(ep==e) for e in range(211)}

urdf=ET.parse('/home/lp531/ros2_ws/src/z1_ros2/isaac_assets/z1/generated_dual_camera/z1_with_dual_d435_resolved.urdf').getroot()
joints={j.attrib['name']:j for j in urdf.findall('joint')}
chain=[joints[f'joint{i}'] for i in range(1,7)]+[joints['gripperStatorLink']]
def fk(q):
    q=np.atleast_2d(q);n=len(q);R=np.broadcast_to(np.eye(3),(n,3,3)).copy();p=np.zeros((n,3))
    for i,j in enumerate(chain):
        o=j.find('origin');xyz=np.fromstring(o.attrib.get('xyz','0 0 0'),sep=' ')
        p+=np.einsum('nij,j->ni',R,xyz)
        R=R@Rotation.from_euler('xyz',np.fromstring(o.attrib.get('rpy','0 0 0'),sep=' ')).as_matrix()
        if i<6:R=R@Rotation.from_rotvec(q[:,i,None]*np.fromstring(j.find('axis').attrib['xyz'],sep=' ')).as_matrix()
    return p,R

source=next((EVAL/'steps15').glob('*/episodes/beverage_bottle_016_*/trace.jsonl'))
sample=json.loads(source.read_text().splitlines()[0]);pf,Rf=fk(sample['joint_state'][:6]);cs=sample['contact_status']
Rw=Rotation.from_quat(cs['gripper_world_orientation_xyzw']).as_matrix()@Rf[0].T
tw=np.array(cs['gripper_world_position_m'])-Rw@pf[0]
positions,rotations=fk(states[:,:6]);positions=positions@Rw.T+tw;rotations=Rw@rotations
starts=[]
for r in records:
    s=json.loads((Path(r['source_path'])/'episode_start.json').read_text())
    starts.append(np.array(s['scene_randomization']['object_positions_m']['beverage_bottle'])+[0,0,.0785])
centers=np.array(starts)[ep]
relative=np.einsum('nji,nj->ni',rotations,centers-positions)
up=rotations[:,2,:]
targets=[]
verified=json.loads((OUT.parent/'results.json').read_text())
analysis=json.loads((EVAL/'review/analysis.json').read_text())
for episode in [16,21,26]:
    folder=next((EVAL/'steps15').glob(f'*/episodes/beverage_bottle_{episode:03d}_*'))
    rows=[json.loads(x) for x in (folder/'trace.jsonl').read_text().splitlines()]
    # Reconstruct video frame order: one before each chunk and one at every
    # action's final interpolation substep. Verified against encoded frame count.
    video_index=[];count=-1
    for row in rows:
        if row['policy_action_index']==0 and row['execution_substep']==0:count+=1
        if row['execution_substep']==5 and not row.get('policy_action_paused'):count+=1
        video_index.append(count)
    cap=cv2.VideoCapture(str(folder/'videos/wrist.mp4'));assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT))==count+1;cap.release()
    ref=next(r for r in analysis['episodes'] if r['steps']==15 and r['episode']==episode)
    for which in ['final','closest']:
        ix=len(rows)-1 if which=='final' else min(range(len(rows)),key=lambda i:abs(rows[i]['simulation_elapsed_s']-ref['closest_before_close']['time_s']))
        row=rows[ix];c=row['contact_status'];R=Rotation.from_quat(c['gripper_world_orientation_xyzw']).as_matrix()
        posdist=np.linalg.norm(relative-np.array(c['relative_position_gripper_m']),axis=1)
        angles=np.degrees(np.arccos(np.clip(up@R[2,:],-1,1)))
        score=np.sqrt((posdist/.01)**2+(angles/5)**2)
        st=np.sqrt(np.mean((states[:,:6]-np.array(row['joint_state'][:6]))**2,axis=1))
        t=dict(name=f'ep{episode}_{which}',episode=episode,snapshot=which,source_elapsed_s=row['simulation_elapsed_s'],state=row['joint_state'],relative_position_m=c['relative_position_gripper_m'],physically_graspable=next(v['physically_graspable'] for v in verified if v['episode']==episode and v['snapshot']==which),video_frame=video_index[ix],matches={})
        for camera in ['wrist','global','global_right']:
            filename='global_third_person' if camera=='global' else camera
            cap=cv2.VideoCapture(str(folder/'videos'/f'{filename}.mp4'));cap.set(cv2.CAP_PROP_POS_FRAMES,video_index[ix]);ok,image=cap.read();cap.release();assert ok
            cv2.imwrite(str(OUT/f"{t['name']}_{camera}.png"),image)
        for pool in ['original','old_correction','new_fragment']:
            mask=train & preopen & np.array([records[e]['source']==pool for e in ep])
            # New fragments are named new_fragment in the split manifest.
            if not mask.any():continue
            ids=np.flatnonzero(mask);near=ids[np.argsort(score[ids])[:40]]
            combined=ids[np.argsort(score[ids]**2+(st[ids]/.1)**2)[:40]]
            candidates=list(dict.fromkeys(near.tolist()+combined.tolist()))
            similar=mask&(posdist<=.02)&(angles<=10)&(st<=.15)
            t['matches'][pool]=dict(candidate_indices=candidates,combined_top_indices=combined[:3].tolist(),near_10mm_5deg=int((mask&(posdist<=.01)&(angles<=5)).sum()),near_20mm_10deg=int((mask&(posdist<=.02)&(angles<=10)).sum()),similar_with_joint_015rad=dict(frames=int(similar.sum()),episodes=sorted(set(ep[similar].tolist())),future_close_in_16=int((similar&(offset>=-15)).sum()),sampling_mass=float(weight[similar].sum())),nearest_position_mm=float(posdist[near[0]]*1000),nearest_angle_deg=float(angles[near[0]]),nearest_joint_rms_rad=float(st[near[0]]))
        t['_posdist']=posdist;t['_angle']=angles;t['_score']=score;t['_joint']=st;targets.append(t)

def info(idx,t):
    e=int(ep[idx]);f=int(frame[idx]);rec=records[e];truth=actions[idx:idx+16,-1]
    assert len(truth)==16 and ep[idx+15]==e
    close=np.flatnonzero(truth>=.6)
    return dict(index=int(idx),episode=e,source=rec['source'],split=rec['split'],frame=f,offset_from_close=int(offset[idx]),input_gripper=float(states[idx,-1]),label=float(actions[idx,-1]),first_closed_step=int(close[0]+1) if len(close) else None,truth=truth.tolist(),key=bool(key[idx]),sampling_weight=float(weight[idx]),expected_draws_800k=float(weight[idx]*800000),position_distance_mm=float(t['_posdist'][idx]*1000),orientation_distance_deg=float(t['_angle'][idx]),joint_rms_rad=float(t['_joint'][idx]),source_path=rec['source_path'],relative_position_m=relative[idx].tolist())

groups={}
for name,mask in [('before_close_2s',train&(offset>=-20)&(offset<0)),('open_input_at_close_label',train&(offset==0)&(states[:,-1]<=-.88)),('open_input_future_close_16',train&preopen&(offset>=-15)),('open_input_all',train&preopen)]:
    groups[name]=dict(anchors=int(mask.sum()),episodes=int(len(set(ep[mask]))),probability_mass=float(weight[mask].sum()),expected_draws_800k=float(weight[mask].sum()*800000),key_anchors=int(key[mask].sum()))
print('sampling',json.dumps(groups),flush=True)
print('sources',sorted(set(r['source'] for r in records)),flush=True)
for t in targets:
    for pool,d in t['matches'].items():d['geometry_top']=[info(i,t) for i in d['candidate_indices'][:5]]
    print(t['name'],json.dumps({p:{k:v for k,v in m.items() if k not in ['candidate_indices','geometry_top']} for p,m in t['matches'].items()}),flush=True)

# Image comparison and inference use the exact current checkpoint. Cached images
# already contain crop/resize, so that one preprocessing step is bypassed.
import torch
import torchvision.transforms.functional as TF
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.processor import ImageCropResizeProcessorStep
torch.set_num_threads(4)
policy=ACTPolicy.from_pretrained(CP,local_files_only=True,strict=True).eval()
pre,post=make_pre_post_processors(policy.config,pretrained_path=CP)
pre.steps=[s for s in pre.steps if not isinstance(s,ImageCropResizeProcessorStep)]
cams=['wrist','global','global_right'];maps={}
def image_at(idx,cam):
    e=int(ep[idx]);pair=(e,cam)
    if pair not in maps:maps[pair]=np.load(CACHE/f'observation.images.{cam}'/f'episode_{e:06d}.npy',mmap_mode='r')
    return np.array(maps[pair][int(frame[idx])],copy=True)
def images_target(t):
    result=[]
    for cam in cams:
        im=cv2.cvtColor(cv2.imread(str(OUT/f"{t['name']}_{cam}.png")),cv2.COLOR_BGR2RGB)
        im=torch.from_numpy(im.copy()).permute(2,0,1).float()/255
        im=TF.resize(TF.crop(im,-140 if cam=='wrist' else -80,0,640,640),[224,224],antialias=True)
        result.append(im.numpy())
    return result
def predict(states_batch,images_batch):
    batch={'observation.state':torch.tensor(np.array(states_batch),dtype=torch.float32)}
    for i,cam in enumerate(cams):batch[f'observation.images.{cam}']=torch.tensor(np.array([x[i] for x in images_batch]),dtype=torch.float32)
    with torch.inference_mode():
        prepared=pre(batch);pred=post(policy.predict_action_chunk(prepared)).cpu().numpy()
        features=[]
        for cam in cams:
            v=policy.model.backbone(prepared[f'observation.images.{cam}'])['feature_map']
            v=torch.nn.functional.adaptive_avg_pool2d(v,(3,3)).flatten(1)
            features.append(torch.nn.functional.normalize(v,dim=1).cpu().numpy())
    return pred,np.stack(features,axis=1)

allids=sorted(set(i for t in targets for d in t['matches'].values() for i in d['candidate_indices']))
features={};predictions={}
for start in range(0,len(allids),12):
    ids=allids[start:start+12];images=[[image_at(i,c) for c in cams] for i in ids]
    pred,feat=predict(states[ids],images)
    for i,p,f in zip(ids,pred,feat):features[i]=f;predictions[i]=p[:,-1].tolist()
    if start%120==0:print('inference',start,len(allids),flush=True)
for t in targets:
    pred,feat=predict([t['state']],[images_target(t)])
    t['video_observation_prediction']=pred[0,:,-1].tolist()
    for pool,d in t['matches'].items():
        ids=d['candidate_indices'];sims={i:features[i]@feat[0].T for i in ids}
        # Matching view only; image similarity is a ranking, not proof of input
        # equivalence or a calibrated coverage probability.
        ranked=sorted(ids,key=lambda i:float(sims[i][0,0]),reverse=True)
        chosen=list(dict.fromkeys(ids[:3]+ranked[:3]+d['combined_top_indices']))
        d['examples']=[]
        for idx in chosen:
            item=info(idx,t);item['wrist_feature_cosine']=float(sims[idx][0,0]);item['global_feature_cosine']=float(sims[idx][1,1]);item['right_feature_cosine']=float(sims[idx][2,2]);item['prediction']=predictions[idx]
            d['examples'].append(item)
        d['visual_best_among_geometry_candidates']=ranked[0]
        del d['candidate_indices']
    for k in list(t):
        if k.startswith('_'):del t[k]
report=dict(checkpoint=str(CP),training_episodes=190,validation_excluded=split['validation_episodes'],eligible_training_anchors=int(train.sum()),open_preclose_training_anchors=len(indices),sampling=groups,targets=targets,
    limitations=['Training relative geometry assumes initial bottle position remains fixed/upright before first close; per-frame object pose is absent.',
                'Evaluation images are encoded rollout video frames, not exact inference-input RGB; state/image timing and codec differ.',
                'Image search reranks a union of 40 nearest geometric and 40 geometry-plus-joint frames per source, not all training images.',
                'Near-neighbor geometry and visual resemblance do not prove full input equivalence.',
                'Predictions are for current gentle ACT; sampled expected counts are expectations, not recorded sampler visitation counts.'])
(OUT/'comparison.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
np.savez_compressed(OUT/'geometry_arrays.npz',relative=relative,up=up,states=states,actions=actions,episode=ep,frame=frame,weight=weight,eligible=train,preopen=preopen)
print('complete',OUT/'comparison.json',flush=True)
