"""Verify raw-to-training mapping and build external weights; never edit datasets."""
from pathlib import Path
import collections,hashlib,json
import numpy as np
import pyarrow.dataset as ds
from sampling import classify_frames
P=Path(__file__).resolve().parent
WORK=P.parent
DATA=Path('/home/lp531/smolvla_data/z1_tabletop_bottle_pi05_lora_v1')
RAW=Path('/home/lp531/smolvla_data/z1_tabletop_bottle_recovery_boundary_only_prepared_v1')
SOURCE=Path('/home/lp531/smolvla_training/pi05_bottle_lora_20261001/checkpoints/108000/pretrained_model')
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def prepare():
    split=json.loads((WORK/'split_manifest.json').read_text())['episodes']
    table=ds.dataset(DATA/'data',format='parquet').to_table(columns=['index','episode_index','frame_index','observation.state','action']).sort_by('index')
    ix=np.asarray(table['index']);ep=np.asarray(table['episode_index']);fi=np.asarray(table['frame_index'])
    states=np.array(table['observation.state'].to_pylist(),dtype=np.float32);actions=np.array(table['action'].to_pylist(),dtype=np.float32)
    assert len(table)==65773 and np.array_equal(ix,np.arange(len(table)))
    weights=np.ones(len(table));labels=np.full(len(table),'normal',dtype='<U16');summary=[]
    for entry in split:
        e=entry['episode_index'];loc=np.flatnonzero(ep==e)
        assert len(loc)==entry['frames'] and np.array_equal(fi[loc],np.arange(len(loc)))
        rows=[json.loads(s) for s in (RAW/f"episode_{entry['original_episode_index']:06d}"/'frames.jsonl').read_text().splitlines()]
        assert np.array_equal(states[loc],np.array([r['observation.state'] for r in rows],dtype=np.float32)),e
        assert np.array_equal(actions[loc],np.array([r['action'] for r in rows],dtype=np.float32)),e
        weights[loc],labels[loc]=classify_frames(states[loc],actions[loc])
        summary.append({'episode':e,'split':entry['split'],'counts':dict(collections.Counter(labels[loc]))})
    train=ep<221
    assert sum(x['split']=='train' for x in split)==221
    np.save(P/'weights.npy',weights)
    details={}
    for name,w in [('uniform',np.ones(len(weights))),('weighted',weights)]:
        details[name]={label:{'frames':int(((labels==label)&train).sum()),'probability':float(w[(labels==label)&train].sum()/w[train].sum())} for label in sorted(set(labels))}
    manifest={'source_checkpoint':str(SOURCE),'source_adapter_sha256':sha(SOURCE/'adapter_model.safetensors'),'source_config_sha256':sha(SOURCE/'config.json'),'weights_sha256':sha(P/'weights.npy'),'dataset_meta_sha256':sha(DATA/'meta/info.json'),'split_sha256':sha(WORK/'split_manifest.json'),'total_frames':len(table),'train_frames':int(train.sum()),'validation_frames':int((~train).sum()),'frame_and_action_equality_verified':True,'window':'unchanged native contiguous 50-step window; native episode-end padding','stage_weights':{'normal':1,'open_wait':.25,'wrist_approach':2,'close_transition':3},'probabilities':details,'episodes':summary}
    (P/'manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps({k:v for k,v in manifest.items() if k!='episodes'},indent=2))
if __name__=='__main__':prepare()
