"""Matched warm-start LoRA runs: the only A/B treatment is anchor sampling weight."""
from pathlib import Path
import argparse,hashlib,json,sys,io,re
from contextlib import redirect_stdout
import numpy as np
import torch
from sampling import sampler_class
P=Path(__file__).resolve().parent
WORK=P.parent
ROOT=Path('/home/lp531/smolvla_training/pi05_bottle_sampling_ab_20261003')
sys.path.insert(0,str(WORK))
from train_lora import make_config
from lerobot.configs.policies import PreTrainedConfig
from lerobot.scripts import lerobot_train as trainer
from peft import get_peft_model_state_dict
from safetensors.torch import load_file

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def config(arm,smoke=False):
    manifest=json.loads((P/'manifest.json').read_text());source=Path(manifest['source_checkpoint'])
    cfg=make_config()
    cfg.policy=PreTrainedConfig.from_pretrained(str(source),local_files_only=True)
    cfg.policy.pretrained_path=source;cfg.policy.use_peft=True
    cfg.policy.optimizer_lr=3e-5;cfg.policy.scheduler_warmup_steps=800
    cfg.policy.scheduler_decay_steps=12000;cfg.policy.scheduler_decay_lr=1e-5
    cfg.steps=12000;cfg.save_freq=2000;cfg.eval_steps=4000
    cfg.seed=20261003;cfg.cudnn_deterministic=True;cfg.log_freq=40
    cfg.output_dir=ROOT/('smoke_'+arm if smoke else arm)
    cfg.job_name='pi05_bottle_ab_'+arm
    if smoke:
        cfg.steps=4;cfg.save_freq=4;cfg.eval_steps=0;cfg.num_workers=0;cfg.log_freq=1
    return cfg

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--arm',choices=['uniform','weighted'],required=True);parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    sys.argv=[sys.argv[0]]
    manifest=json.loads((P/'manifest.json').read_text());source=Path(manifest['source_checkpoint'])
    for path,key in [(source/'adapter_model.safetensors','source_adapter_sha256'),(source/'config.json','source_config_sha256'),(P/'weights.npy','weights_sha256'),(WORK/'split_manifest.json','split_sha256'),(Path('/home/lp531/smolvla_data/z1_tabletop_bottle_pi05_lora_v1/meta/info.json'),'dataset_meta_sha256')]:
        if sha(path)!=manifest[key]:raise RuntimeError(f'Frozen experiment input changed: {path}')
    cfg=config(args.arm,args.smoke);cfg.validate()
    tag=('smoke_' if args.smoke else '')+args.arm
    (P/f'{tag}_config.json').write_text(json.dumps(cfg.to_dict(),indent=2,default=str))
    weights=np.load(P/'weights.npy')
    if args.arm=='uniform':weights=np.ones_like(weights)
    trainer.EpisodeAwareSampler=sampler_class(weights)
    original_make_policy=trainer.make_policy
    def checked_policy(**kwargs):
        captured=io.StringIO()
        with redirect_stdout(captured):policy=original_make_policy(**kwargs)
        print(captured.getvalue(),flush=True)
        if 'All keys loaded successfully!' not in captured.getvalue():raise RuntimeError('Base weight load not verified')
        expected=load_file(str(source/'adapter_model.safetensors'))
        actual=get_peft_model_state_dict(policy)
        if actual.keys()!=expected.keys():raise RuntimeError('Adapter keys differ from common initialization')
        for key,tensor in actual.items():
            if not torch.equal(tensor.detach().cpu(),expected[key].to(dtype=tensor.dtype)):
                raise RuntimeError(f'Adapter initialization mismatch: {key}')
        learnable=sum(t.numel() for t in policy.parameters() if t.requires_grad)
        if learnable!=8358944:raise RuntimeError(f'Unexpected trainable count {learnable}')
        (P/f'{tag}_initialization.json').write_text(json.dumps({'source_checkpoint':str(source),'adapter_sha256':manifest['source_adapter_sha256'],'all_adapter_tensors_equal':True,'adapter_tensors':len(actual),'trainable_parameters':learnable,'optimizer':'fresh AdamW; identical for both arms','seed':cfg.seed},indent=2))
        print('INITIALIZATION_VERIFIED',tag,learnable,flush=True)
        return policy
    trainer.make_policy=checked_policy
    print('EXPERIMENT_START',tag,'microsteps',cfg.steps,'updates',cfg.steps//4,flush=True)
    trainer.train(cfg)
    last=cfg.output_dir/'checkpoints'/f'{cfg.steps:06d}'/'pretrained_model'
    state=load_file(str(last/'adapter_model.safetensors'))
    original=load_file(str(source/'adapter_model.safetensors'))
    if not all(torch.isfinite(t).all() for t in state.values()):raise RuntimeError('Non-finite saved adapter')
    if not any(not torch.equal(v,original[k].to(v.dtype)) for k,v in state.items()):raise RuntimeError('Training did not change adapter')
    result={'complete':True,'arm':args.arm,'smoke':args.smoke,'microsteps':cfg.steps,'optimizer_updates':cfg.steps//4,'last_checkpoint':str(last),'adapter_finite':True,'adapter_changed':True,'physical_success_evaluated':False}
    (P/f'{tag}_complete.json').write_text(json.dumps(result,indent=2));print('EXPERIMENT_COMPLETE',tag,flush=True)
if __name__=='__main__':main()
