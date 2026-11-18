"""Peach ACT final model; 30 predictions, execute 15 at 10Hz, no TE."""
import importlib.util
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
TRAIN=Path('/home/lp531/pi05_lora_workspace/act_peach397_chunk30_b64_20261118')
CHECKPOINT=Path('/home/lp531/smolvla_training/act_peach397_chunk30_b64_20261118/checkpoints/060000/pretrained_model')
spec=importlib.util.spec_from_file_location('z1_isaac_bridge.peach_act_runtime',HERE/'runtime_rollout.py')
runtime=importlib.util.module_from_spec(spec);sys.modules[spec.name]=runtime;spec.loader.exec_module(runtime)


def check_pins():
    import hashlib
    for filename,digest in json.loads((HERE/'source_snapshot.json').read_text())['files'].items():
        with open(filename,'rb') as stream:
            actual=hashlib.file_digest(stream,'sha256').hexdigest() if hasattr(hashlib,'file_digest') else None
            if actual is None:
                h=hashlib.sha256()
                for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
                actual=h.hexdigest()
        if actual!=digest:raise ValueError('Pinned file changed: '+filename)


class PeachRollout(runtime.SmolVLASimRollout):
    def __init__(self,args):
        check_pins()
        if args.checkpoint!=CHECKPOINT or args.policy_type!='act' or args.objects!=['peach'] or args.execute_steps!=15 or args.episodes_per_object!=30:
            raise ValueError('Require final peach ACT and exactly30 episodes/execute15')
        if (args.rate,args.execution_rate,args.binary_gripper,args.gripper_action_encoding,args.freeze_target_until_close)!=(10,60,True,'binary',False):
            raise ValueError('Unexpected execution contract')
        if not json.loads((HERE/'bridge_verified.json').read_text())['passed']:raise ValueError('Bridge verification required')
        super().__init__(args)
        self.coverage=json.loads((HERE/'coverage.json').read_text())
        data=self.coverage['objects']['peach']
        self.covered_cells={'peach':{tuple(c) for c in data['cells']}}
        self.training_target_positions={'peach':data['positions']}
        self.training_seeds=set(self.coverage['training_seeds'])

    def run_suite(self):
        root=super().run_suite()
        sys.path.insert(0,'/home/lp531/robot/pi05_four_objects_eval_20261117')
        from reporting import stage_summary
        results=[json.loads(p.read_text()) for p in sorted((root/'episodes').glob('*/result.json'))]
        (root/'stage_report.json').write_text(json.dumps(stage_summary(results),indent=2)+'\n')
        (root/'evaluation_provenance.json').write_text(json.dumps(dict(checkpoint=str(CHECKPOINT),predicted_steps=30,execute_steps=15,action_hz=10,execution_hz=60,temporal_ensemble_used=False,coverage=self.coverage,evaluation_kind='new seeds in training-covered cells; no held-out dataset'),indent=2)+'\n')
        return root


if __name__=='__main__':
    runtime.SmolVLASimRollout=PeachRollout
    runtime.main()
