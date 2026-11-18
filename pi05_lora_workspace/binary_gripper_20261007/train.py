"""Prepare/verify binary-gripper LoRA; train only with an explicit --train flag."""
import argparse
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import re
import sys

RUN = Path(__file__).resolve().parent
WORK = RUN.parent
DATA = Path('/home/lp531/smolvla_data/z1_bottle_three_camera_pi05_binary_20261007')
OUTPUT = Path('/home/lp531/smolvla_training/pi05_bottle_binary_chunk15_attn_ffn_20261007')
sys.path.insert(0, str(WORK))
from train_lora import make_config as previous_config


def make_config():
    cfg = previous_config(microbatch=1)
    cfg.dataset.root = str(DATA)
    cfg.dataset.repo_id = 'lp531/z1_bottle_three_camera_pi05_binary_20261007'
    cfg.dataset.eval_split = 0.0
    cfg.eval_steps = 0
    cfg.policy.chunk_size = 15
    cfg.policy.optimizer_lr = 5e-5
    cfg.policy.scheduler_decay_lr = 5e-6
    cfg.peft.target_modules = (
        r'.*paligemma_with_expert\.(paligemma\.model\.language_model|gemma_expert\.model)'
        r'\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)'
    )
    accumulation = cfg.accelerator.gradient_accumulation.steps
    cfg.steps = 10000 * accumulation
    cfg.policy.scheduler_warmup_steps = 500 * accumulation
    cfg.policy.scheduler_decay_steps = cfg.steps
    cfg.save_freq = 1000 * accumulation
    cfg.log_freq = 10 * accumulation
    cfg.output_dir = OUTPUT
    cfg.job_name = 'pi05_bottle_binary_chunk15_attn_ffn'
    cfg.seed = 20261007
    return cfg


def verify(cfg):
    import torch
    from lerobot.configs.types import FeatureType
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    from lerobot.utils.feature_utils import dataset_to_policy_features

    audit = json.loads((RUN / 'conversion_audit.json').read_text())
    assert audit['passed'] and audit['episodes'] == 160 and audit['frames'] == 37557
    contract = json.loads((DATA / 'meta/gripper_contract.json').read_text())
    assert contract['action_gripper_encoding'] == {'open': 0, 'close': 1}
    meta = LeRobotDatasetMetadata(cfg.dataset.repo_id, root=DATA)
    assert meta.stats['action']['q01'][-1] == 0 and meta.stats['action']['q99'][-1] == 1
    features = dataset_to_policy_features(meta.features)
    cfg.policy.output_features = {k: v for k, v in features.items() if v.type is FeatureType.ACTION}
    cfg.policy.input_features = {k: v for k, v in features.items() if v.type is not FeatureType.ACTION}
    device = cfg.policy.device
    cfg.policy.device = 'meta'
    try:
        # Construct real modules and adapters with metadata-only tensors, without GPU allocation/training.
        with torch.device('meta'):
            policy = PI05Policy(deepcopy(cfg.policy))
            names = [name for name, module in policy.named_modules()
                     if isinstance(module, torch.nn.Linear) and re.fullmatch(cfg.peft.target_modules, name)]
            coverage = Counter(('language' if '.language_model.' in n else 'expert',
                                'attention' if '.self_attn.' in n else 'ffn') for n in names)
            assert dict(coverage) == {('language','attention'):72, ('language','ffn'):54,
                                      ('expert','attention'):72, ('expert','ffn'):54}, coverage
            wrapped = policy.wrap_with_peft(peft_cli_overrides=asdict(cfg.peft))
            trainable = [(n, p) for n, p in wrapped.named_parameters() if p.requires_grad]
            assert trainable and all('lora_' in n or 'modules_to_save' in n for n, _ in trainable)
            assert not any('vision_tower' in n for n, _ in trainable)
            for suffix in cfg.peft.full_training_modules:
                assert any(suffix in n and 'modules_to_save' in n for n, _ in trainable), suffix
            result = dict(passed=True, verification='real model + PEFT on meta tensors; no weights loaded or training run',
                          module_coverage={f'{network}_{kind}': count for (network,kind),count in coverage.items()},
                          matched_modules=len(names), trainable_parameters=sum(p.numel() for _,p in trainable),
                          effective_batch=cfg.batch_size*cfg.accelerator.gradient_accumulation.steps,
                          optimizer_updates=cfg.steps//cfg.accelerator.gradient_accumulation.steps,
                          chunk_size=cfg.policy.chunk_size, learning_rate=cfg.policy.optimizer_lr,
                          base_weights=str(cfg.policy.pretrained_path), vision_frozen=True,
                          full_training_modules=cfg.peft.full_training_modules,
                          binary_quantiles=[0,1], train_episodes=160, validation_episodes=0,
                          training_started=False)
    finally:
        cfg.policy.device = device
    assert (Path(cfg.policy.pretrained_path) / 'model.safetensors').is_file()
    (RUN / 'training_setup_verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', action='store_true', help='Explicitly start training; omitted means config verification only')
    args = parser.parse_args()
    sys.argv = [sys.argv[0]]
    cfg = make_config()
    cfg.validate()
    verify(cfg)
    cfg.validate()
    (RUN / 'resolved_train_config.json').write_text(json.dumps(cfg.to_dict(), indent=2, default=str) + '\n')
    if args.train:
        from lerobot.scripts.lerobot_train import train
        train(cfg)


if __name__ == '__main__':
    main()
