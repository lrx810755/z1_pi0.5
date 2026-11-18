"""Continue four-demo weights with a new optimizer and explicit microstep schedule."""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

RUN = Path(__file__).resolve().parent
WORK = RUN.parent
SOURCE = Path('/home/lp531/smolvla_training/pi05_bottle_small4_chunk15_attn_ffn_20261007/checkpoints/008000/pretrained_model')
OUTPUT = Path('/home/lp531/smolvla_training/pi05_bottle_small4_continue_20261008')
spec = importlib.util.spec_from_file_location('small_previous', WORK/'small_overfit_20261007/train.py')
previous = importlib.util.module_from_spec(spec)
spec.loader.exec_module(previous)
previous.RUN = RUN

def make_config():
    cfg = previous.make_config()
    cfg.policy.pretrained_path = SOURCE
    cfg.policy.use_peft = True
    cfg.policy.optimizer_lr = 2e-5
    cfg.policy.scheduler_decay_lr = 2e-6
    accumulation = cfg.accelerator.gradient_accumulation.steps
    cfg.steps = 10000 * accumulation
    cfg.policy.scheduler_warmup_steps = 100 * accumulation
    cfg.policy.scheduler_decay_steps = cfg.steps
    cfg.save_freq = 1000 * accumulation
    cfg.output_dir = OUTPUT
    cfg.job_name = 'pi05_small4_continue'
    return cfg

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    args = parser.parse_args()
    sys.argv = [sys.argv[0]]
    cfg = make_config()
    cfg.validate()
    # Reuse the actual four-episode decoding checks, whose original assertion expects base weights.
    cfg.policy.pretrained_path = WORK/'weights/pi05_base'
    try:
        previous.verify(cfg)
    finally:
        cfg.policy.pretrained_path = SOURCE
    assert (SOURCE/'adapter_model.safetensors').is_file()
    assert cfg.steps == cfg.policy.scheduler_decay_steps == 80000
    assert cfg.policy.scheduler_warmup_steps == 800
    assert cfg.batch_size * cfg.accelerator.gradient_accumulation.steps == 8
    verification = json.loads((RUN/'dataset_verification.json').read_text())
    verification.update(optimizer_updates=10000, save_optimizer_updates=list(range(1000,10001,1000)),
                        base_weights=str(WORK/'weights/pi05_base'), initial_adapter=str(SOURCE),
                        fresh_optimizer=True, peak_lr=2e-5, end_lr=2e-6, warmup_optimizer_updates=100)
    (RUN/'dataset_verification.json').write_text(json.dumps(verification, indent=2))
    (RUN/'resolved_train_config.json').write_text(json.dumps(cfg.to_dict(), indent=2, default=str))
    if args.train:
        import lerobot.scripts.lerobot_train as trainer
        from check_actions import check_actions
        original_save = trainer.save_checkpoint
        def save_and_check(**kwargs):
            # The extra first-update checkpoint exercises the entire check before unattended training.
            kwargs['cfg'].save_freq = 8000
            original_save(**kwargs)
            check_actions(**kwargs)
        trainer.save_checkpoint = save_and_check
        cfg.save_freq = 8
        trainer.train(cfg)
        (RUN/'completed.json').write_text(json.dumps({'optimizer_updates':10000,'output':str(OUTPUT)}))

if __name__ == '__main__':
    main()
