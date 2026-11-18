"""Choose an existing saved adapter using the recorded held-out loss."""
import json
from pathlib import Path
import re

work = Path(__file__).resolve().parent
output = Path('/home/lp531/smolvla_training/pi05_bottle_lora_20261001')
matches = re.findall(r'step (\d+): eval_loss=([\d.eE+-]+)', (work / 'training.log').read_text())
ranked = sorted((float(loss), int(step)) for step, loss in matches)
for loss, step in ranked:
    model = output / 'checkpoints' / f'{step:06d}' / 'pretrained_model'
    if model.is_dir():
        link = output / 'best'
        link.symlink_to(model, target_is_directory=True)
        (output / 'best_validation.json').write_text(json.dumps({'step': step, 'eval_loss': loss,
            'path': str(model), 'criterion': 'held-out demonstration flow-matching loss; task success not measured'},
            indent=2) + '\n')
        print('BEST_SAVED_ADAPTER', model, 'eval_loss', loss)
        break
else:
    raise RuntimeError('No evaluated checkpoint exists')
