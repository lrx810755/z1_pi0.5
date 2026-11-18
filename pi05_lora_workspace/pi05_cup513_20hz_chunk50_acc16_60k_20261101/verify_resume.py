from verify import verify_prepared
from config import RUN
import json
p=verify_prepared()
checkpoint=__import__('pathlib').Path(json.loads((RUN/'latest_checkpoint.json').read_text())['checkpoint'])
for f in ['optimizer_param_groups.json','optimizer_state.safetensors','rng_state.safetensors','scheduler_state.json','training_step.json']:
    assert (checkpoint.parent/'training_state'/f).stat().st_size > 0
print('Prepared data pins and complete checkpoint files verified')
