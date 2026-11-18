"""Check full release targets, exact joint commands, spatial split and baseline recipe."""
import json
from pathlib import Path
import numpy as np
from prepare import RUN,DATA,full_labels,write
from train import make_config

split=json.loads((RUN/'split_manifest.json').read_text())
with np.load(DATA/'lowdim.npz') as archive:
 actions=archive['actions'];states=archive['states']
for record in split['source_records']:
 rows=[json.loads(line) for line in (Path(record['source_path'])/'frames.jsonl').read_text().splitlines()]
 labels,close,opened=full_labels(rows)
 a=np.asarray([row['action'] for row in rows],np.float32)
 s=np.asarray([row['observation.state'] for row in rows],np.float32)
 first,last=record['dataset_from_index'],record['dataset_to_index']
 np.testing.assert_array_equal(actions[first:last,:6],a[:,:6])
 np.testing.assert_array_equal(states[first:last],s)
 np.testing.assert_array_equal(actions[first:last,-1],labels)
 assert close==record['close_frame'] and opened==record['open_frame']
 assert np.all(labels[:close]==0) and np.all(labels[close:opened]==1) and np.all(labels[opened:]==0)
 assert record['split']==('validation' if tuple(record['grid']) in {(1,1),(2,3)} else 'train')
current=make_config().to_dict()
previous=json.loads(Path('/home/lp531/pi05_lora_workspace/act_cup_dynamic_cached_20261020/resolved_train_config.json').read_text())
for key in ['policy','optimizer','accelerator','scheduler','seed','batch_size','num_workers','steps','eval_steps','save_freq','use_policy_training_preset']:
 assert current[key]==previous[key],(key,current[key],previous[key])
assert not current['resume'] and current['policy']['pretrained_path'] is None
write('training_inputs_verified.json',dict(passed=True,episodes=len(split['source_records']),train_episodes=len(split['train_episodes']),validation_episodes=len(split['validation_episodes']),frames=len(actions),full_open_close_release_labels=True,joint_commands_exact=True,states_exact=True,baseline_hyperparameters_equal=True,fresh_act_training=True))
print((RUN/'training_inputs_verified.json').read_text())
