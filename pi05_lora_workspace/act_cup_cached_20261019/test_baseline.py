"""The winning bottle recipe is the authority for this cup run."""
import json
from pathlib import Path
from prepare import RUN
from train import make_config

BOTTLE=Path('/home/lp531/pi05_lora_workspace/act_gentle_cached_20261015')

def test_optimizer_architecture_and_loader_match_bottle_exactly():
    actual=make_config().to_dict();expected=json.loads((BOTTLE/'resolved_train_config.json').read_text())
    for key in ['policy','optimizer','accelerator','scheduler','parallelism','seed','batch_size','num_workers',
                'prefetch_factor','persistent_workers','dataloader_multiprocessing_context','steps',
                'eval_steps','save_freq','log_freq','cudnn_deterministic','use_policy_training_preset','ema']:
        assert actual[key]==expected[key],(key,actual[key],expected[key])
    assert actual['resume'] is False and actual['policy']['pretrained_path'] is None


def test_sampling_mass_and_all_preclose_keys_match_bottle_rule():
    import numpy as np
    split=json.loads((RUN/'split_manifest.json').read_text())
    with np.load(RUN/'sampling_arrays.npz') as a:
        assert abs(a['weights'][a['source_correction']].sum()-.3)<1e-12
        assert abs(a['weights'][~a['source_correction']].sum()-.7)<1e-12
        assert not np.isin(a['episode'][a['weights']>0],split['validation_episodes']).any()
        for r in split['source_records']:
            rows=[json.loads(x) for x in (Path(r['source_path'])/'frames.jsonl').read_text().splitlines()]
            close=r['close_frame'];expected=set(range(max(0,close-20),close))
            if r['source']=='correction':
                meta=json.loads((Path(r['source_path'])/'correction.json').read_text())
                for stage in meta['stages']:
                    expected.update(i for i,x in enumerate(rows[:close]) if stage['start_sim_time']<=x['simulation_time']<=stage['end_sim_time'])
            mask=a['episode']==r['episode_index']
            assert set(a['frame'][mask][a['key'][mask]])==expected
            valid=a['eligible_train'][mask]|a['eligible_validation'][mask]
            assert np.all(a['frame'][mask][valid]+15<r['frames'])
        for pool in [False,True]:
            eligible=a['eligible_train']&(a['source_correction']==pool)
            key=a['key']&eligible;other=~a['key']&eligible
            assert abs(a['weights'][key][0]/a['weights'][other][0]-2)<1e-12
