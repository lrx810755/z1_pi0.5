"""Verify actual sampler draws and every new raw fragment mask without GPU or policy."""
from pathlib import Path
import json
import numpy as np
from sampling import WindowSampler
from windows import anchor_mask
W=Path(__file__).resolve().parent
manifest=json.loads((W/'windows.json').read_text());records=manifest['records']
ends=np.cumsum([r['frame_count'] for r in records[:192]]);starts=np.r_[0,ends[:-1]]
s=WindowSampler(starts,ends,shuffle=True,seed=20261012)
draws=np.asarray(list(s))
with np.load(W/'existing_192_sampling_arrays.npz') as a:
    assert a['eligible_train_anchor'][draws].all()
    correction=a['correction'][s._absolute_indices]
    mass=float(s._probabilities[correction].sum())
    assert abs(mass-.3)<1e-12
    original_mass=float(s._probabilities[~correction].sum())
for r in records[192:]:
    with np.load(W/'new_fragment_masks'/(r['task_id']+'.npz')) as a:
        expected=anchor_mask(np.full(r['frame_count'],r['episode_index']),np.arange(r['frame_count']),[r])
        np.testing.assert_array_equal(expected,a['eligible_train_anchor'])
        if r['split']=='validation':assert not expected.any()
        valid=a['eligible_evaluation_anchor'];idx=np.flatnonzero(valid)
        assert idx[0]==r['anchor_start'] and idx[-1]+14==r['frame_end_inclusive']
report=dict(passed=True,draws_tested=len(draws),all_draws_inside_training_windows=True,
            existing_dataset_original_pool_mass=original_mass,existing_dataset_correction_pool_mass=mass,
            new_raw_fragment_masks_verified=len(records)-192,validation_excluded=True,
            training_started=False)
(W/'sampler_verified.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(report,ensure_ascii=False,indent=2))
