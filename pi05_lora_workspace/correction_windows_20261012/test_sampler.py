import json
from pathlib import Path
import unittest
import tempfile
import numpy as np
W=Path(__file__).resolve().parent

class SamplerTests(unittest.TestCase):
    def test_extra_dataset_rows_cannot_silently_use_a_smaller_dataset(self):
        with np.load(W/'existing_192_sampling_arrays.npz') as a:
            arrays={k:np.r_[a[k],a[k][-1]] for k in a.files}
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'wrong_dataset.npz';np.savez(p,**arrays)
            with self.assertRaises(ValueError):self.make_sampler(eligibility_path=p)

    def make_sampler(self,shuffle=True,**kwargs):
        from sampling import WindowSampler
        records=json.loads((W/'windows.json').read_text())['records'][:192]
        ends=np.cumsum([r['frame_count'] for r in records]);starts=np.r_[0,ends[:-1]]
        return WindowSampler(starts,ends,shuffle=shuffle,seed=43,**kwargs)

    def test_every_draw_is_in_training_window_and_pool_mass_is_preserved(self):
        a=np.load(W/'existing_192_sampling_arrays.npz');sampler=self.make_sampler()
        draws=np.array(list(sampler))
        self.assertTrue(a['eligible_train_anchor'][draws].all())
        self.assertEqual(len(draws),len(sampler.indices))
        self.assertAlmostEqual(float(sampler._probabilities[a['correction'][sampler._absolute_indices]].sum()),.3)

    def test_resume_reproduces_remaining_draws(self):
        expected=list(self.make_sampler());resumed=self.make_sampler()
        resumed.load_state_dict({'epoch':0,'start_index':101})
        self.assertEqual(list(resumed),expected[101:])

    def test_sequential_sampling_still_excludes_validation_and_invalid_anchors(self):
        a=np.load(W/'existing_192_sampling_arrays.npz');sampler=self.make_sampler(shuffle=False)
        draws=np.array(list(sampler))
        self.assertTrue(a['eligible_train_anchor'][draws].all())
        self.assertTrue((np.diff(draws)>0).all())

    def test_relative_subset_mapping_is_preserved(self):
        base=self.make_sampler(shuffle=False);allowed=base.indices[::3]
        mapping={v:i for i,v in enumerate(allowed)}
        # A real filtered dataset exposes only absolute indices listed in the mapping.
        sampler=self.make_sampler(shuffle=False,absolute_to_relative_idx=mapping)
        self.assertEqual(list(sampler),list(range(len(allowed))))

if __name__=='__main__':unittest.main()
