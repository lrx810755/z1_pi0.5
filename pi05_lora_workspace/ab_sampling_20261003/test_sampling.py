import unittest
import numpy as np
import torch
from sampling import classify_frames, sampler_class

class SamplingTests(unittest.TestCase):
    def test_phase_priority_and_contiguous_source(self):
        # Motion, then open stationary wait, then first closure; no source edits.
        state=np.zeros((60,7));state[:,6]=-.9
        state[:20,3]=np.arange(20)*.03;state[20:,3]=state[19,3]
        action=state.copy();action[45:,6]=-.7
        before=(state.copy(),action.copy())
        weights,labels=classify_frames(state,action)
        self.assertEqual(weights[10],2)
        self.assertEqual(weights[30],.25)
        np.testing.assert_array_equal(weights[40:51],3)
        self.assertEqual(weights[55],1)
        np.testing.assert_array_equal(state,before[0]);np.testing.assert_array_equal(action,before[1])
        self.assertTrue(np.all(weights>0))

    def test_no_closure_rejected(self):
        q=np.zeros((20,7));q[:,-1]=-.9
        with self.assertRaises(ValueError):classify_frames(q,q)

    def test_weighted_sampling_excludes_holdout_and_maps_subset(self):
        weights=np.ones(12);weights[8:]=1000;weights[1]=20
        cls=sampler_class(weights)
        # Training uses only episode 0 and 2; episode 3 is held out.
        mapping={0:0,1:1,2:2,6:3,7:4}
        kwargs=dict(dataset_from_indices=[0,3,6,8],dataset_to_indices=[3,6,8,12],episode_indices_to_use=[0,2],shuffle=True,seed=42,absolute_to_relative_idx=mapping)
        a=cls(**kwargs);b=cls(**kwargs)
        first=list(a);self.assertEqual(first,list(b));self.assertTrue(set(first)<=set(range(5)))
        draws=[]
        for _ in range(300):draws.extend(a)
        self.assertGreater(draws.count(1)/len(draws),.7)
        a.load_state_dict({'epoch':0,'start_index':2});self.assertEqual(list(a),first[2:])

    def test_uniform_sampling_and_unshuffled_evaluation(self):
        cls=sampler_class(np.ones(10))
        sampler=cls([0,5],[5,10],episode_indices_to_use=[0],shuffle=True,seed=5)
        draws=[]
        for _ in range(1000):draws.extend(sampler)
        counts=np.bincount(draws,minlength=5)/len(draws)
        self.assertTrue(np.all(abs(counts-.2)<.03))
        eval_sampler=sampler_class(np.arange(1,11))([0,5],[5,10],episode_indices_to_use=[1],shuffle=False)
        self.assertEqual(list(eval_sampler),list(range(5,10)))

if __name__=='__main__':unittest.main()
