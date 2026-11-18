import tempfile
import unittest
from pathlib import Path
import numpy as np
from sampling import build_weights

class SamplingTests(unittest.TestCase):
    def test_windows_split_and_pool_mass(self):
        episode = np.repeat([0,1,2], 40)
        frame = np.tile(np.arange(40),3)
        records = [dict(episode_index=i,source='original' if i==0 else 'old_correction',
                        anchor_start=0 if i==0 else 10,anchor_end_inclusive=39 if i==0 else 30)
                   for i in range(3)]
        eligible, weights = build_weights(episode,frame,records,{0,1},16)
        self.assertFalse(eligible[episode==2].any())
        self.assertTrue(np.array_equal(frame[eligible & (episode==1)],np.arange(10,16)))
        self.assertAlmostEqual(weights[episode==0].sum(),.7)
        self.assertAlmostEqual(weights[episode==1].sum(),.3)
        self.assertEqual(weights[~eligible].sum(),0)
    def test_missing_correction_pool_is_rejected(self):
        with self.assertRaises(ValueError):
            build_weights(np.zeros(40,int),np.arange(40),
                          [dict(episode_index=0,source='original',anchor_start=0,anchor_end_inclusive=39)],{0},16)

if __name__=='__main__': unittest.main()
