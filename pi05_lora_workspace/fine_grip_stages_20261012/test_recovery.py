import unittest
import numpy as np
from recovery import apply_recovery_weight

class RecoveryTests(unittest.TestCase):
    def test_logged_recovery_is_boosted_and_overlap_uses_maximum(self):
        rows=[{'simulation_time':i/10} for i in range(8)]
        stages=[dict(stage='approach',start_sim_time=.2,end_sim_time=.5)]
        weight,mask=apply_recovery_weight(rows,stages,5,np.array([1.,1.,1.,1.5,2.,2.,1.,1.]))
        np.testing.assert_array_equal(weight,[1,1,2,2,2,2,1,1])
        np.testing.assert_array_equal(np.flatnonzero(mask),[2,3,4])
    def test_preparation_and_missing_stages_are_not_invented_recovery(self):
        rows=[{'simulation_time':i/10} for i in range(8)]
        for stages in [[],[dict(stage='prepare_lift',start_sim_time=0,end_sim_time=.7)]]:
            weight,mask=apply_recovery_weight(rows,stages,5,np.ones(8))
            self.assertFalse(mask.any());np.testing.assert_array_equal(weight,np.ones(8))
if __name__=='__main__':unittest.main()
