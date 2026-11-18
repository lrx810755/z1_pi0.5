import unittest
import numpy as np
from sampling import weighted_probabilities

class FineSamplingTests(unittest.TestCase):
    def test_real_phase_ratios_and_source_masses(self):
        ep=np.array([0,0,0,160,160,160,3,170])
        eligible=np.array([1,1,1,1,1,1,0,0],bool)
        w=np.array([1,1.5,2,1,1.5,2,2,2],float)
        p=weighted_probabilities(ep,eligible,w)
        self.assertAlmostEqual(p[ep<160].sum(),.7)
        self.assertAlmostEqual(p[ep>=160].sum(),.3)
        for base in [0,3]:
            self.assertAlmostEqual(p[base+1]/p[base],1.5)
            self.assertAlmostEqual(p[base+2]/p[base],2.)
        np.testing.assert_array_equal(p[-2:],[0.,0.])

    def test_recovery_overlay_uses_maximum(self):
        phase=np.array([1,1.5,2,2,1.5,1.])
        recovery=np.array([0,1,1,0,0,0],bool)
        actual=np.maximum(phase,np.where(recovery,2.,1.))
        np.testing.assert_array_equal(actual,[1,2,2,2,1.5,1])

if __name__=='__main__':unittest.main()
