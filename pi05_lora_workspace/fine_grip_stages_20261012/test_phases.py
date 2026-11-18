import unittest
from phases import annotate

def rows():
    return [dict(phase='READY' if i<10 else 'GRASPING' if i<20 else 'PHYSICAL_TEST_LIFT' if i<30 else 'TRANSPORT',
                 action=[0]*6+([-0.9] if i<10 else [-.8+(i-10)*.04] if i<14 else [-.64]),
                 physical_grasp_ok=i>=30,object_touching_table=i<30,grasp_success=i>=30) for i in range(60)]

class PhaseTests(unittest.TestCase):
    def test_unchanged_command_suffix_is_pre_lift_hold(self):
        ids,w,e=annotate(rows(),10,opening=55)
        self.assertEqual(e['hold_start'],15)
        self.assertEqual(e['lift_start'],20)
        self.assertEqual(e['stable_confirmed'],34)
        self.assertEqual(list(ids[14:21]),[2,3,3,3,3,3,4])
        self.assertEqual(list(w[19:21]),[2.,1.5])
        self.assertEqual(ids[50],0)
    def test_early_plateau_followed_by_readjustment_is_not_hold(self):
        r=rows();r[18]['action'][-1]=-.7
        ids,w,e=annotate(r,10,opening=55)
        self.assertIsNone(e['hold_start'])
        self.assertTrue(all(ids[10:20]==2))
    def test_transient_grasp_does_not_confirm_lift(self):
        r=rows()
        for i in range(21,24):r[i].update(physical_grasp_ok=True,object_touching_table=False,grasp_success=True)
        self.assertEqual(annotate(r,10,opening=55)[2]['stable_confirmed'],34)
    def test_missing_lift_is_rejected(self):
        r=rows()
        for x in r:x['phase']='GRASPING'
        with self.assertRaises(ValueError):annotate(r,10)
if __name__=='__main__':unittest.main()
