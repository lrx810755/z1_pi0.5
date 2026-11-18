import unittest
from protocol import validate_profile, rank_profiles

class ProtocolTests(unittest.TestCase):
    def test_invalid_parameters(self):
        for values in [(1,2,80),(float('nan'),.8,80),(1,.8,0)]:
            with self.assertRaises(ValueError):validate_profile(*values)

    def test_incomplete_trials_cannot_win(self):
        rows=[dict(profile='a',strict_success=True,grasped=True,success=True,penetration_m=0)]
        self.assertEqual(rank_profiles(rows,2,{'a':80}),[])

    def test_all_failures_do_not_produce_recommendation(self):
        rows=[dict(profile='a',strict_success=False,grasped=False,success=False,penetration_m=0)]
        self.assertEqual(rank_profiles(rows,1,{'a':80}),[])

    def test_lower_effort_breaks_equal_performance(self):
        rows=[dict(profile=p,strict_success=True,grasped=True,success=True,penetration_m=0) for p in ['a','b']]
        self.assertEqual(rank_profiles(rows,1,{'a':800,'b':80})[0]['profile'],'b')

if __name__=='__main__':unittest.main()
