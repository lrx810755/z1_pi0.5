import unittest
from selection import check_requested_tasks

class SelectionTests(unittest.TestCase):
    def test_explicit_stopped_snapshot_accepts_nineteen_of_twenty(self):
        self.assertEqual(check_requested_tasks(set(range(20)),set(range(19)),list(range(19))),19)
    def test_default_full_plan_does_not_silently_accept_nineteen(self):
        with self.assertRaises(ValueError):check_requested_tasks(set(range(20)),set(range(19)),list(range(20)))
    def test_cannot_drop_an_already_accepted_sample(self):
        with self.assertRaises(ValueError):check_requested_tasks(set(range(20)),set(range(19)),list(range(18)))
    def test_duplicate_or_foreign_tasks_are_rejected(self):
        for requested in [[0,0],[0,21]]:
            with self.assertRaises(ValueError):check_requested_tasks(set(range(20)),{0},requested)
if __name__=='__main__':unittest.main()
