import unittest
from check_progress import decide

class CheckTests(unittest.TestCase):
    def test_incomplete_stopped_restarts_collection(self):
        self.assertEqual(decide(127,'failed',False),'resume_collection')
    def test_complete_dataset_stopped_starts_training_pipeline(self):
        self.assertEqual(decide(160,'failed',False),'start_audit_and_training')
    def test_running_collection_or_conversion_is_not_duplicated(self):
        for phase in ('collecting','auditing','converting','preflight'):
            self.assertEqual(decide(160,phase,True),'continue_running')
    def test_formal_training_and_finished_stop_checks(self):
        self.assertEqual(decide(160,'training',True),'training_started')
        self.assertEqual(decide(160,'complete',False),'finished')
    def test_failed_training_is_restarted(self):
        self.assertEqual(decide(160,'training',False),'start_audit_and_training')

if __name__=='__main__':unittest.main()
