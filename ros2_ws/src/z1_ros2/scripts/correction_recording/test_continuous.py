import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from supervisor import collection_rounds

class ContinuousTests(unittest.TestCase):
    def test_retries_past_initial_round_even_without_progress(self):
        with patch('supervisor.completed_ids',side_effect=[set(),set(),set(),{'a'}]):
            self.assertEqual(list(collection_rounds(Path('/unused'),{'hash':'h','tasks':[{'id':'a'}]},1,True)),[0,1,2])
    def test_completed_dataset_never_starts_another_round(self):
        with patch('supervisor.completed_ids',return_value={'a'}):
            self.assertEqual(list(collection_rounds(Path('/unused'),{'hash':'h','tasks':[{'id':'a'}]},1,True)),[])
    def test_finite_mode_still_respects_round_limit(self):
        with patch('supervisor.completed_ids',return_value=set()):
            self.assertEqual(list(collection_rounds(Path('/unused'),{'hash':'h','tasks':[{'id':'a'}]},2,False)),[0,1])

if __name__=='__main__':unittest.main()
