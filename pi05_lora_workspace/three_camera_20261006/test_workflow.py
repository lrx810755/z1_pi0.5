import unittest
import workflow
from types import SimpleNamespace
from unittest.mock import patch

class DebtTests(unittest.TestCase):
    def test_only_removed_reachable_cells_are_refilled(self):
        counts={(x,y):8 for y in range(4) for x in range(5)}
        deficits={(0,0):3,(1,0):3,(2,0):2,(3,0):8,(4,0):5,(2,1):2,(3,1):4,(4,1):8}
        for cell,debt in deficits.items():counts[cell]-=debt
        self.assertEqual(workflow.deficits(counts),deficits)
        self.assertEqual(sum(workflow.deficits(counts).values()),35)
        self.assertFalse(any(y==4 for x,y in workflow.deficits(counts)))
    def test_no_more_recording_once_refilled(self):
        self.assertEqual(workflow.deficits({(x,y):8 for y in range(4) for x in range(5)}),{})
    def test_failures_keep_cycling_beyond_three_rounds_until_all_35_saved(self):
        counts={(x,y):8 for y in range(4) for x in range(5)}
        for cell,n in {(0,0):3,(1,0):3,(2,0):2,(3,0):8,(4,0):5,(2,1):2,(3,1):4,(4,1):8}.items():counts[cell]-=n
        calls=[]
        def collect(args,**kwargs):
            calls.append(args)
            x,y=map(int,args[args.index('--target-grid')+1].split(','))
            self.assertLess(y,4)
            if len(calls)<=30:return SimpleNamespace(returncode=4)
            counts[(x,y)]+=int(args[args.index('--episodes')+1])
            return SimpleNamespace(returncode=0)
        with patch.object(workflow,'saved_counts',side_effect=lambda:(sum(counts.values()),counts)), \
             patch.object(workflow.subprocess,'run',side_effect=collect), \
             patch.object(workflow.run,'status'),patch.object(workflow.time,'sleep'):
            workflow.collect_missing(125)
        self.assertGreater(len(calls),30)
        self.assertEqual(sum(counts.values()),160)
        self.assertEqual(workflow.deficits(counts),{})

if __name__=='__main__':unittest.main()
