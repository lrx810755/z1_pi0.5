import unittest
import run

class SplitTests(unittest.TestCase):
    def test_holdout_covers_grids_without_splitting_episodes(self):
        entries=[{'scene_randomization':{'target_grid':[g,0],'seed':g*8+i},'source':str(g*8+i)} for g in range(20) for i in range(8)]
        train,val=run.split_entries(entries)
        self.assertEqual((len(train),len(val)),(140,20))
        self.assertEqual({tuple(x['scene_randomization']['target_grid']) for x in val},{(g,0) for g in range(20)})
        self.assertFalse({x['source'] for x in train}&{x['source'] for x in val})
        self.assertEqual(train+val, sum(run.split_entries(entries),[]))
    def test_duplicate_seed_rejected(self):
        entries=[{'scene_randomization':{'target_grid':[0,0],'seed':1},'source':str(i)} for i in range(2)]
        with self.assertRaises(ValueError):run.split_entries(entries)
    def test_single_episode_rejected(self):
        with self.assertRaises(ValueError):run.split_entries([{'scene_randomization':{'seed':1},'source':'one'}])

if __name__=='__main__':unittest.main()
