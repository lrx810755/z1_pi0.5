import unittest

from prepare_dataset import split_episodes


class SplitTests(unittest.TestCase):
    def test_stratified_complete_disjoint_and_deterministic(self):
        entries = [{"group": group} for group, count in
                   [("baseline", 184), ("left", 22), ("right", 20), ("high", 20)]
                   for _ in range(count)]
        train, val = split_episodes(entries)
        self.assertEqual((len(train), len(val)), (221, 25))
        self.assertEqual(set(train) | set(val), set(range(246)))
        self.assertFalse(set(train) & set(val))
        self.assertEqual(split_episodes(entries), (train, val))
        self.assertEqual({g: sum(entries[i]["group"] == g for i in val)
                          for g in ("baseline", "left", "right", "high")},
                         {"baseline": 19, "left": 2, "right": 2, "high": 2})


if __name__ == "__main__":
    unittest.main()
