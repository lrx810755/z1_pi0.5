import unittest
from mapping import map_records

class MappingTests(unittest.TestCase):
    def test_source_identity_resolves_reordered_append_indices(self):
        records=[dict(episode_index=192,source='new_fragment',source_path='a',frame_count=40,anchor_start=3,anchor_end_inclusive=20),
                 dict(episode_index=193,source='new_fragment',source_path='b',frame_count=50,anchor_start=2,anchor_end_inclusive=30)]
        actual=map_records(records,{'a':193,'b':192},{192:(1000,50),193:(1050,40)})
        self.assertEqual([r['source_path'] for r in actual],['b','a'])
        self.assertEqual(actual[0]['absolute_anchor_start'],1002)
        self.assertEqual(actual[1]['absolute_anchor_end_inclusive'],1070)
    def test_frame_count_mismatch_is_rejected(self):
        records=[dict(episode_index=192,source='new_fragment',source_path='a',frame_count=40,anchor_start=0,anchor_end_inclusive=20)]
        with self.assertRaises(ValueError):map_records(records,{'a':192},{192:(0,41)})
    def test_unknown_source_cannot_fall_back_to_planned_id(self):
        records=[dict(episode_index=192,source='new_fragment',source_path='a',frame_count=40,anchor_start=0,anchor_end_inclusive=20)]
        with self.assertRaises(ValueError):map_records(records,{}, {192:(0,40)})
if __name__=='__main__':unittest.main()
