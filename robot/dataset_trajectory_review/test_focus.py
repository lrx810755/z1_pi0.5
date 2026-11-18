import unittest
from render_focus import select_holds


class FocusTests(unittest.TestCase):
    def test_only_long_continuous_holds_are_selected_and_inputs_stay_unchanged(self):
        def item(name,kind,seconds):
            return dict(dataset='test',episode=name,findings=[dict(type=kind,recorded_s=seconds)])
        source=[item('a','长停顿候选',7.9),item('b','长停顿候选',8.),
                item('c','同类阶段耗时偏长',100.),item('d','长停顿候选',11.)]
        result=select_holds(source,8.)
        self.assertEqual([e['episode'] for e in result],['d','b'])
        result[0]['findings'][0]['recorded_s']=20
        self.assertEqual(source[3]['findings'][0]['recorded_s'],11.)


if __name__=='__main__':unittest.main()
