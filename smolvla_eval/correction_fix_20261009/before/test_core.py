import tempfile, unittest
from pathlib import Path
from core import make_tasks, completed_ids, strict_final, recovery_waypoints, atomic_json

class Tests(unittest.TestCase):
    def test_order_and_quota(self):
        tasks=make_tasks({'xyz_m':[.02,.02,.012], 'tilt_deg':12})
        self.assertEqual(len(tasks),80)
        self.assertEqual(len({t['id'] for t in tasks}),80)
        self.assertEqual([tasks[i]['kind'] for i in [0,20,40,56,62,68,72,76]],['tilt','low','low_tilt','left','right','far','near','high'])
        self.assertTrue(all(0<=t['grid'][1]<4 for t in tasks))
    def test_resume_only_valid_saved(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            for name,ok in [('episode_000001',True),('episode_000002',False),('.recording_000003',True)]:
                p=root/name;p.mkdir();atomic_json(p/'correction.json',{'task_id':name,'plan_hash':'abc','accepted':ok})
                atomic_json(p/'episode_summary.json',{'task_success':True});(p/'frames.jsonl').write_text('{}\n')
            self.assertEqual(completed_ids(root,'abc'),{'episode_000001'})
            self.assertEqual(completed_ids(root,'other'),set())
    def test_strict_rejects_box_edge(self):
        s={'task_success':True,'release_sequence_complete':True,'object_position_m':[.32,.14,.01]}
        self.assertTrue(strict_final(s));s['object_position_m']=[.267,.143,.028]
        self.assertFalse(strict_final(s))
    def test_lift_precedes_rotation(self):
        stages=recovery_waypoints([.5,0,.05],[0,0,0,1],[.5,0,.06],[1,0,0,0], 'low_tilt')
        self.assertEqual(stages[0]['orientation'],[0,0,0,1]);self.assertGreater(stages[0]['position'][2],.05)
        self.assertEqual(stages[1]['position'],stages[0]['position'])
        self.assertEqual(stages[-1]['position'],[.5,0,.06])
    def test_incomplete_does_not_count(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'episode_000001';p.mkdir();atomic_json(p/'correction.json',{'task_id':'x','plan_hash':'abc','accepted':True})
            self.assertEqual(completed_ids(Path(d),'abc'),set())
    def test_planned_sweep_checks_between_waypoints(self):
        from geometry import Geometry
        geometry=Geometry.__new__(Geometry)
        geometry.height=lambda q: .0 if .045<q[0]<.055 else .02
        with self.assertRaisesRegex(RuntimeError,'间隙'):
            geometry.check_path([[0]*7,[.1,0,0,0,0,0,0]])
    def test_supervisor_keeps_server_error_detail(self):
        import io, urllib.error
        from unittest.mock import patch
        from supervisor import api
        error=urllib.error.HTTPError('http://test',400,'bad',{},io.BytesIO(b'{"error":"IK failed"}'))
        with patch('urllib.request.urlopen',side_effect=error):
            with self.assertRaisesRegex(RuntimeError,'IK failed'):api(8769,'/api/test',{})
    def test_alignment_uses_measured_three_axis_error_and_bounds_step(self):
        from core import alignment_step
        self.assertEqual(alignment_step([.12,.01,.02],[.12,.01,.02]),[0.,0.,0.])
        step=alignment_step([.12,.018,.026],[.12,.01,.02])
        self.assertAlmostEqual(sum(x*x for x in step)**.5,.006)
        self.assertGreater(step[1],0);self.assertGreater(step[2],0)
if __name__=='__main__':unittest.main()
