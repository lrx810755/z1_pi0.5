import copy, json, math, unittest
from pathlib import Path
from select_candidates import select_window, good_row, prefix_violation

class SelectionTests(unittest.TestCase):
    def row(self,t,x=.42):
        return {'simulation_elapsed_s':t,'joint_state':[0,0,0,0,0,0,-.9],'command_target_collisions_enabled':True,'command_target_frozen':False,'contact_status':{'object_position_m':[x,-.1,0],'object_speed_m_s':.01,'cylinder_tilt_deg':1,'cylinder_tilt_valid':True,'object_touching_table':True,'object_pose_stamp_s':t+100,'gripper_commanded_closed':False,'physical_grasp_ok':False,'was_grasped':False,'target_collisions_enabled':True,'frozen':False}}
    def sequence(self):return [self.row(3+i*.05) for i in range(12)]
    def test_real_stable_push_accepts(self):
        window,reason=select_window(self.sequence(),[.4,-.1,0]);self.assertEqual(window,(0,8));self.assertIsNone(reason)
    def test_initial_layout_randomization_is_not_policy_push(self):
        window,_=select_window(self.sequence(),[.42,-.1,0]);self.assertIsNone(window)
    def test_previous_tilt_then_upright_must_reject(self):
        rows=self.sequence();rows[0]['contact_status']['cylinder_tilt_deg']=40
        self.assertIsNone(select_window(rows,[.4,-.1,0])[0])
    def test_previous_flight_then_table_must_reject(self):
        rows=self.sequence();rows[0]['contact_status']['object_position_m'][2]=.03
        self.assertIsNone(select_window(rows,[.4,-.1,0])[0])
    def test_closed_gripper_breaks_sustained_window(self):
        rows=self.sequence();rows[5]['contact_status']['gripper_commanded_closed']=True
        self.assertIsNone(select_window(rows,[.4,-.1,0])[0])
    def test_fast_motion_breaks_sustained_window(self):
        rows=self.sequence();rows[5]['contact_status']['object_speed_m_s']=.021
        self.assertIsNone(select_window(rows,[.4,-.1,0])[0])
    def test_stale_pose_cannot_fake_stability(self):
        rows=self.sequence()
        for r in rows:r['contact_status']['object_pose_stamp_s']=100
        self.assertIsNone(select_window(rows,[.4,-.1,0])[0])
    def test_table_contact_is_required(self):
        r=self.row(3);r['contact_status']['object_touching_table']=False
        self.assertFalse(good_row(r,[.4,-.1,0]))
    def test_actual_manifest_against_full_source_prefixes(self):
        m=json.loads((Path(__file__).parent/'candidates_manifest.json').read_text())
        self.assertEqual(len(m['candidates']),19)
        self.assertEqual(len({c['source_episode_path'] for c in m['candidates']}),19)
        for c in m['candidates']:
            with self.subTest(candidate=c['candidate_id']):
                self.assertFalse(c['source_final_box_success'])
                rows=[json.loads(x) for x in Path(c['source_trace_path']).read_text().splitlines()]
                i=c['takeover_trace_index_zero_based'];s=c['stable_window']['start_trace_index_zero_based'];initial=c['cup_initial_xyz_m']
                self.assertTrue(all(prefix_violation(x,initial) is None for x in rows[:i+1]))
                self.assertTrue(all(good_row(x,initial,c['selection_displacement_threshold_m']) for x in rows[s:i+1]))
                self.assertGreaterEqual(rows[i]['simulation_elapsed_s']-rows[s]['simulation_elapsed_s'],.4-1e-9)
                self.assertEqual(len(Path(c['source_prefix_commands_path']).read_text().splitlines()),i+1)
                self.assertLess(c['push_not_initial_settling_evidence']['max_displacement_first2s_m'],.001)
                self.assertGreater(c['push_not_initial_settling_evidence']['first_1mm_displacement_time_s'],10)
                self.assertEqual(len(c['initial_five_object_layout']),5)
                self.assertEqual(c['cup_xyz_m'],rows[i]['contact_status']['object_position_m'])
                self.assertEqual(c['cup_orientation_xyzw'],rows[i]['contact_status']['object_orientation_xyzw'])
if __name__=='__main__':unittest.main(verbosity=2)
