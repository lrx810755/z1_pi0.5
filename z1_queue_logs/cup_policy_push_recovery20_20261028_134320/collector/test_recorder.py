"""Pure acceptance tests; no ROS, scene or GPU required."""
import copy
import importlib.util
from pathlib import Path
import unittest
import json
import os
import tempfile
import threading
from types import SimpleNamespace
from unittest.mock import patch

try:
    from recorder import accept_episode, validate_start, source_scene
except ImportError:
    accept_episode=validate_start=source_scene=None

class AcceptanceTests(unittest.TestCase):
    def setUp(self):
        self.ctx={'task':{'id':'push_1','kind':'push_recovery','grid':[2,1],'split':'train'},'source_identity':{'model':'source'},'plan_hash':'hash','attempt_id':'a','candidate':{'seed':12,'source_final_box_success':False,'cup_xyz_m':[.4,-.1,0],'raw_initial_scene_randomization':{'seed':12,'target_grid':None,'object_positions_m':{'cup':[.42,-.1,0]}}}}
        self.rows=[]
        for i in range(50):
            grip=-.9 if i<10 or i>=30 else -.3
            self.rows.append({'simulation_time':100+i*.1,'action':[0,0,0,0,0,0,grip],'observation.state':[0,0,0,0,0,0,grip]})
        self.meta={'recording_ready':True,'recovery_complete':True,'recovery_start_sim_s':100,'recovery_end_sim_s':102,'stable_grasp_end_sim_s':102.2,'withdrawal_rise_m':.04,'events':[{'event':'gripper_intent','time':101,'state':'close','target':-.3},{'event':'gripper_intent','time':103,'state':'open','target':-.9}]}
        self.final={k:True for k in ('task_success','physical_grasp_ok','placement_stable','release_sequence_complete','released','target_collisions_enabled')}
        self.final.update(frozen=False,holding_observed=False,left_finger_touching_table=False,right_finger_touching_table=False)
        self.final.update(inside_success_volume=True,object_speed_m_s=0.,gripper_commanded_closed=False,gripper_position_rad=-.9)
        self.dynamic={'samples':150,'violations':[],'finger_table_contact':False,'frame_samples':50,'frame_violations':[]}
        self.dynamic['status_samples']=[{'simulation_time':103+i*.1,'status':copy.deepcopy(self.final)} for i in range(20)]
        self.evidence={k:True for k in ('physical_grasp_verified','object_lifted_ever','bilateral_contact_ever','release_sequence_complete','task_success')}
    def accept(self,**kw):
        self.assertTrue(callable(accept_episode),'acceptance helper must exist')
        args=dict(rows=self.rows,meta=self.meta,status=self.final,dynamic=self.dynamic,ctx=self.ctx,roll=-90.,episode_evidence=self.evidence)
        args.update(kw);return accept_episode(**args)
    def test_accepts_complete_expert_only_episode(self):
        data=self.accept();self.assertTrue(all(data['training_mask']['expert_frame_mask']))
        self.assertEqual(sum(data['training_mask']['valid_action_horizon30_anchors']),21)
        self.assertEqual(data['correction']['recovery_intervals_s'],[{'start':100.,'end':102.2}])
        self.assertEqual(data['correction']['source_type'],'actual_policy_push_recovery')
        self.assertEqual(data['correction']['actual_takeover_cup_position_m'],[.4,-.1,0])
    def test_rejects_final_success_without_physical_grasp(self):
        self.final['physical_grasp_ok']=False
        with self.assertRaisesRegex(ValueError,'physical_grasp'):self.accept()
    def test_rejects_success_without_recorded_bilateral_lift(self):
        self.evidence['object_lifted_ever']=False
        with self.assertRaisesRegex(ValueError,'lift'):self.accept()
    def test_rejects_holding_after_release(self):
        self.final['holding_observed']=True
        with self.assertRaisesRegex(ValueError,'holding'):self.accept()
    def test_latched_success_cannot_accept_cup_currently_outside_box(self):
        self.final['inside_success_volume']=False
        with self.assertRaisesRegex(ValueError,'inside_success'):self.accept()
    def test_latched_success_cannot_accept_recent_unstable_release(self):
        self.dynamic['status_samples'][-3]['status']['object_speed_m_s']=.2
        with self.assertRaisesRegex(ValueError,'stable.*tail'):self.accept()
    def test_rejects_one_dynamic_status_violation(self):
        self.dynamic['violations']=[{'frozen':True}]
        with self.assertRaisesRegex(ValueError,'dynamic'):self.accept()
    def test_rejects_missing_perframe_physics_evidence(self):
        self.dynamic['frame_samples']=49
        with self.assertRaisesRegex(ValueError,'frame'):self.accept()
    def test_rejects_finger_table_contact_ever(self):
        self.dynamic['finger_table_contact']=True
        with self.assertRaisesRegex(ValueError,'table'):self.accept()
    def test_rejects_wrong_roll(self):
        with self.assertRaisesRegex(ValueError,'90'):self.accept(roll=-85.)
    def test_rejects_incomplete_release_tail(self):
        self.dynamic['frame_samples']=45
        with self.assertRaisesRegex(ValueError,'16'):self.accept(rows=self.rows[:45])
    def test_rejects_wrong_or_missing_recovery_interval(self):
        self.meta.pop('stable_grasp_end_sim_s')
        with self.assertRaisesRegex(ValueError,'stable.grasp'):self.accept()
    def test_rejects_insufficient_withdrawal(self):
        self.meta['withdrawal_rise_m']=.01
        with self.assertRaisesRegex(ValueError,'withdrawal'):self.accept()
    def test_rejects_intervention_actions_in_expert_recording(self):
        self.meta['events'].append({'event':'intervention_start','time':100.5})
        with self.assertRaisesRegex(ValueError,'intervention'):self.accept()
    def test_start_requires_expert_hold_gate(self):
        self.assertTrue(callable(validate_start),'start gate must exist')
        self.meta['recording_ready']=False
        with self.assertRaisesRegex(ValueError,'hold'):validate_start(self.ctx,self.meta,self.final)
    def test_scene_metadata_preserves_original_layout(self):
        self.assertTrue(callable(source_scene),'source metadata helper must exist')
        scene=source_scene(self.ctx)
        self.assertEqual(scene['object_positions_m']['cup'],[.42,-.1,0]);self.assertEqual(scene['source_grid'],[2,1])
    def test_actual_takeover_uses_observed_replay_pose(self):
        self.meta['actual_cup_position_at_recording_m']=[.401,-.102,.0001]
        data=self.accept()
        self.assertEqual(data['correction']['actual_takeover_cup_position_m'],[.401,-.102,.0001])
        self.assertEqual(data['correction']['source_takeover_cup_expected_position_m'],[.4,-.1,0])
    def test_successful_source_cannot_be_saved_as_failure_recovery(self):
        self.ctx['candidate']['source_final_box_success']=True
        with self.assertRaisesRegex(ValueError,'source'):self.accept()
    def test_private_rejection_does_not_call_base_save_or_erase_evidence(self):
        from recorder import make_recorder_class
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp);episode=folder/'.recording_000000';episode.mkdir()
            (episode/'frames.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in self.rows))
            (folder/'evidence_a.json').write_text(json.dumps(self.meta))
            (folder/'current.json').write_text(json.dumps(self.ctx))
            owner=self
            class Base:
                def __init__(self):
                    self._lock=threading.RLock();self._recording=True;self._sync_failure='';self.minimum_frames=10
                    self._frames_file=(episode/'frames.jsonl').open('a');self._episode_dir=episode
                    self._task_status=copy.deepcopy(owner.final);self._episode_evidence=owner.evidence
                    self.base_saves=0
                def _save_callback(self,req,res):self.base_saves+=1;res.success=True;return res
            geometry=SimpleNamespace(link_fk=lambda joints:([0,0,0],None))
            Recorder=make_recorder_class(Base,geometry=geometry,roll_fn=lambda rotation:-90.)
            recorder=Recorder();recorder.ctx=self.ctx;recorder.takeover_status={'object_position_m':[.4,-.1,0]}
            recorder.dynamic_samples=150;recorder.frame_dynamic_samples=50
            recorder._task_status['task_success']=False
            response=SimpleNamespace(success=None,message='')
            with patch.dict(os.environ,{'Z1_PUSH_RECOVERY_CURRENT':str(folder/'current.json')}):
                result=recorder._save_callback(None,response)
            self.assertFalse(result.success);self.assertEqual(recorder.base_saves,0)
            self.assertTrue((folder/'evidence_a.json').exists());self.assertTrue((folder/'evidence_a.rejection.json').exists())
            self.assertFalse((episode/'correction.json').exists())
            self.assertTrue(recorder._recording)
            recorder._frames_file.close()
    def test_status_callback_counts_exact_fresh_payload_and_contact(self):
        from recorder import make_recorder_class
        class Base:
            def __init__(self):self._lock=threading.RLock();self._recording=True;self._task_status={}
            def _task_status_callback(self,msg):
                try:self._task_status=json.loads(msg.data)
                except ValueError:pass
            def get_clock(self):return SimpleNamespace(now=lambda:SimpleNamespace(nanoseconds=100000000000))
        Recorder=make_recorder_class(Base,geometry=object(),roll_fn=lambda _:0)
        recorder=Recorder()
        recorder._task_status_callback(SimpleNamespace(data=json.dumps({'frozen':False,'target_collisions_enabled':True,'left_finger_touching_table':True})))
        recorder._task_status_callback(SimpleNamespace(data='malformed'))
        self.assertEqual(recorder.dynamic_samples,1)
        self.assertTrue(recorder.contact)
        self.assertEqual(len(recorder.dynamic_violations),1)
if __name__=='__main__':unittest.main(verbosity=2)
