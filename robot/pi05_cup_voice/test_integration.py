import importlib.util,json,sys,tempfile,threading,unittest
from pathlib import Path
from queue import Queue
from types import SimpleNamespace
from unittest.mock import Mock,patch
R=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('voice_entry',R/'entry.py');entry=importlib.util.module_from_spec(spec);spec.loader.exec_module(entry)
b=entry.backend
from z1_isaac_bridge.smolvla_voice_core import VoiceSession,VoiceCommand
class Tests(unittest.TestCase):
 def bare(self):
  n=object.__new__(b.SmolVLAVoiceBackend);n._session=VoiceSession();n._session_lock=threading.Lock();n._commands=Queue();n._stop_event=threading.Event();n._publish_status=Mock();n._set_active=Mock();return n
 def test_model_and_timing(self):
  args=entry.runtime.parse_args(json.loads((R/'backend_args.json').read_text()))
  self.assertEqual(args.policy_type,'pi05');self.assertEqual(args.objects,['cup']);self.assertEqual(args.rate,10);self.assertEqual(args.execute_steps,15);self.assertEqual(args.execution_rate,60)
  cfg=json.loads((args.checkpoint/'config.json').read_text());self.assertEqual(cfg['chunk_size'],30)
 def test_unsupported_target_never_queued(self):
  n=self.bare();n._command_callback(SimpleNamespace(data='{"command":"grasp","target":"apple"}'));self.assertTrue(n._commands.empty());self.assertEqual(n._session.state,'IDLE');self.assertIn('error',n._publish_status.call_args.kwargs)
 def test_stop_is_immediate(self):
  n=self.bare();n._session.state='EXECUTING';n._command_callback(SimpleNamespace(data='{"command":"stop"}'));self.assertTrue(n._stop_event.is_set());self.assertTrue(n._commands.empty())
 def test_start_grasp_result_chain(self):
  n=self.bare();n._commands.put(VoiceCommand('start'));n._commands.put(VoiceCommand('grasp','cup'));n._randomize_scene=Mock(return_value=12345);n.run_episode=Mock(return_value=dict(success=False,failure_category='finger_table_contact',target_object='cup'))
  with tempfile.TemporaryDirectory() as td:
   n.args=SimpleNamespace(output=Path(td))
   with patch.object(b.rclpy,'ok',side_effect=[True,True,False]):root=n.run_voice_loop()
   n._randomize_scene.assert_called_once_with('cup');n.run_episode.assert_called_once_with('cup',1,root,prepared_seed=12345,stop_event=n._stop_event);self.assertEqual(n._session.state,'IDLE')
 def test_stop_between_accept_and_rollout_is_preserved(self):
  n=self.bare();n._commands.put(VoiceCommand('start'));n._commands.put(VoiceCommand('grasp','cup'));n._randomize_scene=Mock(return_value=123)
  accept=n._session.accept
  def interleaved(command):
   result=accept(command)
   if command.command=='grasp':n._stop_event.set()
   return result
  n._session.accept=interleaved
  def run(*args,**kwargs):
   self.assertTrue(kwargs['stop_event'].is_set())
   return dict(success=False,failure_category='operator_stop',target_object='cup')
  n.run_episode=run
  with tempfile.TemporaryDirectory() as td:
   n.args=SimpleNamespace(output=Path(td))
   with patch.object(b.rclpy,'ok',side_effect=[True,True,False]):n.run_voice_loop()
 def test_text_intent_to_ros_protocol(self):
  sys.path.insert(0,'/home/lp531/robot')
  from asr import VoiceTranslator,RuntimeConfig
  spec=importlib.util.spec_from_file_location('cup_frontend',R/'frontend.py');frontend=importlib.util.module_from_spec(spec);spec.loader.exec_module(frontend)
  parse_voice_command=frontend.parse_voice_command
  translator=VoiceTranslator(RuntimeConfig(no_ollama=True),load_asr=False)
  self.assertEqual(parse_voice_command(translator,'小白抓取杯子'),('grasp','cup'))
  self.assertIsNone(parse_voice_command(translator,'小白抓取苹果'))
if __name__=='__main__':unittest.main()
