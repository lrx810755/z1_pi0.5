import ast,threading,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

class StartTests(unittest.TestCase):
 def wrapper(self,status,advance=None):
  tree=ast.parse(Path(__file__).with_name('basic_nodes.py').read_text());method=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='start_recording')
  class Base:
   def start_recording(self):return {'success':True}
  timer=SimpleNamespace(monotonic=Mock(side_effect=[0.,.1,.2,11.]),sleep=Mock(side_effect=advance))
  code=ast.Module(body=[ast.ClassDef(name='Wrapper',bases=[ast.Name(id='Base',ctx=ast.Load())],keywords=[],body=[method],decorator_list=[])],type_ignores=[])
  ns=dict(Base=Base,time=timer,context=lambda:dict(task=dict(seed=123)));exec(compile(ast.fix_missing_locations(code),'<actual recorder start>','exec'),ns)
  obj=ns['Wrapper']();obj._lock=threading.RLock();obj._recording_status=status;obj._force_stop_event=threading.Event();return obj,timer
 def test_waits_for_three_frames_before_movement(self):
  state=dict(recording=True,frame_count=0,randomization_seed=123)
  obj,timer=self.wrapper(state,lambda _:state.update(frame_count=3))
  self.assertTrue(obj.start_recording()['success']);self.assertEqual(timer.sleep.call_count,1)
 def test_sync_failure_rejected_immediately(self):
  obj,timer=self.wrapper(dict(sync_failure='missing right camera'))
  with self.assertRaisesRegex(RuntimeError,'missing right camera'):obj.start_recording()
  timer.sleep.assert_not_called()
 def test_stale_seed_cannot_satisfy_start(self):
  obj,timer=self.wrapper(dict(recording=True,frame_count=30,randomization_seed=999))
  with self.assertRaisesRegex(RuntimeError,'three synchronized frames'):obj.start_recording()
if __name__=='__main__':unittest.main()
