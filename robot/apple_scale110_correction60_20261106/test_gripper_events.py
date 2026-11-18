import ast,json,tempfile,threading,unittest
from pathlib import Path
from types import SimpleNamespace as N
from core import atomic_json

class Events(unittest.TestCase):
 def test_first_close_event_is_preserved_and_wrong_attempt_is_ignored(self):
  tree=ast.parse((Path(__file__).parent/'nodes.py').read_text())
  method=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='_record_gripper_event')
  with tempfile.TemporaryDirectory() as td:
   path=Path(td)/'evidence.json';atomic_json(path,{'attempt_id':'one'})
   current={'attempt_id':'one'}
   env=dict(json=json,context=lambda:current,evidence_path=lambda _:path,atomic_json=atomic_json)
   exec(compile(ast.fix_missing_locations(ast.Module(body=[method],type_ignores=[])),'nodes.py','exec'),env)
   clock=[1000000000]
   node=N(ctx={'attempt_id':'one'},_lock=threading.RLock(),get_clock=lambda:N(now=lambda:N(nanoseconds=clock[0])))
   event=env['_record_gripper_event']
   event(node,'open',-.9);clock[0]=2000000000;event(node,'close',0.)
   clock[0]=3000000000;event(node,'close',-.4)
   data=json.loads(path.read_text());self.assertEqual(data['close_request_sim_time'],2.)
   self.assertEqual([x['state'] for x in data['gripper_command_events']],['open','close','close'])
   current['attempt_id']='two';event(node,'open',-.9)
   self.assertEqual(json.loads(path.read_text()),data)
if __name__=='__main__':unittest.main()
