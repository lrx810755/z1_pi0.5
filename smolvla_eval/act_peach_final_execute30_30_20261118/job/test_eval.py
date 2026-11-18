import importlib.util,sys,unittest
from pathlib import Path
import numpy as np
p=Path(__file__).parent/'runtime_rollout.py'
spec=importlib.util.spec_from_file_location('z1_isaac_bridge.peach_act_runtime',p)
runtime=importlib.util.module_from_spec(spec);sys.modules[spec.name]=runtime;spec.loader.exec_module(runtime)
class Contract(unittest.TestCase):
 def test_peach_act_parser_and_thirty_step_execution(self):
  args=runtime.parse_args(['--policy-type','act','--checkpoint','/tmp/unused','--objects','peach','--target-xy-bounds','.3','.7','-.3','.1','--execute-steps','30','--episodes-per-object','30','--rate','10','--execution-rate','60','--binary-gripper','--gripper-action-encoding','binary'])
  self.assertEqual(args.objects,['peach']);self.assertEqual(args.episodes_per_object,30)
  full=np.arange(210).reshape(1,30,7)
  np.testing.assert_array_equal(runtime.execution_window(full,args.execute_steps),full[0,:30])
if __name__=='__main__':unittest.main()
