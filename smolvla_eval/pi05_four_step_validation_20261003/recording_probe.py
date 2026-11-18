import argparse,json,threading,time
from pathlib import Path
import rclpy
from rclpy.executors import MultiThreadedExecutor
from std_srvs.srv import Trigger
from replay import Replay
from z1_isaac_bridge.smolvla_sim_rollout import parse_args
ROOT=Path(__file__).resolve().parent
DATA=Path('/home/lp531/smolvla_data/z1_bottle_collision_consistent_supplement_20261003_raw_v1')
def main():
 cli=argparse.ArgumentParser();cli.add_argument('--label',required=True);label=cli.parse_args().label
 args=parse_args(['--checkpoint',str(DATA),'--objects','beverage_bottle','--wrist-rgb-topic','/left_wrist_camera/color/image_raw']);args.training_manifest=None
 rclpy.init();node=Replay(args);executor=MultiThreadedExecutor(num_threads=4);executor.add_node(node);thread=threading.Thread(target=executor.spin,daemon=True);thread.start();started=False
 try:
  node._call_trigger(node._reset_episode,120);node._activate_policy('beverage_bottle')
  _,joints,_,_=node._snapshot();node._publish_command(joints)
  node._call_trigger(node.create_client(Trigger,'/smolvla/recording/start'),10);started=True
  deadline=time.monotonic()+4
  while time.monotonic()<deadline:node._publish_command(joints);time.sleep(.08)
  folders=list(DATA.glob('.recording_*'));frames=sum(len((p/'frames.jsonl').read_text().splitlines()) for p in folders)
  result={'label':label,'frames':frames,'diagnostic_only':True,'dataset_saved':False};(ROOT/f'recording_probe_{label}.json').write_text(json.dumps(result,indent=2));print(result,flush=True)
 finally:
  if started:node._call_trigger(node.create_client(Trigger,'/smolvla/recording/discard'),20)
  node._set_active(False);executor.shutdown();thread.join(timeout=2);node.destroy_node();rclpy.shutdown()
if __name__=='__main__':main()
