import json,time,math
from pathlib import Path
from collections import Counter
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, QoSProfile, ReliabilityPolicy, DurabilityPolicy
from sensor_msgs.msg import Image,JointState
from rosgraph_msgs.msg import Clock
from std_msgs.msg import String
from z1_isaac_bridge.recording_sync import RecordingSynchronizer
from z1_isaac_bridge.episode_recorder import JOINT_NAMES
OUT=Path(__file__).resolve().parent
rclpy.init()
n=Node('read_only_time_alignment_monitor')
f=(OUT/'messages.jsonl').open('w');pairs=(OUT/'shadow_pairs.jsonl').open('w')
sync=RecordingSynchronizer(('wrist','global','global_right'),tolerance=.045,state_gap=.1,action_age=.4)
clock=[None]; start=time.monotonic(); next_target=[None]; counts=Counter();failure=Counter();subs=[]
def stamp(t):return t.sec+t.nanosec*1e-9
def write(data): f.write(json.dumps(data,ensure_ascii=False)+'\n')
def onclock(m):
    t=stamp(m.clock)
    if clock[0] is not None and t<clock[0]:sync.reset();next_target[0]=None
    clock[0]=t
subs.append(n.create_subscription(Clock,'/clock',onclock,qos_profile_sensor_data))
def sensor(name):
    def cb(m):
        t=stamp(m.header.stamp);counts[name]+=1
        if name in sync.cameras:payload=None
        else:
            values=dict(zip(m.name,m.position))
            if not all(j in values and math.isfinite(values[j]) for j in JOINT_NAMES):return
            payload=[values[j] for j in JOINT_NAMES]
        sync.push(name,t,payload)
        write(dict(kind=name,stamp=t,clock=clock[0],age=None if clock[0] is None else clock[0]-t,wall_elapsed=time.monotonic()-start))
    return cb
for name,topic in [('wrist','/left_wrist_camera/color/image_raw'),('global','/global_camera/color/image_raw'),('global_right','/global_right_camera/color/image_raw'),('state','/joint_states'),('action','/isaac_joint_commands')]:
    subs.append(n.create_subscription(Image if name in sync.cameras else JointState,topic,sensor(name),qos_profile_sensor_data))
def status(m):
    try:obj=json.loads(m.data)
    except ValueError:return
    if clock[0] is not None:sync.push('status',clock[0],obj)
def recording(m):
    try:obj=json.loads(m.data)
    except ValueError:return
    write(dict(kind='recording_status',clock=clock[0],wall_elapsed=time.monotonic()-start,data=obj))
subs.append(n.create_subscription(String,'/smolvla/status',status,10))
q=QoSProfile(depth=10,reliability=ReliabilityPolicy.RELIABLE,durability=DurabilityPolicy.TRANSIENT_LOCAL)
subs.append(n.create_subscription(String,'/smolvla/recording_status',recording,q))
while time.monotonic()-start<65:
    rclpy.spin_once(n,timeout_sec=.02)
    now=clock[0]
    if now is None or time.monotonic()-start<2:continue
    if next_target[0] is None:next_target[0]=math.floor((now-.5)*10)/10
    if now<next_target[0]+.5:continue
    target=next_target[0];next_target[0]+=.1
    try:
        sample=sync.select(target);sync.commit(sample)
        image_stamps={k:v[0] for k,v in sample['images'].items()}
        pairs.write(json.dumps(dict(target=target,clock=now,images=image_stamps,state_bracket=sample['state_bracket'],action_stamp=sample['action'][0],action_age=target-sample['action'][0],spread=max([target,*image_stamps.values()])-min([target,*image_stamps.values()])))+'\n')
        counts['paired']+=1
    except ValueError as e:failure[str(e).split(':')[0]]+=1
f.close();pairs.close()
(OUT/'counts.json').write_text(json.dumps(dict(counts=counts,shadow_pair_failures=failure),indent=2))
print(json.dumps(dict(counts=counts,shadow_pair_failures=failure)))
n.destroy_node();rclpy.shutdown()
