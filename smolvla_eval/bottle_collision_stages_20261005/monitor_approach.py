"""Read-only capture of bottle/arm motion and planner events; sends no commands."""
import json
import time
import sys
from pathlib import Path
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, QoSProfile, DurabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from tf2_msgs.msg import TFMessage
import tf2_ros
from z1_isaac_bridge.grasp_evaluator import canonical_object_name


def main():
    output = Path(sys.argv[1]); output.mkdir(parents=True, exist_ok=True)
    duration = float(sys.argv[2]) if len(sys.argv) > 2 else 75
    rclpy.init(); node = Node('bottle_approach_readonly_monitor')
    buffer = tf2_ros.Buffer(); listener = tf2_ros.TransformListener(buffer, node)
    state = {'object': None, 'joints': {}, 'command': {}, 'task': {}, 'assistant': {}, 'scene': {}}
    events = (output/'events.jsonl').open('w'); samples = (output/'samples.jsonl').open('w')

    def stamp(s): return s.sec + s.nanosec/1e9
    def emit(kind, data):
        events.write(json.dumps({'wall': time.time(), 'kind':kind, 'data':data}, ensure_ascii=False)+'\n');events.flush()
    def status(message, key):
        try: data=json.loads(message.data)
        except ValueError:return
        old=state[key];state[key]=data
        if key=='assistant' and data != old:
            emit(key,data)
            if data.get('phase') in ('PLAN_READY','ERROR','FAILED'):
                print(data.get('phase'), data.get('message',''), flush=True)
    def joints(message, key):
        state[key]=dict(zip(message.name,message.position))
        state[key+'_stamp']=stamp(message.header.stamp)
    def truth(message):
        for t in message.transforms:
            if canonical_object_name(t.child_frame_id) == 'beverage_bottle':
                p=t.transform.translation;q=t.transform.rotation
                state['object']={'position':[p.x,p.y,p.z], 'orientation':[q.x,q.y,q.z,q.w], 'stamp':stamp(t.header.stamp)}
    latched=QoSProfile(depth=10,durability=DurabilityPolicy.TRANSIENT_LOCAL)
    node.create_subscription(String,'/smolvla/pregrasp_status',lambda m:status(m,'assistant'),latched)
    node.create_subscription(String,'/smolvla/status',lambda m:status(m,'task'),10)
    node.create_subscription(String,'/smolvla/scene_grasp_status',lambda m:status(m,'scene'),latched)
    node.create_subscription(TFMessage,'/isaac/object_transforms',truth,qos_profile_sensor_data)
    node.create_subscription(JointState,'/joint_states',lambda m:joints(m,'joints'),qos_profile_sensor_data)
    node.create_subscription(JointState,'/isaac_joint_commands',lambda m:joints(m,'command'),qos_profile_sensor_data)
    deadline=time.monotonic()+duration;next_sample=0;count=0
    try:
        while time.monotonic()<deadline:
            rclpy.spin_once(node,timeout_sec=0.02)
            if time.monotonic()<next_sample:continue
            next_sample=time.monotonic()+.05
            row={'wall':time.time(),**state}
            for target,source,label in [('world','link06','ee_world'),('world','gripperStator','stator_world'),('link00','link06','ee_base')]:
                try:
                    t=buffer.lookup_transform(target,source,Time());p=t.transform.translation;q=t.transform.rotation
                    row[label]={'position':[p.x,p.y,p.z],'orientation':[q.x,q.y,q.z,q.w],'stamp':stamp(t.header.stamp)}
                except tf2_ros.TransformException:pass
            samples.write(json.dumps(row,ensure_ascii=False)+'\n');count+=1
        print('Captured samples:',count,flush=True)
    finally:
        events.close();samples.close();node.destroy_node();rclpy.shutdown()


if __name__=='__main__':main()
