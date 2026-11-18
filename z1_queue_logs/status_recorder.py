import json, time
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from std_msgs.msg import String

class Rec(Node):
    def __init__(self):
        super().__init__('z1_status_recorder')
        qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                         reliability=ReliabilityPolicy.RELIABLE)
        self.create_subscription(String, '/smolvla/pregrasp_status', self.cb, qos)
    def cb(self, msg):
        try:
            d = json.loads(msg.data)
        except Exception:
            return
        d['_t'] = time.strftime('%F %T')
        print(json.dumps(d, ensure_ascii=False), flush=True)

rclpy.init()
node = Rec()
try:
    rclpy.spin(node)
except KeyboardInterrupt:
    pass
