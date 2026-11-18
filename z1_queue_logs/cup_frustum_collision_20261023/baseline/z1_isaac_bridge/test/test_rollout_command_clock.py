import ast,threading
from pathlib import Path
from types import SimpleNamespace

def test_policy_command_uses_observed_simulation_clock_for_recording():
    path=Path(__file__).parents[1]/'z1_isaac_bridge'/'smolvla_sim_rollout.py';tree=ast.parse(path.read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='SmolVLASimRollout');method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_publish_command')
    sent=[]
    namespace={'JointState':lambda:SimpleNamespace(header=SimpleNamespace(stamp=SimpleNamespace(sec=0,nanosec=0))),'JOINT_NAMES':('joint1','joint2','joint3','joint4','joint5','joint6','jointGripper')}
    exec(compile(ast.Module(body=[method],type_ignores=[]),str(path),'exec'),namespace)
    node=SimpleNamespace(_lock=threading.Lock(),_joint_sim_stamp=123.456,get_clock=lambda:SimpleNamespace(now=lambda:SimpleNamespace(to_msg=lambda:SimpleNamespace(sec=1791000000,nanosec=0))),_command_pub=SimpleNamespace(publish=sent.append))
    namespace['_publish_command'](node,[0.]*7)
    assert sent[0].header.stamp.sec==123
    assert sent[0].header.stamp.nanosec==456000000
