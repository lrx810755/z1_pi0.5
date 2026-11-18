import ast
from pathlib import Path
from types import SimpleNamespace

from builtin_interfaces.msg import Time
from sensor_msgs.msg import JointState
from rclpy.parameter import Parameter


SCRIPT = Path(__file__).resolve().parents[2] / 'isaac_assets/z1/scripts/run_z1_ros2_scene.py'


def scene_method(name):
    tree = ast.parse(SCRIPT.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
               and n.name == 'SceneResetInterface')
    return next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)


def test_scene_reset_node_uses_simulation_clock():
    init = scene_method('__init__')
    assignment = next(n for n in init.body if isinstance(n, ast.Assign)
                      and any(isinstance(t, ast.Attribute) and t.attr == '_node'
                              for t in n.targets))
    obj = SimpleNamespace(_context=object())
    fake_rclpy = SimpleNamespace(create_node=lambda name, **kwargs: kwargs)
    exec(compile(ast.Module(body=[assignment], type_ignores=[]), 'reset_clock', 'exec'),
         {'self': obj, 'rclpy': fake_rclpy, 'Parameter': Parameter})
    overrides = {p.name: p.value for p in obj._node.get('parameter_overrides', [])}
    assert overrides.get('use_sim_time') is True


def test_home_commands_include_the_simulation_timestamp():
    method = scene_method('_reset_callback')
    statements = next(n.body for n in method.body if isinstance(n, ast.Try))
    start = next(i for i, n in enumerate(statements) if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == 'home_command' for t in n.targets))
    end = next(i for i in range(start, len(statements)) if isinstance(statements[i], ast.For))
    sent = []
    stamp = Time(sec=123, nanosec=456_000_000)
    obj = SimpleNamespace(_JointState=JointState, _robot=SimpleNamespace(dof_names=['joint1']),
                          _joint_command_pub=SimpleNamespace(publish=sent.append),
                          _node=SimpleNamespace(get_clock=lambda: SimpleNamespace(
                              now=lambda: SimpleNamespace(to_msg=lambda: stamp))))
    exec(compile(ast.Module(body=statements[start:end + 1], type_ignores=[]), 'home_stamp', 'exec'),
         {'self': obj, 'home_positions': [0.2]})
    assert len(sent) == 3
    assert all(m.header.stamp == stamp and list(m.position) == [0.2] for m in sent)
