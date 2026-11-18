import xml.etree.ElementTree as ET

import numpy as np

from .config import (
    ARM_JOINTS,
    MAX_SAFE_SPEED_RAD_S,
    TELEOP_KD,
    TELEOP_KP,
    TELEOP_MAX_EFFORT,
)


def load_urdf_limits(urdf_path):
    root = ET.parse(urdf_path).getroot()
    limits = {}
    for joint in root.findall("joint"):
        name = joint.attrib.get("name")
        limit = joint.find("limit")
        if name is None or limit is None:
            continue
        limits[name] = {
            "lower": float(limit.attrib["lower"]),
            "upper": float(limit.attrib["upper"]),
            "velocity": float(limit.attrib.get("velocity", MAX_SAFE_SPEED_RAD_S)),
            "effort": float(limit.attrib.get("effort", 0.0)),
        }
    return limits

def clamp(value, lower, upper):
    return float(min(max(value, lower), upper))



def configure_drive_gains(robot):
    dof_names = list(robot.dof_names)
    controller = robot.get_articulation_controller()
    kps, kds = controller.get_gains()
    max_efforts = np.asarray(controller.get_max_efforts(), dtype=np.float64)
    for idx, name in enumerate(dof_names):
        if name not in ARM_JOINTS:
            continue
        kps[idx] = TELEOP_KP.get(name, kps[idx])
        kds[idx] = TELEOP_KD.get(name, kds[idx])
        max_efforts[idx] = TELEOP_MAX_EFFORT.get(name, max_efforts[idx])
    controller.set_gains(kps=kps, kds=kds)
    controller.set_max_efforts(max_efforts)
    print("Configured teleop drive gains and max efforts:")
    for idx, name in enumerate(dof_names):
        if name in ARM_JOINTS:
            print(f"  {name}: kp={kps[idx]:.1f} kd={kds[idx]:.1f} max_effort={max_efforts[idx]:.1f}")

