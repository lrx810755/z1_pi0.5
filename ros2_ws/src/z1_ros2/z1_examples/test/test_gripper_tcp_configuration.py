import xml.etree.ElementTree as ET
from pathlib import Path


REPOSITORY = Path(__file__).parents[2]
URDF = REPOSITORY / "z1_description" / "urdf" / "z1.urdf.xacro"
SRDF = REPOSITORY / "z1_moveit" / "config" / "z1_description.srdf"
GUI = Path(__file__).parents[1] / "z1_examples" / "red_target_gui.py"


def test_gripper_tcp_is_fixed_to_stator_at_measured_opening_center():
    root = ET.parse(URDF).getroot()
    joint = root.find(".//joint[@name='gripper_tcp_joint']")
    assert joint is not None
    assert joint.attrib["type"] == "fixed"
    assert joint.find("parent").attrib["link"] == "gripperStator"
    assert joint.find("child").attrib["link"] == "gripper_tcp"
    assert joint.find("origin").attrib["xyz"] == "0.095 0.0 0.021"


def test_moveit_uses_stator_approach_while_open_has_correct_semantics():
    root = ET.parse(SRDF).getroot()
    arm = root.find(".//group[@name='z1_arm']/chain")
    assert arm.attrib["tip_link"] == "link06"
    open_state = root.find(".//group_state[@name='open']/joint[@name='jointGripper']")
    closed_state = root.find(".//group_state[@name='closed']/joint[@name='jointGripper']")
    assert float(open_state.attrib["value"]) == -0.90
    assert float(closed_state.attrib["value"]) == 0.0
    source = GUI.read_text(encoding="utf-8")
    assert 'end_effector_link="gripperStator"' in source
    assert "self.alignment_mode" not in source
    assert "tcp_approach_distance" not in source
    assert "gripper_target_deg" not in source
