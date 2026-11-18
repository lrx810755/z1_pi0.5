import ast
import unittest
from pathlib import Path


LAUNCH_FILE = (
    Path(__file__).parents[1] / "launch" / "red_ball_real_grasp.launch.py"
)


class RedBallRealGraspLaunchTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = LAUNCH_FILE.read_text(encoding="utf-8")
        cls.tree = ast.parse(cls.source)

    def test_launch_starts_gui_as_the_only_task_interface(self):
        self.assertIn('executable="red_target_gui.py"', self.source)
        self.assertNotIn("red_ball_camera_moveit.py", self.source)
        self.assertNotIn("rqt_image_view", self.source)

    def test_launch_has_no_terminal_task_controls(self):
        declared_arguments = [
            node
            for node in ast.walk(self.tree)
            if isinstance(node, ast.Call)
            and getattr(node.func, "id", None) == "DeclareLaunchArgument"
        ]
        self.assertEqual(declared_arguments, [])
        for legacy_control in (
            '"execute"',
            '"no_preview"',
            '"execution_mode"',
            '"stop_distance"',
            '"speed_scale"',
        ):
            self.assertNotIn(legacy_control, self.source)


if __name__ == "__main__":
    unittest.main()
