"""Verify the prototype boundary without importing or contacting robot software."""
import ast
import subprocess
import sys
from pathlib import Path
import unittest

from experimental.legibility_optimizer import robot_interfaces as robot

ROOT = Path(__file__).resolve().parents[2]


class NoExecutionTests(unittest.TestCase):
    def test_all_missing_robot_operations_fail_explicitly(self):
        calls = [lambda: robot.fk_position([0], ["joint"]),
                 lambda: robot.retime_trajectory(None, robot_state=None, planning_group=None),
                 lambda: robot.validate_retimed_trajectory(None, baseline=None, planning_scene=None),
                 lambda: robot.execute_validated_trajectory(
                     None, validation=True, explicit_authorization=True)]
        for call in calls:
            with self.subTest(call=call), self.assertRaises(robot.MissingRobotIntegration):
                call()

    def test_no_hardware_imports_or_command_calls_in_prototype(self):
        forbidden_modules = {"rospy", "rclpy", "actionlib", "moveit_commander", "control_msgs",
                             "socket", "subprocess", "requests"}
        forbidden_calls = {"Publisher", "ServiceProxy", "SimpleActionClient", "send_goal",
                           "publish", "execute", "go", "system", "Popen"}
        for path in (ROOT / "experimental").rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if isinstance(node, ast.Import):
                    self.assertFalse({x.name.split(".")[0] for x in node.names} & forbidden_modules)
                elif isinstance(node, ast.ImportFrom):
                    self.assertNotIn((node.module or "").split(".")[0], forbidden_modules)
                elif isinstance(node, ast.Call):
                    name = getattr(node.func, "attr", getattr(node.func, "id", ""))
                    self.assertNotIn(name, forbidden_calls, str(path))

    def test_offline_demo_runs_with_robot_imports_and_network_blocked(self):
        script = """
import builtins, tempfile
from unittest.mock import patch
import numpy, scipy.optimize
original_import = builtins.__import__
blocked = {'rospy', 'rclpy', 'actionlib', 'moveit_commander', 'moveit_msgs',
           'trajectory_msgs', 'control_msgs'}
def guarded_import(name, *args, **kwargs):
    if name.split('.')[0] in blocked:
        raise AssertionError('Unexpected robot import: ' + name)
    return original_import(name, *args, **kwargs)
with patch('builtins.__import__', side_effect=guarded_import):
    from experimental.legibility_optimizer.demo import main
    from experimental.legibility_optimizer import ros1_adapter, robot_interfaces
    with tempfile.TemporaryDirectory() as directory:
        with patch('socket.socket', side_effect=AssertionError('Unexpected network access')):
            assert main(['--output', directory]) == 0
"""
        run = subprocess.run([sys.executable, "-c", script], cwd=ROOT,
                             capture_output=True, text=True, timeout=60)
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
        self.assertIn('"execution_ready": false', run.stdout)


if __name__ == "__main__":
    unittest.main()
