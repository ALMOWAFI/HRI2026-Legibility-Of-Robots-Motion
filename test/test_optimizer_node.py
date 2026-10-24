"""Offline optimizer topic callback/client tests; no MoveIt or controller is imported."""
import copy
import queue
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import numpy as np

from test_reach_topic import Duration, load_source, request, topic_fixture
from hri_legible_motion import safety
import hri_legible_motion


class OptimizerNodeTests(unittest.TestCase):
    def setUp(self):
        self.topic, modules = topic_fixture()
        self.ros = modules["rospy"]
        modules["hri_legible_motion.reach_topic"] = self.topic
        robot = ModuleType("hri_legible_motion.robot")
        robot.Robot, robot.end_joints = Mock(), Mock()
        modules["hri_legible_motion.robot"] = robot
        self.robot_constructor = robot.Robot
        with patch.object(hri_legible_motion, "reach_topic", self.topic, create=True):
            self.node_module = load_source("offline_optimizer_node", "scripts/legibility_optimizer_node.py", modules)
            self.client_module = load_source("offline_motion_tool", "tools/test_motion.py", modules)
        self.cfg = dict(frame_id="base", motion=dict(ee_link="tool", touch_z=.1, hover_height=0.),
                        safety=dict(workspace_min=[0., -1., 0.], workspace_max=[1., 1., 1.]),
                        blocks=[dict(id=23, name="B1", position=[.5, .2, 0.])], optimizer=dict(timeout=.01))
        self.node = self.node_module.OptimizerNode(self.cfg)
        self.msg = request(self.topic)
        times, xyz = self.topic.from_msg(self.msg)
        self.out = NS(times=times, xyz=xyz, success=True, message="ok", baseline_score=.30,
                      optimized_score=.33, length_ratio=1.02, max_lateral=.1,
                      numerical_success=True, geometric_valid=True, diagnostics={})
        self.solve = Mock(return_value=self.out)
        self.node_module.optimize_planar_reach = self.solve

    def test_success_publishes_only_a_cartesian_proposal_with_honest_score_gain(self):
        self.node.on_reach(self.msg)
        self.solve.assert_called_once()
        reply = self.node.pub.publish.call_args.args[0]
        self.topic.validate_response(self.msg, reply)
        self.robot_constructor.assert_not_called()
        self.ros.Publisher.assert_called_once_with(self.topic.LEGIBLE,
                                                  self.node_module.CartesianTrajectory, queue_size=10)
        logs = " ".join(call.args[0] % call.args[1:] for call in self.ros.loginfo.call_args_list)
        self.assertIn("+3.00 percentage points", logs)
        self.assertIn("10.00 cm", logs)
        self.assertIn("physical_validity=not_checked", logs)

    def test_bad_input_never_calls_optimizer_and_gets_empty_reply(self):
        changes = [lambda msg: setattr(msg.header, "frame_id", "world"),
                   lambda msg: setattr(msg, "tracked_frame", "other"),
                   lambda msg: setattr(msg, "points", []),
                   lambda msg: setattr(msg.points[1].point.pose.position, "x", np.nan),
                   lambda msg: setattr(msg.points[1], "time_from_start", Duration(0.)),
                   lambda msg: setattr(msg.points[1].point.pose.orientation, "w", 0.)]
        for change in changes:
            msg = copy.deepcopy(self.msg)
            change(msg)
            self.node.on_reach(msg)
            self.assertEqual(self.node.pub.publish.call_args.args[0].points, [])
        self.solve.assert_not_called()

    def test_failed_solver_or_invalid_result_is_not_published_as_success(self):
        self.out.success = False
        self.node.on_reach(self.msg)
        self.assertEqual(self.node.pub.publish.call_args.args[0].points, [])
        self.out.success = True
        self.out.times[1] = .6
        self.node.on_reach(self.msg)
        self.assertEqual(self.node.pub.publish.call_args.args[0].points, [])

    def test_solver_validation_error_gets_failure_reply(self):
        self.solve.side_effect = ValueError("invalid model input")
        self.node.on_reach(self.msg)
        self.assertEqual(self.node.pub.publish.call_args.args[0].points, [])

    def test_inconsistent_acceptance_flags_get_failure_reply(self):
        for field in ("numerical_success", "geometric_valid"):
            with self.subTest(field=field):
                setattr(self.out, field, False)
                self.node.on_reach(self.msg)
                self.assertEqual(self.node.pub.publish.call_args.args[0].points, [])
                setattr(self.out, field, True)

    def test_displayed_scores_use_configured_beta_time_samples_and_goal_row(self):
        cfg = copy.deepcopy(self.cfg)
        cfg["blocks"][0]["color"] = "green"
        cfg["motion"].update(start_joints={"joint": 0.}, duration=1.)
        cfg["optimizer"]["beta"] = 7.
        robot = Mock()
        robot.fk.return_value = (np.array([.3, 0., .1]), NS(x=0., y=0., z=0., w=1.))
        robot.plan.return_value = NS(points=[NS(velocities=[0.])])
        optimizer = Mock()
        optimizer.send.side_effect = lambda times, xyz, orientation: (times, xyz)
        with patch.object(self.client_module, "planar_legibility", return_value=.3) as score, patch("builtins.print"):
            self.client_module.plan_reach(robot, cfg, cfg["blocks"][0], optimizer)
        self.assertEqual(score.call_count, 2)
        for call in score.call_args_list:
            self.assertEqual(call.args[2], 0)  # configuration row, not the block id 23
            self.assertEqual(call.kwargs["beta"], 7.)
            np.testing.assert_array_equal(call.kwargs["times"], optimizer.send.call_args.args[0])

    def client(self, answer):
        client = self.client_module.Optimizer.__new__(self.client_module.Optimizer)
        client.cfg, client.legible, client.replies = self.cfg, True, queue.Queue()
        client.pub = Mock()
        def publish(msg):
            response = copy.deepcopy(msg)
            answer(response)
            client.replies.put(response)
        client.pub.publish.side_effect = publish
        return client

    def send(self, client):
        return client.send(*self.topic.from_msg(self.msg), self.msg.points[0].point.pose.orientation)

    def test_client_rejects_frame_and_timestamp_and_endpoint_changes(self):
        changes = [lambda msg: setattr(msg.header, "frame_id", "other"),
                   lambda msg: setattr(msg.points[1], "time_from_start", Duration(.6)),
                   lambda msg: msg.points.reverse(),
                   lambda msg: setattr(msg.points[-1].point.pose.position, "x", .6),
                   lambda msg: setattr(msg, "points", [])]
        for change in changes:
            with self.subTest(change=change), self.assertRaises(safety.SafetyError):
                self.send(self.client(change))

    def test_client_accepts_valid_result(self):
        times, xyz = self.send(self.client(lambda msg: None))
        expected_t, expected_xyz = self.topic.from_msg(self.msg)
        np.testing.assert_array_equal(times, expected_t)
        np.testing.assert_array_equal(xyz, expected_xyz)

    def test_timeout_uses_wall_clock_when_ros_time_does_not_advance(self):
        client = self.client(lambda msg: setattr(msg.header, "stamp", -1))
        with self.assertRaisesRegex(safety.SafetyError, "No answer"):
            self.send(client)


if __name__ == "__main__":
    unittest.main()
