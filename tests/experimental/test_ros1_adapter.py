from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import numpy as np

from experimental.legibility_optimizer import ros1_adapter as adapter
from experimental.legibility_optimizer.demo import synthetic_case


def message_fixture():
    q, t, _ = synthetic_case()
    return NS(joint_trajectory=NS(
        joint_names=["joint_b", "joint_a", "joint_c"],
        points=[NS(positions=list(row), velocities=[9, 9, 9],
                   accelerations=[8, 8, 8], effort=[7, 7, 7],
                   time_from_start=NS(to_sec=lambda value=value: float(value)))
                for row, value in zip(q, t)]),
        multi_dof_joint_trajectory=NS(joint_names=[], points=[]))


class AdapterTests(unittest.TestCase):
    def test_extract_preserves_order_positions_and_time(self):
        names, q, t = adapter.extract_path(message_fixture())
        self.assertEqual(names, ("joint_b", "joint_a", "joint_c"))
        expected_q, expected_t, _ = synthetic_case()
        np.testing.assert_array_equal(q, expected_q)
        np.testing.assert_array_equal(t, expected_t)

    def test_invalid_names_dimensions_and_empty_path_rejected(self):
        for names in ([], ["joint_a"] * 3, ["a", "", "c"], ["a", "b"]):
            msg = message_fixture()
            msg.joint_trajectory.joint_names = names
            with self.subTest(names=names), self.assertRaises(ValueError):
                adapter.extract_path(msg)
        msg = message_fixture()
        msg.joint_trajectory.points = []
        with self.assertRaises(ValueError):
            adapter.extract_path(msg)

    def test_unretimed_message_timestamps_rejected(self):
        msg = message_fixture()
        for point in msg.joint_trajectory.points:
            point.time_from_start = NS(to_sec=lambda: 0.0)
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            adapter.extract_path(msg)

    def test_multi_dof_rejected_not_silently_discarded(self):
        for multi in (NS(joint_names=["base"], points=[]), NS(joint_names=[], points=[NS()])):
            msg = message_fixture()
            msg.multi_dof_joint_trajectory = multi
            with self.assertRaisesRegex(ValueError, "multi-DOF"):
                adapter.extract_path(msg)

    def invoke(self, msg, **overrides):
        kwargs = dict(fk_position=lambda q, names: q, goal_frame="measured_frame",
                      fk_frame="measured_frame", joint_limits={
                          "joint_a": (-1, 1), "joint_b": (-2, 2), "joint_c": (-3, 3)},
                      max_joint_deviation={"joint_a": 0.1, "joint_b": 0.2, "joint_c": 0.3})
        kwargs.update(overrides)
        return adapter.optimize_robot_trajectory(msg, synthetic_case()[2], 0, **kwargs)

    def test_limits_and_fk_follow_message_joint_order(self):
        observed = []
        def fk(q, names):
            observed.append(names)
            return q
        with patch.object(adapter, "optimize_joint_path", return_value=NS()) as solve:
            candidate = self.invoke(message_fixture(), fk_position=fk)
            args, kwargs = solve.call_args
            self.assertEqual(kwargs["joint_lower"], [-2, -1, -3])
            self.assertEqual(kwargs["max_joint_deviation"], [0.2, 0.1, 0.3])
            args[4](np.zeros(3))
        self.assertEqual(observed, [candidate.joint_names])
        self.assertEqual(candidate.goal_frame, "measured_frame")
        self.assertFalse(hasattr(candidate, "joint_trajectory"))

    def test_missing_limits_and_frame_mismatch_rejected(self):
        for kwargs in (dict(fk_frame="another_frame"), dict(goal_frame="", fk_frame=""),
                       dict(joint_limits={"joint_a": (-1, 1)}),
                       dict(max_joint_deviation={"wrong_joint": 1}),
                       dict(velocity_limits={"wrong_joint": 1})):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.invoke(message_fixture(), **kwargs)


if __name__ == "__main__":
    unittest.main()
