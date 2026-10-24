"""Offline contract tests using small ROS message stand-ins, without ROS or hardware."""
import copy
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import Mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


class Duration:
    def __init__(self, seconds):
        self.nanoseconds = round(float(seconds) * 1e9)

    def to_sec(self):
        return self.nanoseconds / 1e9

    def __eq__(self, other):
        return isinstance(other, Duration) and self.nanoseconds == other.nanoseconds


class Point:
    def __init__(self, time_from_start=None):
        self.time_from_start = time_from_start
        self.point = NS(pose=NS(position=NS(x=0., y=0., z=0.),
                               orientation=NS(x=0., y=0., z=0., w=1.)))


class Trajectory:
    def __init__(self):
        self.header = NS(frame_id="", stamp=None)
        self.tracked_frame, self.points = "", []


def load_source(name, relative, modules):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    previous = {key: sys.modules.get(key) for key in modules}
    sys.modules.update(modules)
    try:
        spec.loader.exec_module(module)
    finally:
        # Restore only our stand-ins; retain normal imports such as NumPy/SciPy extensions.
        for key, value in previous.items():
            if value is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = value
    return module


def topic_fixture():
    rospy = ModuleType("rospy")
    rospy.Duration = Duration
    rospy.Time = NS(now=Mock(return_value=123))
    rospy.is_shutdown = Mock(return_value=False)
    rospy.loginfo, rospy.logerr = Mock(), Mock()
    rospy.Publisher = Mock(return_value=Mock(get_num_connections=Mock(return_value=1)))
    rospy.Subscriber = Mock()
    messages = ModuleType("moveit_msgs.msg")
    messages.CartesianTrajectory, messages.CartesianTrajectoryPoint = Trajectory, Point
    modules = {"rospy": rospy, "moveit_msgs.msg": messages}
    topic = load_source("offline_reach_topic", "src/hri_legible_motion/reach_topic.py", modules)
    return topic, modules


def request(topic):
    return topic.to_msg([0., .500000001, 1.], [[.3, 0., .1], [.4, .1, .1], [.5, .2, .1]],
                        NS(x=0., y=0., z=0., w=1.), "base", "tool", stamp=0)


class ReachTopicTests(unittest.TestCase):
    def setUp(self):
        self.topic, self.modules = topic_fixture()
        self.msg = request(self.topic)

    def test_roundtrip_preserves_zero_stamp_and_exact_nanosecond_timing(self):
        self.assertEqual(self.msg.header.stamp, 0)
        times, xyz = self.topic.validate_message(self.msg, "base", "tool")
        xyz[1, 1] += .03
        reply = self.topic.reply_message(self.msg, times, xyz)
        self.topic.validate_response(self.msg, reply)
        self.assertEqual(reply.points[1].time_from_start.nanoseconds, 500000001)
        self.assertEqual(self.msg.points[1].point.pose.position.y, .1)

    def test_rejects_invalid_timing_shapes_and_nonfinite_values(self):
        valid = np.array([[.3, 0., .1], [.4, .1, .1], [.5, .2, .1]])
        for times in ([0., 0., 1.], [0., 2., 1.], [-1., 0., 1.], [0., np.nan, 1.], [0., np.inf, 1.]):
            with self.subTest(times=times), self.assertRaises(ValueError):
                self.topic.validate_samples(times, valid)
        for times, xyz in (([0., 1.], valid), ([0., .5, 1.], valid[:, :2]),
                           ([0., .5, 1.], valid * np.nan)):
            with self.assertRaises(ValueError):
                self.topic.to_msg(times, xyz, NS(x=0., y=0., z=0., w=1.), "base", "tool")

    def test_rejects_frame_or_link_mismatch(self):
        for frame, link in (("world", "tool"), ("base", "other_tool")):
            with self.subTest(frame=frame, link=link), self.assertRaises(ValueError):
                self.topic.validate_message(self.msg, frame, link)

    def test_rejects_nonunit_nonfinite_and_changing_orientation(self):
        for q in (NS(x=0., y=0., z=0., w=0.), NS(x=np.nan, y=0., z=0., w=1.),
                  NS(x=0., y=0., z=1., w=0.)):
            msg = copy.deepcopy(self.msg)
            msg.points[1].point.pose.orientation = q
            with self.subTest(q=q), self.assertRaises(ValueError):
                self.topic.validate_message(msg, "base", "tool")

    def test_empty_failure_reply_preserves_identity(self):
        reply = self.topic.reply_message(self.msg)
        self.assertEqual(reply.points, [])
        self.assertEqual(reply.header.stamp, self.msg.header.stamp)
        self.assertEqual(reply.header.frame_id, "base")
        with self.assertRaises(ValueError):
            self.topic.validate_response(self.msg, reply)

    def test_rejects_reordered_or_changed_timestamps(self):
        for change in (lambda msg: msg.points.reverse(),
                       lambda msg: setattr(msg.points[1], "time_from_start", Duration(.6))):
            reply = copy.deepcopy(self.msg)
            change(reply)
            with self.assertRaises(ValueError):
                self.topic.validate_response(self.msg, reply)

    def test_rejects_changed_endpoint_height_orientation_and_identity(self):
        changes = [lambda msg: setattr(msg.points[-1].point.pose.position, "x", .501),
                   lambda msg: setattr(msg.points[1].point.pose.position, "z", .2),
                   lambda msg: setattr(msg.header, "stamp", 999)]
        for change in changes:
            reply = copy.deepcopy(self.msg)
            change(reply)
            with self.assertRaises(ValueError):
                self.topic.validate_response(self.msg, reply)
        reply = copy.deepcopy(self.msg)
        for p in reply.points:
            p.point.pose.orientation = NS(x=0., y=0., z=1., w=0.)
        with self.assertRaises(ValueError):
            self.topic.validate_response(self.msg, reply)


if __name__ == "__main__":
    unittest.main()
