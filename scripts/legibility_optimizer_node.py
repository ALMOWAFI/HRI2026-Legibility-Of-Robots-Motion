#!/usr/bin/env python3
"""Legibility optimizer node: predictable reach in, legible reach out (no robot motion).

Subscribes /hri/predictable_reach, finds the target cube from where the reach ends, runs
legibility_optimizer.optimize_planar_reach() (same start, end, height and duration, only
x-y changes) and publishes the result on /hri/legible_reach with the request's stamp.
An empty reply means the optimizer failed. IK and the safety checks happen in the node
that executes the reach (test_motion.py), exactly as for the predictable reach.

usage: legibility_optimizer_node.py [config.yaml]     (default: config/experiment.yaml)
"""
import os
import sys

import rospy
import yaml
from moveit_msgs.msg import CartesianTrajectory

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))  # works without catkin build
from hri_legible_motion import reach_topic  # noqa: E402
from hri_legible_motion.legibility_optimizer import PlanarConfig, optimize_planar_reach  # noqa: E402


class OptimizerNode:
    def __init__(self, cfg):
        self.cfg = cfg
        self.config = PlanarConfig.from_cfg(cfg)
        self.goals = reach_topic.hover_points(cfg)
        self.pub = rospy.Publisher(reach_topic.LEGIBLE, CartesianTrajectory, queue_size=10)
        rospy.Subscriber(reach_topic.PREDICTABLE, CartesianTrajectory, self.on_reach, queue_size=10)
        rospy.loginfo("Optimizer ready: %s -> %s (%s)", reach_topic.PREDICTABLE, reach_topic.LEGIBLE, self.config)

    def reply(self, request, times=(), xyz=()):
        orientation = request.points[0].point.pose.orientation if request.points else None
        self.pub.publish(reach_topic.to_msg(times, xyz, orientation, request.header.frame_id,
                                            request.tracked_frame, stamp=request.header.stamp))

    def on_reach(self, msg):
        times, xyz = reach_topic.from_msg(msg)
        target = reach_topic.target_of(self.cfg, xyz[-1]) if len(xyz) > 2 else None
        if target is None:
            rospy.logerr("Reach does not end at a cube's hover point (or has < 3 points); not optimized.")
            return self.reply(msg)
        name = self.cfg["blocks"][target]["name"]
        try:
            out = optimize_planar_reach(times, xyz, self.goals, target, self.config)
        except ValueError as e:
            rospy.logerr("%s: %s", name, e)
            return self.reply(msg)
        if not out.success:
            rospy.logerr("%s: optimizer failed: %s", name, out.message)
            return self.reply(msg)
        rospy.loginfo("%s: legibility %.3f -> %.3f, path x%.3f, max lateral %.3f m",
                      name, out.baseline_score, out.optimized_score, out.length_ratio, out.max_lateral)
        self.reply(msg, out.times, out.xyz)


def main():
    args = rospy.myargv()[1:]
    with open(args[0] if args else os.path.join(ROOT, "config", "experiment.yaml")) as f:
        cfg = yaml.safe_load(f)
    rospy.init_node("legibility_optimizer")
    OptimizerNode(cfg)
    rospy.spin()


if __name__ == "__main__":
    main()
