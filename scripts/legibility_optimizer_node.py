#!/usr/bin/env python3
"""Legibility optimizer node: predictable reach in, legible reach out (no robot motion).

Subscribes /hri/predictable_reach, finds the target cube from where the reach ends, runs
legibility_optimizer.optimize_planar_reach() (same start, end, height and duration, only
x-y changes) and publishes the result on /hri/legible_reach with the request's stamp.
An empty reply means the optimizer failed. IK and the safety checks happen in the node
that executes the reach (test_motion.py), exactly as for the predictable reach.

usage: legibility_optimizer_node.py [config.yaml]     (default: config/experiment.yaml)
"""
import argparse
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

    def reply(self, request, times=None, xyz=None):
        self.pub.publish(reach_topic.reply_message(request, times, xyz))

    def on_reach(self, msg):
        try:
            times, xyz = reach_topic.validate_message(msg, self.cfg["frame_id"], self.cfg["motion"]["ee_link"])
            target = reach_topic.target_of(self.cfg, xyz[-1])
            if target is None:
                raise ValueError("Reach does not end at a configured cube hover point.")
            name = self.cfg["blocks"][target]["name"]
            out = optimize_planar_reach(times, xyz, self.goals, target, self.config)
            if not (out.success and out.numerical_success and out.geometric_valid):
                rospy.logerr("%s: optimizer rejected: %s (numerical_success=%s, geometric_valid=%s)",
                             name, out.message, getattr(out, "numerical_success", False),
                             getattr(out, "geometric_valid", False))
                reasons = [attempt.get("rejection", attempt.get("message", "unknown"))
                           for attempt in out.diagnostics.get("attempts", []) if not attempt.get("accepted")]
                if reasons:
                    rospy.logerr("%s: rejected attempts: %s", name, reasons)
                return self.reply(msg)
            self.reply(msg, out.times, out.xyz)
        except (ValueError, TypeError, AttributeError, IndexError) as e:
            rospy.logerr("Optimizer rejected reach: %s", e)
            return self.reply(msg)
        rospy.loginfo("%s: model score %.4f -> %.4f (gain %+.2f percentage points), "
                      "max lateral %.2f cm, path x%.4f; beta=%.3g, lambda=%.3g, "
                      "numerical_success=%s, geometric_valid=%s, physical_validity=not_checked. "
                      "MoveIt and robot validation remain required.",
                      name, out.baseline_score, out.optimized_score,
                      100 * (out.optimized_score - out.baseline_score), 100 * out.max_lateral,
                      out.length_ratio, self.config.beta, self.config.efficiency_weight,
                      getattr(out, "numerical_success", out.success), getattr(out, "geometric_valid", False))
        diagnostics = out.diagnostics
        rospy.loginfo("%s: active constraints=%s; minimum target margin=%s; search attempts=%d",
                      name, diagnostics.get("active_constraints", []),
                      diagnostics.get("optimized_metrics", {}).get("minimum_goal_margin"),
                      len(diagnostics.get("attempts", [])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config_path", nargs="?", help="Configuration YAML (legacy positional form)")
    parser.add_argument("--config", help="Configuration YAML shared with test_motion.py")
    args = parser.parse_args(rospy.myargv()[1:])
    if args.config and args.config_path:
        parser.error("Use either --config or the positional config path.")
    path = os.path.abspath(args.config or args.config_path or os.path.join(ROOT, "config", "experiment.yaml"))
    with open(path) as f:
        cfg = yaml.safe_load(f)
    rospy.init_node("legibility_optimizer")
    rospy.loginfo("Optimizer configuration: %s (loaded once; restart after editing)", path)
    OptimizerNode(cfg)
    rospy.spin()


if __name__ == "__main__":
    main()
