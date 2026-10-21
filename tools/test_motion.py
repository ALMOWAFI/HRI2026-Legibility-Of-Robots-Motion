#!/usr/bin/env python3
"""Plan a test reach from the start pose to the hover point above one cube, show it in
RViz (MotionPlanning > Planned Path) and optionally run it on the UR5e.

usage: test_motion.py <B1..B4 | orange|red|yellow|green | all> [--legible] [--execute]

Plans the predictable (straight) reach and publishes it on /hri/predictable_reach.
--legible: waits for the optimizer's answer on /hri/legible_reach (start
scripts/legibility_optimizer_node.py first) and plans that reach instead: same start,
end, height and duration, only x-y differ. Tool orientation is kept as at the start pose.
Every trajectory passes hri_legible_motion.safety (workspace, speed, joint jumps) before it
is shown or sent; --execute also needs a person at an interactive terminal to type 'go'.
"""
import os
import queue
import sys

import numpy as np
import rospy
import yaml
from moveit_msgs.msg import CartesianTrajectory

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))  # works without catkin build
from hri_legible_motion import reach_topic  # noqa: E402
from hri_legible_motion.legibility_optimizer import baseline_path, planar_legibility, sample  # noqa: E402
from hri_legible_motion.robot import Robot, end_joints  # noqa: E402
from hri_legible_motion.safety import SafetyError, require_operator  # noqa: E402

CONFIG = os.path.join(ROOT, "config", "experiment.yaml")


class Optimizer:
    """Publishes predictable reaches; with legible=True also waits for each answer."""

    def __init__(self, cfg, legible):
        self.cfg, self.legible = cfg, legible
        self.pub = rospy.Publisher(reach_topic.PREDICTABLE, CartesianTrajectory, queue_size=10)
        self.replies = queue.Queue()
        if legible:
            rospy.Subscriber(reach_topic.LEGIBLE, CartesianTrajectory, self.replies.put)
        deadline = rospy.Time.now() + rospy.Duration(5.0 if legible else 1.0)
        while self.pub.get_num_connections() == 0 and rospy.Time.now() < deadline:
            rospy.sleep(0.1)
        if legible and self.pub.get_num_connections() == 0:
            raise SafetyError("Nobody subscribes to %s: start scripts/legibility_optimizer_node.py first."
                              % reach_topic.PREDICTABLE)

    def send(self, times, xyz, orientation):
        """Publish the predictable reach; returns the legible (times, xyz) if legible, else None."""
        stamp = rospy.Time.now()
        self.pub.publish(reach_topic.to_msg(times, xyz, orientation, self.cfg["frame_id"],
                                            self.cfg["motion"]["ee_link"], stamp=stamp))
        if not self.legible:
            return None
        timeout = self.cfg.get("optimizer", {}).get("timeout", 60.0)
        deadline = rospy.get_time() + timeout
        while rospy.get_time() < deadline:
            try:
                msg = self.replies.get(timeout=max(0.1, deadline - rospy.get_time()))
            except queue.Empty:
                break
            if msg.header.stamp != stamp:
                continue  # answer to another request
            if not msg.points:
                raise SafetyError("The optimizer found no legible path (see its log).")
            return reach_topic.from_msg(msg)
        raise SafetyError("No answer from the optimizer within %.0f s." % timeout)


def plan_reach(robot, cfg, block, optimizer):
    m = cfg["motion"]
    start = m["start_joints"]
    p0, orientation = robot.fk(start)
    p1 = np.array([block["position"][0], block["position"][1], m["touch_z"] + m["hover_height"]])
    # Samples are uniform in time under the min-jerk law, so MoveIt only solves IK per sample.
    times, xyz = sample(baseline_path(p0, p1), m["duration"])
    goals = reach_topic.hover_points(cfg)
    score = planar_legibility(xyz, goals, block["id"])
    legible = optimizer.send(times, xyz, orientation)
    if legible is not None:
        times, xyz = legible
        if np.linalg.norm(xyz[0] - p0) > 1e-3 or np.linalg.norm(xyz[-1] - p1) > 1e-3 \
                or np.abs(xyz[:, 2] - p1[2]).max() > 1e-3:
            raise SafetyError("Legible reach to %s does not keep start, end and height." % block["name"])
        print("Legibility %.3f (straight) -> %.3f (legible)" % (score, planar_legibility(xyz, goals, block["id"])))
    else:
        print("Legibility %.3f" % score)
    traj = robot.plan(start, times, xyz, orientation)
    length = np.linalg.norm(np.diff(xyz, axis=0), axis=1).sum()
    speed = np.linalg.norm(np.diff(xyz, axis=0), axis=1) / np.diff(times)
    qd = np.array([p.velocities for p in traj.points])
    print("Start  [%.3f %.3f %.3f]\nHover  [%.3f %.3f %.3f]  (%s, %s)" % (tuple(p0) + tuple(p1) + (block["name"], block["color"])))
    print("%s %.3f m in %.1f s, peak speed %.2f m/s, %d points, max joint speed %.0f deg/s"
          % ("Legible path" if legible is not None else "Straight line", length, times[-1], speed.max(),
             len(times), np.degrees(np.abs(qd)).max()))
    return traj


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    with open(CONFIG) as f:
        cfg = yaml.safe_load(f)
    blocks = {k: b for b in cfg["blocks"] for k in (b["name"], b["color"])}
    if len(args) != 1 or (args[0] not in blocks and args[0] != "all"):
        sys.exit(__doc__)
    targets = cfg["blocks"] if args[0] == "all" else [blocks[args[0]]]
    execute, legible = "--execute" in sys.argv, "--legible" in sys.argv
    if execute and not sys.stdin.isatty():
        sys.exit("--execute needs a person at an interactive terminal.")
    rospy.init_node("test_motion", anonymous=True)
    robot = Robot(cfg)
    m = cfg["motion"]
    start = m["start_joints"]
    orientation = robot.fk(start)[1]
    optimizer = Optimizer(cfg, legible)

    reaches = []
    for b in targets:
        traj = plan_reach(robot, cfg, b, optimizer)
        back = robot.plan_return(end_joints(traj), orientation)  # checked now, replanned after the reach
        print("Return to start: %.1f s\n" % back.points[-1].time_from_start.to_sec())
        reaches.append((b, traj))
    robot.show(start, [traj, back])
    print("Shown in RViz (MotionPlanning > Planned Path; last one if several).")
    if not execute:
        return

    robot.connect()
    print("Speed slider must be 100 %, or the reach will not take the planned time.")
    if not robot.go_to_start(orientation, cfg["setup"]["safe_height"]):
        return
    for b, traj in reaches:
        robot.check_at(start, "the start pose; the planned reach starts there")
        if not require_operator("Run the %s reach to %s (%.1f s)?"
                                % ("legible" if legible else "predictable", b["color"], m["duration"])):
            return
        robot.execute(traj)
        print("Reach to %s done." % b["color"])
        rospy.sleep(1.0)
        robot.execute(robot.plan_return(robot.current_joints(), orientation))
        robot.move_joints(start, 2.0)  # residual correction, a fraction of a degree
        print("Back at start.\n")


if __name__ == "__main__":
    try:
        main()
    except SafetyError as e:
        sys.exit("SAFETY STOP: %s" % e)
