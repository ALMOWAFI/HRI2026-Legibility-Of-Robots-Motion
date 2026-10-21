#!/usr/bin/env python3
"""Plan a test reach from the start pose to the hover point above one cube, show it in
RViz (MotionPlanning > Planned Path) and optionally run it on the UR5e.

usage: test_motion.py <B1..B4 | orange|red|yellow|green | all> [--legible] [--execute]

Predictable (straight) reach by default; --legible bends it sideways by experiment.bend.legible
(peak, at mid-reach) to the experiment.bend_side side, in the same horizontal plane.
Tool orientation is kept as at the start pose.
Every trajectory passes hri_legible_motion.safety (workspace, speed, joint jumps) before it
is shown or sent; --execute also needs a person at an interactive terminal to type 'go'.
"""
import os
import sys

import numpy as np
import rospy
import yaml

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))  # works without catkin build
from hri_legible_motion import trajectories  # noqa: E402
from hri_legible_motion.legibility import baseline_path, legible_path, sample  # noqa: E402
from hri_legible_motion.robot import Robot, end_joints  # noqa: E402
from hri_legible_motion.safety import SafetyError, require_operator  # noqa: E402

CONFIG = os.path.join(ROOT, "config", "experiment.yaml")


def plan_reach(robot, cfg, block, legible=False):
    m, e = cfg["motion"], cfg["experiment"]
    start = m["start_joints"]
    p0, orientation = robot.fk(start)
    p1 = np.array([block["position"][0], block["position"][1], m["touch_z"] + m["hover_height"]])
    # Samples are uniform in time under the min-jerk law, so MoveIt only solves IK per sample.
    if legible:
        bend = trajectories.bend_vector(p0, p1, e["bend_side"][block["id"]], e["bend"]["legible"])
        times, xyz = sample(legible_path(p0, p1, bend), m["duration"])
    else:
        times, xyz = sample(baseline_path(p0, p1), m["duration"])
    traj = robot.plan(start, times, xyz, orientation)
    length = np.linalg.norm(np.diff(xyz, axis=0), axis=1).sum()
    speed = np.linalg.norm(np.diff(xyz, axis=0), axis=1) / np.diff(times)
    qd = np.array([p.velocities for p in traj.points])
    print("Start  [%.3f %.3f %.3f]\nHover  [%.3f %.3f %.3f]  (%s, %s)" % (tuple(p0) + tuple(p1) + (block["name"], block["color"])))
    print("%s %.3f m in %.1f s, peak speed %.2f m/s, %d points, max joint speed %.0f deg/s"
          % ("Legible path" if legible else "Straight line", length, m["duration"], speed.max(), len(times),
             np.degrees(np.abs(qd)).max()))
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

    reaches = []
    for b in targets:
        traj = plan_reach(robot, cfg, b, legible)
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
