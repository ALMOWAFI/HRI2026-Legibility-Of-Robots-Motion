#!/usr/bin/env python3
"""Plan a test reach from the start pose to the hover point above one cube, show it in
RViz (MotionPlanning > Planned Path) and optionally run it on the UR5e.

usage: test_motion.py <B1..B4 | orange|red|yellow|green | all> [--legible] [--execute]

Supports both predictable (straight) and legible (laterally bent) motion.
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
from hri_legible_motion.legibility_optimizer import (  # noqa: E402
    baseline_path,
    legible_path,
    legibility_score,
    sample,
)
from hri_legible_motion import trajectories  # noqa: E402
from hri_legible_motion.robot import Robot, end_joints  # noqa: E402
from hri_legible_motion.safety import SafetyError, require_operator  # noqa: E402

CONFIG = os.path.join(ROOT, "config", "experiment.yaml")


def plan_reach(robot, cfg, block, motion="predictable"):
    m, e = cfg["motion"], cfg["experiment"]
    start = m["start_joints"]
    p0, orientation = robot.fk(start)
    p1 = np.array([block["position"][0], block["position"][1], m["touch_z"] + m["hover_height"]])

    all_goals = np.array([[b["position"][0], b["position"][1], m["touch_z"] + m["hover_height"]]
                          for b in cfg["blocks"]])

    if motion == "predictable":
        path = baseline_path(p0, p1)
    else:
        bend = trajectories.bend_vector(p0, p1, e["bend_side"][block["id"]], e["bend"]["legible"])
        path = legible_path(p0, p1, bend)

    times, xyz = sample(path, m["duration"])
    traj = robot.plan(start, times, xyz, orientation)
    length = float(np.linalg.norm(np.diff(xyz, axis=0), axis=1).sum())
    peak_speed = float(np.linalg.norm(np.diff(xyz, axis=0), axis=1).max() / (times[1] - times[0]))
    qd = np.array([p.velocities for p in traj.points])

    score = legibility_score((times, xyz), all_goals, block["id"])

    print("Start  [%.3f %.3f %.3f]" % tuple(p0))
    print("Hover  [%.3f %.3f %.3f]  (%s, %s, %s)" % (tuple(p1) + (block["name"], block["color"], motion)))
    print("%s path %.3f m in %.1f s, peak speed %.2f m/s, %d points, max joint speed %.0f deg/s"
          % (motion.capitalize(), length, m["duration"], peak_speed, len(times), np.degrees(np.abs(qd)).max()))
    print("Dragan Legibility Score: %.4f" % score)
    return traj


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    with open(CONFIG) as f:
        cfg = yaml.safe_load(f)
    blocks = {k: b for b in cfg["blocks"] for k in (b["name"], b["color"])}
    if len(args) != 1 or (args[0] not in blocks and args[0] != "all"):
        sys.exit(__doc__)
    targets = cfg["blocks"] if args[0] == "all" else [blocks[args[0]]]
    motion = "legible" if "--legible" in sys.argv else "predictable"
    execute = "--execute" in sys.argv
    if execute and not sys.stdin.isatty():
        sys.exit("--execute needs a person at an interactive terminal.")
    rospy.init_node("test_motion", anonymous=True)
    robot = Robot(cfg)
    m = cfg["motion"]
    start = m["start_joints"]
    orientation = robot.fk(start)[1]

    reaches = []
    for b in targets:
        traj = plan_reach(robot, cfg, b, motion=motion)
        back = robot.plan_return(end_joints(traj), orientation)  # checked now, replanned after the reach
        print("Return to start: %.1f s\n" % back.points[-1].time_from_start.to_sec())
        reaches.append((b, traj))
    robot.show(start, [traj, back])
    print("Shown in RViz (MotionPlanning > Planned Path; last one if several).")
    if not execute:
        return

    robot.connect()
    print("Speed slider must be 100 %, or the reach will not take the planned time.")
    off = robot.distance_deg(robot.current_joints(), start)
    if off > cfg["safety"]["start_tolerance_deg"]:
        robot.check_move_joints(start, 8.0)  # refuses big configuration changes before asking
        if not require_operator("Robot is %.0f deg from the start pose. Move there in 8 s?" % off):
            return
        robot.move_joints(start, 8.0)
    for b, traj in reaches:
        robot.check_at(start, "the start pose; the planned reach starts there")
        if not require_operator("Run the %s reach to %s (%.1f s)?" % (motion, b["color"], m["duration"])):
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
