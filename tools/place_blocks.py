#!/usr/bin/env python3
"""Setup before a session: the UR5e points at each cube position from config/experiment.yaml
in turn (orange, red, yellow, green), you put that cube under the tool tip, and it is added to
RViz and the MoveIt scene. Afterwards the real cubes are where the config says they are.

usage: place_blocks.py              plan everything and preview it in RViz (no motion)
       place_blocks.py --execute    run it on the robot (needs a person at the terminal)

Each move: lift to safe_height, travel, lower to point_height above the cube top; slow,
with a stop at every corner. Every move needs 'go' typed at the terminal.
"""
import os
import sys

import numpy as np
import rospy
import yaml

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))  # works without catkin build
from hri_legible_motion.legibility import stop_and_go  # noqa: E402
from hri_legible_motion.robot import Robot, end_joints  # noqa: E402
from hri_legible_motion.safety import SafetyError, require_operator  # noqa: E402
from hri_legible_motion.scene import set_cubes  # noqa: E402

ANSI = {"orange": "\033[38;5;208m", "red": "\033[31m", "yellow": "\033[33m", "green": "\033[32m"}


def plan_visit(robot, cfg, start, orientation, block):
    """From joint dict `start`: up to safe height, over the block, down to the pointing height."""
    s, m = cfg["setup"], cfg["motion"]
    p = robot.fk(start)[0]
    x, y = block["position"][:2]
    z_safe, z_point = s["safe_height"], m["touch_z"] + s["point_height"]
    points = [p, [p[0], p[1], max(p[2], z_safe)], [x, y, max(p[2], z_safe)], [x, y, z_point]]
    times, xyz = stop_and_go(points, s["speed"])
    return robot.plan(start, times, xyz, orientation)


def plan_lift(robot, cfg, start, orientation):
    p = robot.fk(start)[0]
    times, xyz = stop_and_go([p, [p[0], p[1], max(p[2], cfg["setup"]["safe_height"])]], cfg["setup"]["speed"])
    return robot.plan(start, times, xyz, orientation)


def banner(block):
    c = ANSI.get(block["color"], "")
    print("\n" + c + "=" * 60)
    print("   PLACE THE %s CUBE  (%s)" % (block["color"].upper(), block["name"]))
    print("=" * 60 + "\033[0m")
    print("   Centre it under the tool tip, long side (5.3 cm) pointing back along the robot's")
    print("   reach, toward where the tool starts (turned %+.0f deg from straight ahead)." % block.get("yaw", 0.0))
    print("   The robot is stopped and will not move until you type 'go' again.")


def preview(robot, cfg, start, orientation):
    trajs, joints = [], start
    for b in cfg["blocks"]:
        trajs.append(plan_visit(robot, cfg, joints, orientation, b))
        joints = end_joints(trajs[-1])
        print("%-7s %-2s  tool at %s  (%.1f s)" % (b["color"], b["name"], np.round(robot.fk(joints)[0], 3).tolist(),
                                                     trajs[-1].points[-1].time_from_start.to_sec()))
    trajs.append(plan_lift(robot, cfg, joints, orientation))
    robot.show(start, trajs)
    print("All moves pass the safety checks. Preview shown in RViz (MotionPlanning > Planned Path).")
    print("Run with --execute at the robot to place the cubes.")


def run(robot, cfg, start, orientation):
    m = cfg["motion"]
    robot.connect()
    set_cubes(cfg, [])  # nothing placed yet
    print("Clear the table. Speed slider 100 %. Hand on the e-stop for every move.")
    if robot.distance_deg(robot.current_joints(), start) > cfg["safety"]["start_tolerance_deg"]:
        if not require_operator("Move to the start pose (8 s)?"):
            return
        robot.move_joints(start, 8.0)
    robot.check_at(start, "the start pose")

    placed = []
    for b in cfg["blocks"]:
        traj = plan_visit(robot, cfg, robot.current_joints(), orientation, b)
        if not require_operator("Move to the %s position (%s, %.0f s)?"
                                % (b["color"], b["name"], traj.points[-1].time_from_start.to_sec())):
            break
        robot.execute(traj)
        banner(b)
        input("   Press Enter when the cube is placed and your hand is clear... ")
        placed.append(b)
        set_cubes(cfg, placed)
        print("   %s cube added to RViz." % b["color"])

    print("\nPlaced %d/%d cubes: %s" % (len(placed), len(cfg["blocks"]), ", ".join(b["color"] for b in placed)))
    if require_operator("Lift and return to the start pose?"):
        robot.execute(plan_lift(robot, cfg, robot.current_joints(), orientation))
        robot.move_joints(start, 6.0)
        print("Back at start.")


def main():
    execute = "--execute" in sys.argv
    if execute and not sys.stdin.isatty():
        sys.exit("--execute needs a person at an interactive terminal.")
    with open(os.path.join(ROOT, "config", "experiment.yaml")) as f:
        cfg = yaml.safe_load(f)
    rospy.init_node("place_blocks", anonymous=True)
    robot = Robot(cfg)
    start = cfg["motion"]["start_joints"]
    orientation = robot.fk(start)[1]  # tool pointing down, as at the start pose
    (run if execute else preview)(robot, cfg, start, orientation)


if __name__ == "__main__":
    try:
        main()
    except SafetyError as e:
        sys.exit("SAFETY STOP: %s" % e)
