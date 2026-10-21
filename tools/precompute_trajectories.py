#!/usr/bin/env python3
"""Plan the study's reaches once (2 motions x 4 cubes), check them, preview them in RViz and
save them to config/trajectories.npz with a hash (plan §2.4). No robot motion.

usage: precompute_trajectories.py [--only predictable|legible] [--out FILE]

Needs MoveIt (move_group) running: the real robot stack or `roslaunch ur5e_moveit_config demo.launch`.
Each reach: start pose -> hover point above the cube, min-jerk on arc length, duration
motion.duration. Reaches are 2D: the start pose must be at the hover height
(touch_z + hover_height), so every path stays in that horizontal plane. The legible path
comes from legibility.legible_path(start, goal, bend), with bend = the lateral offset
vector at mid-reach (plan §2.1-2.2). Each reach's 2D legibility score is printed and saved.
"""
import argparse
import os
import sys

import numpy as np
import rospy
import yaml

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))  # works without catkin build
from hri_legible_motion import legibility, trajectories  # noqa: E402
from hri_legible_motion.robot import Robot, end_joints  # noqa: E402
from hri_legible_motion.safety import SafetyError  # noqa: E402
from hri_legible_motion.experiment import MOTIONS  # noqa: E402

CONFIG = os.path.join(ROOT, "config", "experiment.yaml")


def path_for(motion, p0, p1, bend):
    if motion == "predictable":
        return legibility.baseline_path(p0, p1)
    return legibility.legible_path(p0, p1, bend)


def lateral_deviation(xyz, p0, p1):
    """Largest horizontal distance of the tool path from the straight line p0 -> p1."""
    d = (p1 - p0)[:2] / np.linalg.norm((p1 - p0)[:2])
    rel = xyz[:, :2] - p0[:2]
    return float(np.abs(rel[:, 0] * d[1] - rel[:, 1] * d[0]).max())


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--only", choices=MOTIONS)
    ap.add_argument("--out")
    args = ap.parse_args(rospy.myargv()[1:])
    with open(CONFIG) as f:
        cfg = yaml.safe_load(f)
    m, e = cfg["motion"], cfg["experiment"]
    out = args.out or os.path.join(ROOT, cfg["trajectories"])

    rospy.init_node("precompute_trajectories", anonymous=True)
    robot = Robot(cfg)
    start = m["start_joints"]
    p0, orientation = robot.fk(start)
    plane_z = m["touch_z"] + m["hover_height"]
    if abs(p0[2] - plane_z) > 0.002:
        sys.exit("Start pose tool0 z is %.3f, the reach plane is %.3f (touch_z + hover_height). "
                 "Move the start pose into the plane: tools/make_layout.py --up %.3f --write"
                 % (p0[2], plane_z, plane_z - p0[2]))
    goals = [b["position"][:2] for b in cfg["blocks"]]

    reaches, trajs, names, scores = {}, [], None, {}
    for motion in ([args.only] if args.only else MOTIONS):
        for b in cfg["blocks"]:
            p1 = np.array([b["position"][0], b["position"][1], plane_z])
            bend = trajectories.bend_vector(p0, p1, e["bend_side"][b["id"]], e["bend"][motion])
            times, xyz = legibility.sample(path_for(motion, p0, p1, bend), m["duration"])
            if np.linalg.norm(xyz[0] - p0) > 1e-3 or np.linalg.norm(xyz[-1] - p1) > 1e-3:
                raise SafetyError("%s path to %s does not start at the start pose / end at the hover point"
                                  % (motion, b["name"]))
            if np.abs(xyz[:, 2] - plane_z).max() > 1e-3:
                raise SafetyError("%s path to %s leaves the reach plane z = %.3f" % (motion, b["name"], plane_z))
            traj = robot.plan(start, times, xyz, orientation)
            robot.plan_return(end_joints(traj), orientation)  # the return must be plannable too
            names = traj.joint_names
            q = np.array([p.positions for p in traj.points])
            qd = np.array([p.velocities for p in traj.points])
            length = np.linalg.norm(np.diff(xyz, axis=0), axis=1).sum()
            speed = np.linalg.norm(np.diff(xyz, axis=0), axis=1) / np.diff(times)
            score = legibility.legibility_score(xyz, goals, b["id"])
            scores[trajectories.key(motion, b["id"])] = score
            print("%-11s %-6s %-6s path %.3f m, peak %.2f m/s, lateral dev %.3f m, max joint %.0f deg/s, "
                  "legibility %.3f" % (motion, b["name"], b["color"], length, speed.max(),
                                       lateral_deviation(xyz, p0, p1), np.degrees(np.abs(qd)).max(), score))
            reaches[trajectories.key(motion, b["id"])] = {"times": times, "positions": q,
                                                          "velocities": qd, "xyz": xyz}
            trajs.append(traj)

    if not reaches:
        sys.exit("Nothing planned.")
    robot.show(start, trajs)
    print("Shown in RViz (MotionPlanning > Planned Path), all reaches in a row.")
    meta = trajectories.save(out, names, start, reaches, {
        "duration": m["duration"], "hover_height": m["hover_height"], "touch_z": m["touch_z"],
        "bend": e["bend"], "bend_side": e["bend_side"], "plane_z": plane_z, "legibility": scores,
        "blocks": {b["name"]: b["position"] for b in cfg["blocks"]}})
    print("Saved %d reaches to %s (hash %s)." % (len(reaches), out, meta["hash"]))


if __name__ == "__main__":
    try:
        main()
    except SafetyError as e:
        sys.exit("SAFETY STOP: %s" % e)
