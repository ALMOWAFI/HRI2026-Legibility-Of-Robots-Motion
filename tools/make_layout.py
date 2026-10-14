#!/usr/bin/env python3
"""Compute the cube arc and the start pose, check that everything is reachable, and
(with --write) update config/experiment.yaml. No robot motion.

usage: make_layout.py [--back M] [--up M] [--radius M] [--spacing M] [--write]

  --back     move the start tool position this far back (toward the base, -x)  [0]
  --up       raise the start tool position by this much                        [0]
  --radius   arc radius around the start tool position (= horizontal reach)   [current]
  --spacing  straight-line distance between neighbouring cubes                  [0.18]

The start pose is solved by IK seeded with the current start_joints (same arm
configuration), tool orientation unchanged. Cubes sit on an arc centred on the start tool
position, symmetric about the robot's x axis through it, long side along the reach line.
After --write: re-run place_blocks.py to put the cubes on the new spots.
"""
import argparse
import math
import os
import re

import numpy as np
import rospy
import yaml
from geometry_msgs.msg import PoseStamped
from moveit_msgs.msg import PositionIKRequest, RobotState
from moveit_msgs.srv import GetPositionFK, GetPositionIK
from sensor_msgs.msg import JointState

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
CONFIG = os.path.join(ROOT, "config", "experiment.yaml")
HEIGHTS = (0.0, 0.02, 0.05, 0.08, 0.12)  # above touch_z: placing, hovering, travelling


def js(joints):
    return JointState(name=list(joints), position=list(joints.values()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--back", type=float, default=0.0)
    ap.add_argument("--up", type=float, default=0.0)
    ap.add_argument("--radius", type=float)
    ap.add_argument("--spacing", type=float, default=0.18)
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()

    cfg = yaml.safe_load(open(CONFIG))
    frame, m = cfg["frame_id"], cfg["motion"]
    rospy.init_node("make_layout", anonymous=True)
    fk = rospy.ServiceProxy("/compute_fk", GetPositionFK)
    ik = rospy.ServiceProxy("/compute_ik", GetPositionIK)

    old = m["start_joints"]
    pose = fk(header=rospy.Header(frame_id=frame), fk_link_names=[m["ee_link"]],
              robot_state=RobotState(joint_state=js(old))).pose_stamped[0].pose
    p_old = np.array([pose.position.x, pose.position.y, pose.position.z])
    p_new = p_old + [-a.back, 0.0, a.up]

    def solve(xyz, seed):
        ps = PoseStamped()
        ps.header.frame_id = frame
        ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = xyz
        ps.pose.orientation = pose.orientation
        res = ik(PositionIKRequest(group_name=m["planning_group"], robot_state=RobotState(joint_state=js(seed)),
                                   pose_stamped=ps, avoid_collisions=True, timeout=rospy.Duration(0.5)))
        if res.error_code.val != 1:
            return None
        sol = dict(zip(res.solution.joint_state.name, res.solution.joint_state.position))
        return {n: sol[n] for n in seed}

    start = solve(p_new, old)
    if start is None:
        raise SystemExit("New start position %s is not reachable." % np.round(p_new, 3).tolist())
    change = max(abs(start[n] - old[n]) for n in old)
    print("Start tool  %s -> %s   (joints change up to %.1f deg)"
          % (np.round(p_old, 3).tolist(), np.round(p_new, 3).tolist(), math.degrees(change)))

    if a.radius is None:
        b0 = cfg["blocks"][0]["position"]
        a.radius = math.hypot(b0[0] - p_old[0], b0[1] - p_old[1])
    step = 2 * math.asin(a.spacing / 2 / a.radius)
    n = len(cfg["blocks"])
    ok = True
    new_blocks = []
    for i, b in enumerate(cfg["blocks"]):
        ang = ((n - 1) / 2 - i) * step  # first block on the +y (left) side
        x, y = p_new[0] + a.radius * math.cos(ang), p_new[1] + a.radius * math.sin(ang)
        reach = [solve([x, y, m["touch_z"] + h], start) is not None for h in HEIGHTS]
        ok &= all(reach)
        old_xy = b["position"][:2]
        print("%-7s x=%.3f y=%+.3f yaw=%+5.1f  %.2f m from base, moved %.1f cm  reachable: %s"
              % (b["color"], x, y, math.degrees(ang), math.hypot(x, y),
                 100 * math.hypot(x - old_xy[0], y - old_xy[1]),
                 "all heights" if all(reach) else "NO at " + ", ".join(
                     "+%.0f cm" % (100 * h) for h, r in zip(HEIGHTS, reach) if not r)))
        new_blocks.append((b, x, y, math.degrees(ang)))
    print("Arc radius %.3f m, spacing %.3f m (%.1f deg)" % (a.radius, a.spacing, math.degrees(step)))
    if not ok:
        raise SystemExit("Some spots are not reachable; config not changed. Try a smaller --radius.")
    if not a.write:
        print("Preview only. Add --write to update %s." % os.path.relpath(CONFIG))
        return

    text = open(CONFIG).read()
    for b, x, y, yaw in new_blocks:
        text, k = re.subn(r"(name: %s,.*?position: )\[[^\]]*\](, yaw: *)[-+0-9.]+" % b["name"],
                          lambda mt: "%s[%.3f, %.3f, %.3f]%s%.1f" % (mt.group(1), x, y, b["position"][2], mt.group(2), yaw),
                          text)
        if k != 1:
            raise SystemExit("Could not find block %s in the config." % b["name"])
    joints = ", ".join("%s: %.4f" % (n_, start[n_]) for n_ in old)
    text, k = re.subn(r"start_joints: \{[^}]*\}", "start_joints: {%s}" % joints, text)
    if k != 1:
        raise SystemExit("Could not find start_joints in the config.")
    text = re.sub(r"# Start of every reach:.*", "# Start of every reach: tool0 at (%.3f, %.3f, %.3f), pointing down "
                  "(make_layout.py)." % tuple(p_new), text)
    text = re.sub(r"# Arc of radius [0-9.]+ m around the start pose's tool position \([^)]*\)",
                  "# Arc of radius %.2f m around the start pose's tool position (%.3f, %.3f)"
                  % (a.radius, p_new[0], p_new[1]), text)
    open(CONFIG, "w").write(text)
    print("Updated %s. Next: robot to the new start, then place_blocks.py." % os.path.relpath(CONFIG))


if __name__ == "__main__":
    main()
