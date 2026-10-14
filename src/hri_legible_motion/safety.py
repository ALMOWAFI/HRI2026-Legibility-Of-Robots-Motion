"""Safety checks run before anything is sent to the UR5e.

check_path()        Cartesian waypoints inside the workspace box, above the cubes, slow enough
check_joints()      joint speeds and step sizes within limits
require_operator()  a human at an interactive terminal confirms the motion

Limits live in config/experiment.yaml under `safety`. Every check raises SafetyError;
callers must not catch it and continue.
"""
import sys

import numpy as np


class SafetyError(RuntimeError):
    pass


def check_path(limits, times, xyz):
    """xyz: (N, 3) tool0 positions in the robot base frame, times: (N,) seconds."""
    lo, hi = np.array(limits["workspace_min"]), np.array(limits["workspace_max"])
    outside = np.any((xyz < lo) | (xyz > hi), axis=1)
    if outside.any():
        i = int(np.argmax(outside))
        raise SafetyError("Waypoint %d %s is outside the workspace box %s .. %s"
                          % (i, np.round(xyz[i], 3).tolist(), lo.tolist(), hi.tolist()))
    speed = np.linalg.norm(np.diff(xyz, axis=0), axis=1) / np.diff(times)
    if speed.max() > limits["max_tcp_speed"]:
        raise SafetyError("Tool speed %.2f m/s exceeds the limit of %.2f m/s"
                          % (speed.max(), limits["max_tcp_speed"]))


def check_joints(limits, times, q):
    """q: (N, 6) joint positions in rad."""
    step = np.degrees(np.abs(np.diff(q, axis=0)))
    qd = step / np.diff(times)[:, None]
    if qd.max() > limits["max_joint_speed_deg"]:
        raise SafetyError("Joint speed %.0f deg/s exceeds the limit of %.0f deg/s"
                          % (qd.max(), limits["max_joint_speed_deg"]))
    if step.max() > limits["max_joint_step_deg"]:
        raise SafetyError("Joint jump of %.1f deg between samples (IK branch flip?)" % step.max())


def require_operator(prompt):
    """Ask a human at the keyboard. Refuses when stdin is not an interactive terminal,
    so scripts, pipes and AI agents cannot confirm a motion."""
    if not sys.stdin.isatty():
        raise SafetyError("No interactive terminal: robot motion needs a person at the keyboard.")
    return input(prompt + " Hand on the e-stop. Type 'go' to move: ").strip().lower() == "go"
