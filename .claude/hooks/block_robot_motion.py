#!/usr/bin/env python3
"""Claude Code PreToolUse hook: Claude must never move the UR5e.

Blocks Bash commands that could move the robot or hand control to the driver.
Humans run those themselves, at the robot, with a hand on the e-stop.
Exit 2 = blocked (stderr is shown to Claude). Anything else = allowed.
"""
import json
import os
import re
import shlex
import sys

# Robot-moving calls / topics / services. Checked against the command text (covers
# python -c and heredocs) and against the contents of every script the command runs.
CODE_PATTERNS = [
    r"--execute\b",
    r"\bresend_robot_program\b",
    r"dashboard/(play|brake_release|power_on|unlock_protective_stop|load_program|restart_safety|raw_request|close_safety_popup)",
    r"\bswitch_controller\b",
    r"\brostopic\s+pub\b",
    r"\b(roslaunch|rosrun)\s+(ur_robot_driver|cu_ur5e_bringup|hri_legible_motion)\b",
    r"FollowJointTrajectory(Action|Goal)",
    r"ExecuteTrajectory",
    r"follow_joint_trajectory/goal",
    r"(execute_trajectory|move_group|sequence_move_group|pickup|place)/goal",
    r"\bmoveit_by_name\b",
    r"/motion/target\b",
    r"(twist_controller|scaled_pos_joint_traj_controller)/command|script_command",
    r"MoveGroupCommander",
    r"\.send_goal\s*\(",
    r"hri_legible_motion\.robot\b|from\s+hri_legible_motion\s+import\s+robot\b",  # Robot.execute
    r"\.(execute|move_joints)\s*\(",
]
# Scripts that must never be run by Claude, by file name.
BLOCKED_SCRIPTS = re.compile(
    r"^(goto_pose\.py|motion_node\.py|experiment_node\.py|move_\w+\.sh|demo_videos\.sh|launch_hardware\.sh|"
    r"trajectory_controller\.sh|twist_controller\.sh|resend_robot_program\.sh)$")
# Contain motion code, but only move the robot with --execute (caught above).
PREVIEW_SAFE = {"test_motion.py", "place_blocks.py"}
INTERPRETERS = {"python", "python3", "bash", "sh", "source", ".", "rosrun", "exec"}
PREFIXES = {"sudo", "timeout", "env", "nohup", "time", "stdbuf"}

MESSAGE = (
    "BLOCKED by .claude/hooks/block_robot_motion.py: this command could move the UR5e "
    "or enable its controller ({}). Claude must never run the robot. Give the user the "
    "exact command to run themselves, at the robot, with a hand on the e-stop."
)


def code_hit(text):
    for p in CODE_PATTERNS:
        if re.search(p, text):
            return p
    return None


def executed_files(command):
    """Files the command runs: `python3 x.py`, `bash x.sh`, `./x`, `~/scripts/x.sh`, `rosrun pkg x.py`."""
    for segment in re.split(r"&&|\|\||[;|&\n()`]|\$\(", command):
        try:
            tokens = shlex.split(segment, comments=True)
        except ValueError:
            tokens = segment.split()
        while tokens and (tokens[0] in PREFIXES or re.match(r"^\w+=", tokens[0])
                          or re.match(r"^-|^\d+[smh]?$", tokens[0])):
            tokens.pop(0)
        if not tokens:
            continue
        candidates = [tokens[0]]
        if os.path.basename(tokens[0]) in INTERPRETERS:
            candidates = [t for t in tokens[1:] if not t.startswith("-")][:2]  # rosrun pkg file
            if not candidates or tokens[1:2] == ["-"]:
                # Interpreter reading stdin (`cat x.py | python3 -`): scan every file named anywhere.
                candidates = re.findall(r"[\w./~-]+", command)
        for c in candidates:
            yield os.path.expanduser(c)


def main():
    command = json.load(sys.stdin).get("tool_input", {}).get("command", "")
    hit = code_hit(command)
    if hit:
        print(MESSAGE.format("matches " + hit), file=sys.stderr)
        sys.exit(2)
    for path in executed_files(command):
        name = os.path.basename(path)
        if BLOCKED_SCRIPTS.match(name):
            print(MESSAGE.format("runs " + name), file=sys.stderr)
            sys.exit(2)
        if name in PREVIEW_SAFE or not os.path.isfile(path) or os.path.getsize(path) > 2_000_000:
            continue
        if os.path.abspath(path) == os.path.abspath(__file__):
            continue
        with open(path, errors="ignore") as f:
            hit = code_hit(f.read())
        if hit:
            print(MESSAGE.format("%s contains %s" % (name, hit)), file=sys.stderr)
            sys.exit(2)


if __name__ == "__main__":
    main()
