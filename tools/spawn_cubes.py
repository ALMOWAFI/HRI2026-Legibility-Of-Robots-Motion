#!/usr/bin/env python3
"""Show the four coloured cubes from config/experiment.yaml in the MoveIt planning scene
and as RViz markers (no robot motion). To place the real cubes, use place_blocks.py.

usage: spawn_cubes.py [--remove] [config.yaml]

Re-run after restarting the bringup (MoveIt's scene starts empty) or RViz (markers are lost).
"""
import os
import sys

import rospy
import yaml

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))  # works without catkin build
from hri_legible_motion.scene import set_cubes  # noqa: E402


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    remove = "--remove" in sys.argv
    with open(args[0] if args else os.path.join(ROOT, "config", "experiment.yaml")) as f:
        cfg = yaml.safe_load(f)
    rospy.init_node("spawn_cubes", anonymous=True)
    ok, viewers = set_cubes(cfg, [] if remove else cfg["blocks"])
    print("%s %d cubes in the MoveIt scene: %s" % ("Removed" if remove else "Added", len(cfg["blocks"]),
                                                    "ok" if ok else "FAILED"))
    print("Markers on /visualization_marker_array: %s" % (
        "sent to RViz" if viewers else "no RViz listening (start RViz, then re-run)"))


if __name__ == "__main__":
    main()
