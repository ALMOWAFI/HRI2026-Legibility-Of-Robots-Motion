#!/usr/bin/env python3
"""Experiment / game node.

Sub:  /joy            (sensor_msgs/Joy)  participant choice
      /motion_status  (String)           trial timing
Pub:  /trial_start    (TrialInfo)
      /choice         (ChoiceEvent)

Trial list, choice logging with timestamps, trial data + score -> CSV.
Does not touch the robot trajectory.
"""
# TODO
