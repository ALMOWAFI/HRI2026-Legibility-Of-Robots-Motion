#!/usr/bin/env python3
"""Motion / planning node (UR5e).

Sub:  /trial_start   (TrialInfo)  target + condition for the trial
Pub:  /motion_status (String)     executing / done / failed / home

Pipeline: fixed block coordinates (config) + target
  -> baseline path -> legibility optimization -> waypoints
  -> MoveIt IK (no re-planning) -> joint trajectory -> execute on UR5e
"""
# TODO
