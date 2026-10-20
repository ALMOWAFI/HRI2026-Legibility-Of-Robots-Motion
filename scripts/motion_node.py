#!/usr/bin/env python3
"""Motion node (UR5e): replays the precomputed reaches, nothing is planned during a trial.

Sub:  /trial_start   (TrialInfo)          reach to target_id with condition = motion type
      /motion_home   (std_msgs/Empty)     bring the robot to the start pose (session start, after a failure)
      /joint_states, /speed_scaling_factor
Pub:  /motion_status (String)             executing / done / failed / home

Per trial: speed slider 100 %, robot at the start pose -> send the stored trajectory ->
'executing' at reach onset (first joint motion in /joint_states, plan §3.4) -> 'done' ->
hold, slow checked return to the start pose -> 'home'. Any failure -> 'failed', and the
robot stays where it is until the experimenter sends /motion_home.

Requests come only from experiment_node, where the experimenter confirms them at the keyboard.
"""
import os
import queue
import threading

import numpy as np
import rospkg
import rospy
from control_msgs.msg import FollowJointTrajectoryGoal
from sensor_msgs.msg import JointState
from std_msgs.msg import Empty, Float64, String
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from hri_legible_motion import trajectories
from hri_legible_motion.msg import TrialInfo
from hri_legible_motion.robot import Robot
from hri_legible_motion.safety import SafetyError

ONSET_SPEED = 0.01        # rad/s, any joint
ONSET_TIMEOUT = 2.0       # s after sending the goal
HOLD_TIME = 1.0           # s at the hover point before returning
HOME_DURATION = 8.0       # s, direct joint move to the start pose (small corrections only)


class MotionNode:
    def __init__(self):
        cfg = rospy.get_param("~")
        self.robot = Robot(cfg)
        self.start = cfg["motion"]["start_joints"]
        self.tolerance = cfg["safety"]["start_tolerance_deg"]
        self.require_scaling = rospy.get_param("~require_speed_scaling", True)

        path = cfg["trajectories"]
        if not os.path.isabs(path):
            path = os.path.join(rospkg.RosPack().get_path("hri_legible_motion"), path)
        names, start, self.reaches, self.meta = trajectories.load(path)
        if any(abs(start[n] - self.start[n]) > 1e-3 for n in self.start):
            raise SafetyError("%s was planned from a different start pose than motion.start_joints. "
                              "Run tools/precompute_trajectories.py again." % path)
        self.joint_names = names
        rospy.loginfo("Loaded %s (hash %s): %s", path, self.meta["hash"], ", ".join(sorted(self.reaches)))

        self.scaling = None
        self.onset = threading.Event()
        self.watch_onset = False
        self.requests = queue.Queue()
        self.status_pub = rospy.Publisher("/motion_status", String, queue_size=10)
        rospy.Subscriber("/trial_start", TrialInfo, lambda m: self.requests.put(("trial", m)))
        rospy.Subscriber("/motion_home", Empty, lambda m: self.requests.put(("home", m)))
        rospy.Subscriber("/speed_scaling_factor", Float64, self.on_scaling)
        rospy.Subscriber("/joint_states", JointState, self.on_joints)
        self.robot.connect()
        self.orientation = self.robot.fk(self.start)[1]

    # --- callbacks -------------------------------------------------------
    def on_scaling(self, msg):
        self.scaling = msg.data

    def on_joints(self, msg):
        if not self.watch_onset:
            return
        if msg.velocity:
            moving = max(abs(v) for v in msg.velocity) > ONSET_SPEED
        else:
            q = dict(zip(msg.name, msg.position))
            moving = self.robot.distance_deg(q, self.start) > 0.1
        if moving:
            self.watch_onset = False
            self.onset.set()

    def status(self, s):
        self.status_pub.publish(String(s))

    # --- requests --------------------------------------------------------
    def go_home(self):
        current = self.robot.current_joints()
        off = self.robot.distance_deg(current, self.start)
        if off > self.tolerance:
            try:
                self.robot.check_move_joints(self.start, HOME_DURATION)
                rospy.loginfo("Moving %.1f deg to the start pose (%.0f s)", off, HOME_DURATION)
                self.robot.move_joints(self.start, HOME_DURATION)
            except SafetyError:
                # Too far for a direct joint move (e.g. stopped mid-reach): checked Cartesian return.
                rospy.loginfo("Returning to the start pose along a checked straight line")
                self.robot.execute(self.robot.plan_return(current, self.orientation))
                self.robot.move_joints(self.start, 2.0)
        self.robot.check_at(self.start, "the start pose")

    def check_scaling(self):
        if self.scaling is None:
            if self.require_scaling:
                raise SafetyError("No /speed_scaling_factor received; is the UR driver running? "
                                  "(set ~require_speed_scaling false only in simulation)")
            return
        if self.scaling < 0.99:
            raise SafetyError("Speed slider at %.0f %%; it must be 100 %% or the reach takes longer "
                              "than planned." % (self.scaling * 100))

    def trajectory(self, key):
        r = self.reaches[key]
        traj = JointTrajectory(joint_names=self.joint_names)
        for t, q, qd in zip(r["times"], r["positions"], r["velocities"]):
            traj.points.append(JointTrajectoryPoint(positions=q.tolist(), velocities=qd.tolist(),
                                                    time_from_start=rospy.Duration(float(t))))
        traj.header.stamp = rospy.Time(0)
        return traj

    def run_trial(self, msg):
        key = trajectories.key(msg.condition, msg.target_id)
        if key not in self.reaches:
            raise SafetyError("No precomputed reach '%s' in the trajectory file" % key)
        self.check_scaling()
        self.robot.check_at(self.start, "the start pose; the stored reach starts there")
        rospy.loginfo("Trial %d: %s", msg.trial_id, key)

        self.onset.clear()
        self.watch_onset = True
        self.robot.client.send_goal(FollowJointTrajectoryGoal(trajectory=self.trajectory(key)))
        if not self.onset.wait(ONSET_TIMEOUT):
            rospy.logwarn("No joint motion seen within %.0f s of sending the reach", ONSET_TIMEOUT)
        self.watch_onset = False
        self.status("executing")
        self.robot.client.wait_for_result()
        r = self.robot.client.get_result()
        if r is None or r.error_code != 0:
            raise SafetyError("Reach failed: %s" % ("no result" if r is None
                                                    else "%d %s" % (r.error_code, r.error_string)))
        self.status("done")

        rospy.sleep(HOLD_TIME)
        self.robot.execute(self.robot.plan_return(self.robot.current_joints(), self.orientation))
        self.robot.move_joints(self.start, 2.0)  # residual correction, a fraction of a degree
        self.robot.check_at(self.start, "the start pose")
        self.status("home")

    def spin(self):
        while not rospy.is_shutdown():
            try:
                kind, msg = self.requests.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                if kind == "home":
                    self.go_home()
                    self.status("home")
                else:
                    self.run_trial(msg)
            except (SafetyError, rospy.ROSException) as e:
                rospy.logerr("SAFETY STOP: %s", e)
                self.status("failed")
            # Requests that arrived while the robot was busy are stale: drop them.
            while not self.requests.empty():
                rospy.logwarn("Dropping request sent while the robot was busy: %s", self.requests.get()[0])


if __name__ == "__main__":
    rospy.init_node("motion_node")
    MotionNode().spin()
