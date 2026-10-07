#!/usr/bin/env python3
"""Send the UR5e to a recorded joint pose and compare the reached TCP with the recording.

usage: goto_pose.py <B1|B2|B3|B4> [seconds]
"""
import sys

import actionlib
import rospy
import tf2_ros
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectoryPoint

NAMES = ["arm_elbow_joint", "arm_shoulder_lift_joint", "arm_shoulder_pan_joint",
         "arm_wrist_1_joint", "arm_wrist_2_joint", "arm_wrist_3_joint"]

# Recorded on 2026-10-07 (joint order as in NAMES), plus the TCP tf_echo gave.
POSES = {
    "B1": ([-0.03644949197769165, -3.1973921261229457, -2.8275001684771937, -1.411511705522873, 1.5201799869537354, -0.13190061250795537], (0.919, 0.153, 0.011)),
    "B2": ([-0.036424312740564346, -3.2043520412840785, -3.0251334349261683, -1.4112602484277268, 1.5201992988586426, -0.13188344637026006], (0.930, -0.031, 0.005)),
    "B3": ([-0.03622959181666374, -3.205991884271139, -3.218473259602682, -1.4111372840455552, 1.5201785564422607, -0.13189250627626592], (0.907, -0.209, 0.003)),
    "B4": ([-0.035736870020627975, -3.1888772449889125, -3.396393124257223, -1.1794453424266358, 1.5202162265777588, -0.13191491762270147], (0.876, -0.371, 0.046)),
}


def find_action():
    topics = [t for t, _ in rospy.get_published_topics() if t.endswith("follow_joint_trajectory/status")]
    topics.sort(key=lambda t: "scaled_pos" not in t)  # prefer the scaled controller
    if not topics:
        sys.exit("No follow_joint_trajectory action found. Is the driver up and External Control playing?")
    return topics[0][: -len("/status")]


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in POSES:
        sys.exit(__doc__)
    name, duration = sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 10.0
    target, recorded_tcp = POSES[name]
    rospy.init_node("goto_pose", anonymous=True)

    js = rospy.wait_for_message("/joint_states", JointState, timeout=5)
    current = dict(zip(js.name, js.position))
    missing = [n for n in NAMES if n not in current]
    if missing:
        sys.exit("Joint names differ from the recording: %s" % missing)
    deltas = {n: t - current[n] for n, t in zip(NAMES, target)}
    print("Moving to %s in %.0f s. Joint changes (deg):" % (name, duration))
    for n, dq in deltas.items():
        print("  %-26s %+7.1f" % (n, dq * 57.2958))
    if input("Hand on the e-stop. Go? [y/N] ").strip().lower() != "y":
        return

    action = find_action()
    print("Using", action)
    client = actionlib.SimpleActionClient(action, FollowJointTrajectoryAction)
    if not client.wait_for_server(rospy.Duration(5)):
        sys.exit("Action server not responding.")
    goal = FollowJointTrajectoryGoal()
    goal.trajectory.joint_names = NAMES
    goal.trajectory.points = [JointTrajectoryPoint(positions=target, velocities=[0.0] * 6,
                                                   time_from_start=rospy.Duration(duration))]
    client.send_goal(goal)
    client.wait_for_result()
    print("Result:", client.get_result())

    buf = tf2_ros.Buffer()
    tf2_ros.TransformListener(buf)
    t = buf.lookup_transform("arm_base_link", "arm_tool0", rospy.Time(0), rospy.Duration(3)).transform.translation
    err = ((t.x - recorded_tcp[0]) ** 2 + (t.y - recorded_tcp[1]) ** 2 + (t.z - recorded_tcp[2]) ** 2) ** 0.5
    print("Reached  [%.3f, %.3f, %.3f]" % (t.x, t.y, t.z))
    print("Recorded [%.3f, %.3f, %.3f]   error %.1f mm" % (recorded_tcp + (err * 1000,)))


if __name__ == "__main__":
    main()
