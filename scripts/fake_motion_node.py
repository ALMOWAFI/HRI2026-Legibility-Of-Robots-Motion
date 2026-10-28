#!/usr/bin/env python3
"""Stand-in for motion_node.py in dry runs: answers like the motion node, moves nothing.

Sub:  /trial_start (TrialInfo), /motion_home (Empty)
Pub:  /motion_status (String)  executing / done / home with the real timing
      (onset after ONSET s, reach motion.duration, return RETURN s)

study.launch dry_run:=true starts it instead of motion_node.py, so the whole game
(screens, sounds, keys, logging) can be tested without the robot, MoveIt or the UR driver.
"""
import queue

import rospy
from std_msgs.msg import Empty, String

from hri_legible_motion.msg import TrialInfo

ONSET = 0.3     # s from the request to the first joint motion (measured on the UR5e: ~0.2-0.3)
RETURN = 4.0    # s hold + return to the start pose


def main():
    rospy.init_node("motion_node")
    duration = float(rospy.get_param("~motion/duration"))
    pub = rospy.Publisher("/motion_status", String, queue_size=10)
    requests = queue.Queue()
    rospy.Subscriber("/trial_start", TrialInfo, lambda m: requests.put(m))
    rospy.Subscriber("/motion_home", Empty, lambda m: requests.put(None))
    rospy.logwarn("FAKE motion node: the robot does not move (dry run).")
    while not rospy.is_shutdown():
        try:
            msg = requests.get(timeout=0.2)
        except queue.Empty:
            continue
        if msg is None:
            rospy.sleep(1.0)
        else:
            rospy.loginfo("Trial %d: %s to block %d (fake)", msg.trial_id, msg.condition, msg.target_id)
            rospy.sleep(ONSET)
            pub.publish(String("executing"))
            rospy.sleep(duration)
            pub.publish(String("done"))
            rospy.sleep(RETURN)
        pub.publish(String("home"))


if __name__ == "__main__":
    main()
