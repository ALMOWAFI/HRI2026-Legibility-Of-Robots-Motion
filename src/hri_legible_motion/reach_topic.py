"""Reaches on topics, as moveit_msgs/CartesianTrajectory (standard type, no catkin build).

/hri/predictable_reach   test_motion.py -> legibility_optimizer_node.py
/hri/legible_reach       legibility_optimizer_node.py -> test_motion.py

header.frame_id = arm_base_link, tracked_frame = the end-effector link, one point per
sample (tool position + orientation, time_from_start). The reply keeps the request's
header.stamp, so a reply belongs to exactly one request. A reply without points means
the optimizer failed. The target is the cube whose hover point the reach ends at.
"""
import numpy as np
import rospy
from moveit_msgs.msg import CartesianTrajectory, CartesianTrajectoryPoint

PREDICTABLE = "/hri/predictable_reach"
LEGIBLE = "/hri/legible_reach"


def to_msg(times, xyz, orientation, frame_id, tracked_frame, stamp=None):
    msg = CartesianTrajectory()
    msg.header.frame_id, msg.header.stamp = frame_id, stamp or rospy.Time.now()
    msg.tracked_frame = tracked_frame
    for t, p in zip(times, xyz):
        pt = CartesianTrajectoryPoint(time_from_start=rospy.Duration(float(t)))
        pt.point.pose.position.x, pt.point.pose.position.y, pt.point.pose.position.z = map(float, p)
        pt.point.pose.orientation = orientation
        msg.points.append(pt)
    return msg


def from_msg(msg):
    """Returns (times, xyz) as numpy arrays."""
    times = np.array([p.time_from_start.to_sec() for p in msg.points])
    xyz = np.array([[p.point.pose.position.x, p.point.pose.position.y, p.point.pose.position.z]
                    for p in msg.points]).reshape(-1, 3)
    return times, xyz


def hover_points(cfg):
    """Hover point (reach end) above every cube, in block-id order."""
    m = cfg["motion"]
    z = m["touch_z"] + m["hover_height"]
    return np.array([[b["position"][0], b["position"][1], z] for b in cfg["blocks"]])


def target_of(cfg, end, tolerance=0.01):
    """Block id whose hover point `end` is at (within `tolerance` m), else None."""
    d = np.linalg.norm(hover_points(cfg) - np.asarray(end, float), axis=1)
    return int(d.argmin()) if d.min() <= tolerance else None
