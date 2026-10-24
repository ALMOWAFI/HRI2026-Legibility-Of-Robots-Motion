"""Reaches on topics, as moveit_msgs/CartesianTrajectory (standard type, no catkin build).

/hri/predictable_reach   test_motion.py -> legibility_optimizer_node.py
/hri/legible_reach       legibility_optimizer_node.py -> test_motion.py

header.frame_id = arm_base_link, tracked_frame = the end-effector link, one point per
sample (tool position + orientation, time_from_start). The reply keeps the request's
header.stamp, so a reply belongs to exactly one request. A reply without points means
the optimizer failed. The target is the cube whose hover point the reach ends at.
"""
import copy

import numpy as np
import rospy
from moveit_msgs.msg import CartesianTrajectory, CartesianTrajectoryPoint

PREDICTABLE = "/hri/predictable_reach"
LEGIBLE = "/hri/legible_reach"


def to_msg(times, xyz, orientation, frame_id, tracked_frame, stamp=None):
    times, xyz = validate_samples(times, xyz)
    _quaternion(orientation)
    msg = CartesianTrajectory()
    msg.header.frame_id = frame_id
    msg.header.stamp = rospy.Time.now() if stamp is None else stamp
    msg.tracked_frame = tracked_frame
    for t, p in zip(times, xyz):
        pt = CartesianTrajectoryPoint(time_from_start=rospy.Duration(float(t)))
        pt.point.pose.position.x, pt.point.pose.position.y, pt.point.pose.position.z = map(float, p)
        pt.point.pose.orientation = copy.deepcopy(orientation)
        msg.points.append(pt)
    return msg


def from_msg(msg):
    """Returns (times, xyz) as numpy arrays."""
    times = np.array([p.time_from_start.to_sec() for p in msg.points])
    xyz = np.array([[p.point.pose.position.x, p.point.pose.position.y, p.point.pose.position.z]
                    for p in msg.points]).reshape(-1, 3)
    return times, xyz


def validate_samples(times, xyz):
    """Validate the numerical part of a reach; this does not establish robot validity."""
    times, xyz = np.asarray(times, dtype=float), np.asarray(xyz, dtype=float)
    if times.ndim != 1 or len(times) < 3 or xyz.shape != (len(times), 3):
        raise ValueError("Reach needs >= 3 timestamps and matching (N, 3) positions.")
    if not np.isfinite(times).all() or not np.isfinite(xyz).all():
        raise ValueError("Reach timestamps and positions must be finite.")
    if times[0] < 0 or np.any(np.diff(times) <= 0):
        raise ValueError("Reach timestamps must be nonnegative and strictly increasing.")
    return times, xyz


def _quaternion(orientation):
    q = np.array([orientation.x, orientation.y, orientation.z, orientation.w], dtype=float)
    if not np.isfinite(q).all() or not np.isclose(np.linalg.norm(q), 1.0, atol=1e-6, rtol=0):
        raise ValueError("Reach orientation must be a finite normalized quaternion.")
    return q


def validate_message(msg, frame_id, tracked_frame):
    """Require the configured frames and one fixed, normalized tool orientation."""
    if msg.header.frame_id != frame_id or msg.tracked_frame != tracked_frame:
        raise ValueError("Reach frame/link mismatch: expected %s / %s, got %s / %s."
                         % (frame_id, tracked_frame, msg.header.frame_id, msg.tracked_frame))
    times, xyz = validate_samples(*from_msg(msg))
    q0 = _quaternion(msg.points[0].point.pose.orientation)
    for point in msg.points[1:]:
        q = _quaternion(point.point.pose.orientation)
        if min(np.linalg.norm(q - q0), np.linalg.norm(q + q0)) > 1e-6:
            raise ValueError("Reach tool orientation must stay fixed.")
    return times, xyz


def validate_response(request, response):
    """Reject a response that changes the request's timing, endpoints, frames or orientation."""
    expected_t, expected_xyz = validate_message(request, request.header.frame_id, request.tracked_frame)
    times, xyz = validate_message(response, request.header.frame_id, request.tracked_frame)
    if response.header.stamp != request.header.stamp:
        raise ValueError("Optimizer response belongs to another request.")
    if len(response.points) != len(request.points) or any(
            a.time_from_start != b.time_from_start for a, b in zip(request.points, response.points)):
        raise ValueError("Optimizer response changed the timestamp sequence.")
    if not np.array_equal(times, expected_t) or not np.array_equal(xyz[[0, -1]], expected_xyz[[0, -1]]):
        raise ValueError("Optimizer response changed timing or endpoints.")
    q0 = _quaternion(request.points[0].point.pose.orientation)
    q1 = _quaternion(response.points[0].point.pose.orientation)
    if min(np.linalg.norm(q0 - q1), np.linalg.norm(q0 + q1)) > 1e-6:
        raise ValueError("Optimizer response changed the tool orientation.")
    if np.max(np.abs(xyz[:, 2] - expected_xyz[0, 2])) > 1e-3:
        raise ValueError("Optimizer response leaves the fixed-height plane.")
    return times, xyz


def reply_message(request, times=None, xyz=None):
    """Copy the exact ROS timing/orientation; an empty reply explicitly reports failure."""
    reply = copy.deepcopy(request)
    if times is None:
        reply.points = []
        return reply
    times, xyz = validate_samples(times, xyz)
    request_times, _ = from_msg(request)
    if not np.array_equal(times, request_times):
        raise ValueError("Optimizer changed the timestamp sequence.")
    for point, p in zip(reply.points, xyz):
        point.point.pose.position.x, point.point.pose.position.y, point.point.pose.position.z = map(float, p)
    validate_response(request, reply)
    return reply


def hover_points(cfg):
    """Hover point (reach end) above every cube, in configuration row order."""
    m = cfg["motion"]
    z = m["touch_z"] + m["hover_height"]
    return np.array([[b["position"][0], b["position"][1], z] for b in cfg["blocks"]])


def target_of(cfg, end, tolerance=0.01):
    """Configuration row whose hover point is at `end`, else None (not a block id)."""
    d = np.linalg.norm(hover_points(cfg) - np.asarray(end, float), axis=1)
    return int(d.argmin()) if d.min() <= tolerance else None
