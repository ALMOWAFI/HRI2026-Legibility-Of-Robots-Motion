"""UR5e helpers shared by the tools: FK, Cartesian planning with our own timing, execution.

Every trajectory returned by plan() / sent by move_joints() has passed
hri_legible_motion.safety. Execution goes to the joint trajectory controller directly.
"""
import actionlib
import numpy as np
import rospy
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from geometry_msgs.msg import Pose
from moveit_msgs.msg import DisplayTrajectory, RobotState, RobotTrajectory
from moveit_msgs.srv import GetCartesianPath, GetPositionFK
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from hri_legible_motion.safety import SafetyError, check_direct_move, check_joints, check_path, require_operator


def joint_state(joints):
    js = JointState()
    js.name, js.position = list(joints.keys()), list(joints.values())
    return js


def end_joints(traj):
    return dict(zip(traj.joint_names, traj.points[-1].positions))


class Robot:
    def __init__(self, cfg):
        self.cfg = cfg
        self.m = cfg["motion"]
        self.limits = cfg["safety"]
        self.frame = cfg["frame_id"]
        self.client = None
        self._fk = rospy.ServiceProxy("/compute_fk", GetPositionFK)
        self._cartesian = rospy.ServiceProxy("/compute_cartesian_path", GetCartesianPath)
        self._display = rospy.Publisher("/move_group/display_planned_path", DisplayTrajectory,
                                        queue_size=1, latch=True)

    def fk(self, joints):
        """Tool position (np.array xyz) and orientation for a joint dict."""
        res = self._fk(header=rospy.Header(frame_id=self.frame), fk_link_names=[self.m["ee_link"]],
                       robot_state=RobotState(joint_state=joint_state(joints)))
        p = res.pose_stamped[0].pose
        return np.array([p.position.x, p.position.y, p.position.z]), p.orientation

    def current_joints(self):
        js = rospy.wait_for_message("/joint_states", JointState, timeout=5)
        return dict(zip(js.name, js.position))

    def distance_deg(self, a, b):
        return float(np.degrees(max(abs(a[n] - b[n]) for n in b)))

    def plan(self, start, times, xyz, orientation):
        """Joint trajectory through the Cartesian samples xyz at the given times, starting at
        joint dict `start` (whose FK must be xyz[0]). Raises SafetyError if anything is off."""
        check_path(self.limits, times, xyz)
        waypoints = []
        for p in xyz[1:]:
            w = Pose()
            w.position.x, w.position.y, w.position.z = p
            w.orientation = orientation
            waypoints.append(w)
        res = self._cartesian(header=rospy.Header(frame_id=self.frame),
                              start_state=RobotState(joint_state=joint_state(start)),
                              group_name=self.m["planning_group"], link_name=self.m["ee_link"],
                              waypoints=waypoints, max_step=1.0, jump_threshold=0.0, avoid_collisions=True)
        pts = res.solution.joint_trajectory.points
        if res.fraction < 1.0 or len(pts) != len(times):
            raise SafetyError("Cartesian path failed: %.0f %% planned, %d/%d points (error %d)"
                              % (res.fraction * 100, len(pts), len(times), res.error_code.val))
        q = np.array([p.positions for p in pts])
        check_joints(self.limits, times, q)
        qd = np.gradient(q, times, axis=0)
        qd[0] = qd[-1] = 0.0
        traj = JointTrajectory(joint_names=res.solution.joint_trajectory.joint_names)
        for t, qi, qdi in zip(times, q, qd):
            traj.points.append(JointTrajectoryPoint(positions=qi.tolist(), velocities=qdi.tolist(),
                                                    time_from_start=rospy.Duration(t)))
        return traj

    def plan_return(self, start, orientation, lift_to=None):
        """From joint dict `start` back to the start pose: optional lift to z = lift_to, then a
        straight line to the start tool position (slow, stop at corners). The end joints are
        checked to be within a small direct move of motion.start_joints."""
        from hri_legible_motion.legibility import stop_and_go
        home = self.m["start_joints"]
        p, goal = self.fk(start)[0], self.fk(home)[0]
        points = [p] + ([[p[0], p[1], max(p[2], lift_to)]] if lift_to else []) + [goal]
        times, xyz = stop_and_go(points, self.cfg["setup"]["speed"])
        traj = self.plan(start, times, xyz, orientation)
        end = end_joints(traj)
        names = list(home)
        check_direct_move(self.limits, names, [end[n] for n in names], [home[n] for n in names], 2.0)
        return traj

    def show(self, start, trajs):
        """Preview in RViz (MotionPlanning > Planned Path)."""
        rospy.sleep(0.5)
        self._display.publish(DisplayTrajectory(
            trajectory_start=RobotState(joint_state=joint_state(start)),
            trajectory=[RobotTrajectory(joint_trajectory=t) for t in trajs]))

    def connect(self):
        self.client = actionlib.SimpleActionClient(self.m["action"], FollowJointTrajectoryAction)
        if not self.client.wait_for_server(rospy.Duration(5)):
            raise SafetyError("Action server not responding. Is the robot program running?")

    def execute(self, traj):
        traj.header.stamp = rospy.Time(0)
        self.client.send_goal(FollowJointTrajectoryGoal(trajectory=traj))
        self.client.wait_for_result()
        r = self.client.get_result()
        if r is None or r.error_code != 0:
            raise SafetyError("Execution failed: %s" % ("no result" if r is None
                                                       else "%d %s" % (r.error_code, r.error_string)))

    def check_move_joints(self, target, duration):
        """Raise SafetyError if move_joints(target, duration) would not be allowed from here."""
        current = self.current_joints()
        names = list(target.keys())
        check_direct_move(self.limits, names, [current[n] for n in names], [target[n] for n in names], duration)

    def move_joints(self, target, duration):
        """Single-segment joint move from the current position: small corrections only
        (size and speed checked), never a configuration change."""
        self.check_move_joints(target, duration)
        names = list(target.keys())
        traj = JointTrajectory(joint_names=names)
        traj.points = [JointTrajectoryPoint(positions=[target[n] for n in names], velocities=[0.0] * len(names),
                                            time_from_start=rospy.Duration(duration))]
        self.execute(traj)

    def go_to_start(self, orientation, lift_to, duration=8.0):
        """Bring the robot to motion.start_joints after the operator types 'go'. Small offsets:
        one direct joint move. Too far for that (e.g. another arm configuration): lift to
        z = lift_to, then the checked straight line of plan_return(). False if declined."""
        start = self.m["start_joints"]
        off = self.distance_deg(self.current_joints(), start)
        if off <= self.limits["start_tolerance_deg"]:
            return True
        try:
            self.check_move_joints(start, duration)
        except SafetyError:
            current = self.current_joints()
            traj = self.plan_return(current, orientation, lift_to=lift_to)
            self.show(current, [traj])
            if not require_operator("Robot is %.0f deg from the start pose: lift to z = %.2f, then a straight "
                                    "line there (%.0f s, preview in RViz)?"
                                    % (off, lift_to, traj.points[-1].time_from_start.to_sec())):
                return False
            self.execute(traj)
            self.move_joints(start, 2.0)  # residual correction, a fraction of a degree
            return True
        if not require_operator("Robot is %.0f deg from the start pose. Move there in %.0f s?" % (off, duration)):
            return False
        self.move_joints(start, duration)
        return True

    def check_at(self, joints, what):
        off = self.distance_deg(self.current_joints(), joints)
        if off > self.limits["start_tolerance_deg"]:
            raise SafetyError("Robot is %.1f deg from %s." % (off, what))
