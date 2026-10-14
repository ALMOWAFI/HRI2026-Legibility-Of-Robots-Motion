"""Explicit missing integrations; none of these functions silently succeeds.

The repository has no URDF/SRDF, selected group, installed MoveIt version,
planning scene or retiming configuration sufficient to implement these here.
These are contracts for a future reviewed robot integration, not a pipeline.
"""


class MissingRobotIntegration(NotImplementedError):
    pass


def fk_position(q, joint_names):
    """Return xyz from the verified robot model, in the measured goals' frame."""
    raise MissingRobotIntegration("Connect and verify actual robot FK and tool/frame transforms")


def retime_trajectory(candidate, *, robot_state, planning_group):
    """Build fresh time, velocity and acceleration data using confirmed limits."""
    raise MissingRobotIntegration("Configure and verify the installed MoveIt retimer")


def validate_retimed_trajectory(trajectory, *, baseline, planning_scene):
    """Check full path, self/environment collisions, limits, goal and controller interpolation.

    Also recheck endpoint preservation, task orientation, motion budget and local
    deviation after any retimer resampling. Validate against a fresh scene/state.
    """
    raise MissingRobotIntegration("Implement robot-specific validation of the retimed trajectory")


def execute_validated_trajectory(trajectory, *, validation, explicit_authorization):
    """Future separate operator-controlled execution; never called by the prototype."""
    raise MissingRobotIntegration("Hardware execution is intentionally unavailable in this prototype")
