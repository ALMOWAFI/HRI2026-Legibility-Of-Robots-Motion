"""Cubes and the pointer on the flange in the MoveIt planning scene (collision checking) and
as RViz markers on /visualization_marker_array (shown by the lab RViz config, whose
MotionPlanning display is off).
"""
import math

import rospy
from geometry_msgs.msg import Pose
from moveit_msgs.msg import AttachedCollisionObject, CollisionObject, ObjectColor, PlanningScene
from moveit_msgs.srv import ApplyPlanningScene
from shape_msgs.msg import SolidPrimitive
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray

_markers = None
# Robot links the pointer touches where it is mounted (not collisions).
TOOL_TOUCH_LINKS = ["arm_ee_adapter", "arm_wrist_3_link", "arm_flange_safety"]


def block_pose(b):
    pose = Pose()
    pose.position.x, pose.position.y, pose.position.z = b["position"]
    half = math.radians(b.get("yaw", 0.0)) / 2
    pose.orientation.z, pose.orientation.w = math.sin(half), math.cos(half)
    return pose


def _apply(scene):
    try:
        rospy.wait_for_service("/apply_planning_scene", timeout=2)
    except rospy.ROSException:
        print("Waiting for MoveIt (move_group) to come up...")
        rospy.wait_for_service("/apply_planning_scene")
    return rospy.ServiceProxy("/apply_planning_scene", ApplyPlanningScene)(scene).success


def _publish(markers):
    global _markers
    if _markers is None:
        _markers = rospy.Publisher("/visualization_marker_array", MarkerArray, queue_size=1, latch=True)
    deadline = rospy.Time.now() + rospy.Duration(3)
    while _markers.get_num_connections() == 0 and rospy.Time.now() < deadline:
        rospy.sleep(0.1)
    _markers.publish(markers)
    rospy.sleep(0.5)  # let it go out before a short-lived node exits
    return _markers.get_num_connections()


def add_tool(cfg, scene, markers):
    """The pointer (cfg["tool"]): a cylinder along the flange's z axis, attached to the
    end-effector link so MoveIt checks it for collisions and it moves with the arm in RViz."""
    t, link = cfg["tool"], cfg["motion"]["ee_link"]
    pose = Pose()
    pose.position.z = t["length"] / 2
    pose.orientation.w = 1.0
    color = ColorRGBA(*t["rgba"])
    obj = CollisionObject(id="pointer", operation=CollisionObject.ADD)
    obj.header.frame_id = link
    obj.primitives = [SolidPrimitive(type=SolidPrimitive.CYLINDER, dimensions=[t["length"], t["radius"]])]
    obj.primitive_poses = [pose]
    scene.robot_state.is_diff = True
    scene.robot_state.attached_collision_objects.append(
        AttachedCollisionObject(link_name=link, object=obj, touch_links=[link] + TOOL_TOUCH_LINKS))
    scene.object_colors.append(ObjectColor(id=obj.id, color=color))
    m = Marker(ns="hri_tool", id=0, type=Marker.CYLINDER, action=Marker.ADD, pose=pose, color=color)
    m.header.frame_id = link
    m.scale.x = m.scale.y = 2 * t["radius"]
    m.scale.z = t["length"]
    markers.markers.append(m)


def set_cubes(cfg, shown):
    """Show exactly the blocks in `shown` (a subset of cfg["blocks"]); remove the others.
    The pointer is always (re)attached. Returns (scene_ok, number of RViz listeners)."""
    names = {b["name"] for b in shown}
    scene = PlanningScene(is_diff=True)
    markers = MarkerArray()
    add_tool(cfg, scene, markers)
    for b in cfg["blocks"]:
        on = b["name"] in names
        obj = CollisionObject(id="cube_%s" % b["name"])
        obj.header.frame_id = cfg["frame_id"]
        m = Marker(ns="hri_cubes", id=b["id"], type=Marker.CUBE)
        m.header.frame_id = cfg["frame_id"]
        if on:
            obj.operation = CollisionObject.ADD
            obj.primitives = [SolidPrimitive(type=SolidPrimitive.BOX, dimensions=cfg["block_size"])]
            obj.primitive_poses = [block_pose(b)]
            scene.object_colors.append(ObjectColor(id=obj.id, color=ColorRGBA(*b["rgba"])))
            m.action = Marker.ADD
            m.pose = block_pose(b)
            m.scale.x, m.scale.y, m.scale.z = cfg["block_size"]
            m.color = ColorRGBA(*b["rgba"])
        else:
            obj.operation = CollisionObject.REMOVE
            m.action = Marker.DELETE
        scene.world.collision_objects.append(obj)
        markers.markers.append(m)
    return _apply(scene), _publish(markers)
