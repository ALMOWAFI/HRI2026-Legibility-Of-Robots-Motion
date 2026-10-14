"""Cubes in the MoveIt planning scene (collision checking) and as RViz markers on
/visualization_marker_array (shown by the lab RViz config, whose MotionPlanning display is off).
"""
import rospy
from geometry_msgs.msg import Pose
from moveit_msgs.msg import CollisionObject, ObjectColor, PlanningScene
from moveit_msgs.srv import ApplyPlanningScene
from shape_msgs.msg import SolidPrimitive
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray

_markers = None


def block_pose(b):
    pose = Pose()
    pose.position.x, pose.position.y, pose.position.z = b["position"]
    pose.orientation.w = 1.0
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


def set_cubes(cfg, shown):
    """Show exactly the blocks in `shown` (a subset of cfg["blocks"]); remove the others.
    Returns (scene_ok, number of RViz listeners)."""
    names = {b["name"] for b in shown}
    scene = PlanningScene(is_diff=True)
    markers = MarkerArray()
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
