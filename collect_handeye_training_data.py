#!/usr/bin/env python

# Import modules
import numpy as np
import rospy, tf, sys, time
from geometry_msgs.msg import Pose, PoseStamped
from rospy_message_converter import message_converter
import message_filters
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo
import tf2_ros
import tf2_geometry_msgs
import moveit_commander
import moveit_msgs.msg
import moveit_msgs.srv
import geometry_msgs.msg
from cv_bridge import CvBridge, CvBridgeError
from PIL import Image as IM
import cv2

import pyrealsense2 as rs

rospy.init_node('collectPoints',anonymous=True)

# User options (change me)
# --------------- Setup options ---------------
workspace_limits = np.asarray([[0.21, 0.64], [-0.4,0.4], [0.13, 0.62]]) # Cols: min max, Rows: x y z (define workspace limits in robot coordinates)
calib_grid_step = 0.04
# ---------------------------------------------

# Construct 3D calibration grid across workspace
gridspace_x = np.linspace(workspace_limits[0][0], workspace_limits[0][1], 1 + (workspace_limits[0][1] - workspace_limits[0][0])/calib_grid_step)
gridspace_y = np.linspace(workspace_limits[1][0], workspace_limits[1][1], 1 + (workspace_limits[1][1] - workspace_limits[1][0])/calib_grid_step)
gridspace_z = np.linspace(workspace_limits[2][0], workspace_limits[2][1], 1 + (workspace_limits[2][1] - workspace_limits[2][0])/calib_grid_step)
calib_grid_x, calib_grid_y, calib_grid_z = np.meshgrid(gridspace_x, gridspace_y, gridspace_z)
num_calib_grid_pts = calib_grid_x.shape[0]*calib_grid_x.shape[1]*calib_grid_x.shape[2]
calib_grid_x.shape = (num_calib_grid_pts,1)
calib_grid_y.shape = (num_calib_grid_pts,1)
calib_grid_z.shape = (num_calib_grid_pts,1)
calib_grid_pts = np.concatenate((calib_grid_x, calib_grid_y, calib_grid_z), axis=1)

# realsense
points = rs.points()
pipeline= rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.infrared, 1, 1280, 720, rs.format.y8, 30)
config.enable_stream(rs.stream.infrared, 2, 1280, 720, rs.format.y8, 30)
# config.enable_stream(rs.stream.depth,    3,1280, 720, rs.format.z16, 30)
profile = pipeline.start(config)

# TODO: Create move group of MoveIt for motion planning
moveit_commander.roscpp_initialize(sys.argv)
global group, robot, scene
robot = moveit_commander.RobotCommander()
scene = moveit_commander.PlanningSceneInterface()
group = moveit_commander.MoveGroupCommander("manipulator")
group.set_planner_id('RRTConnectkConfigDefault')
group.set_num_planning_attempts(5)
group.set_planning_time(5)
group.set_max_velocity_scaling_factor(0.5)

# TODO: Go to the home pose waiting for picking instruction
tool = PoseStamped()
tool.header.frame_id = "world"
tool.pose.orientation.x = 0
tool.pose.orientation.y = 0
tool.pose.orientation.z = 0
tool.pose.orientation.w = 1

for iter in range(num_calib_grid_pts):
    tool.pose.position.x = calib_grid_pts[iter,0]
    tool.pose.position.y = calib_grid_pts[iter,1]
    tool.pose.position.z = calib_grid_pts[iter,2]
    group.set_pose_target(tool, end_effector_link='tool0') #wrist3_Link
    plan = group.plan()
    print("******************************************************")
    print("tool pose: ",tool)
    raw_input('Press enter to continue: ')
    time.sleep(2)
    group.execute(plan)
    time.sleep(2)
    frames = pipeline.wait_for_frames()
    irL_frame = frames.get_infrared_frame(1)
    irR_frame = frames.get_infrared_frame(2)
    # depth_frame = frames.get_infrared_frame(3)
    image_L = np.asanyarray(irL_frame.get_data())
    image_R = np.asanyarray(irR_frame.get_data())
    cv2.imwrite("image_L_%04d.jpg"%iter, image_L)
    cv2.imwrite("image_R_%04d.jpg"%iter, image_R)
