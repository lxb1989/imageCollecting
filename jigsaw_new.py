#!/usr/bin/env python

# Import modules
import sys
sys.path.append('/home/bionicdl-saber/Documents/code/SSD-Tensorflow/notebooks/')
sys.path.append('/home/bionicdl-saber/Documents/code/SSD-Tensorflow/')
sys.path.append('/home/bionicdl-saber/catkin_ws/src/universal_robot/ur_modern_driver/')


import numpy as np
import rospy, tf, sys, time
from geometry_msgs.msg import Pose, PoseStamped
import message_filters
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import CompressedImage
import tf2_ros
import tf2_geometry_msgs
import moveit_commander
import moveit_msgs.msg
import moveit_msgs.srv
import geometry_msgs.msg
from io_interface import *
from PIL import Image as IM
import cv2
import pyrealsense2 as rs
import math
import os
import math
import random
import numpy as np
import tensorflow as tf
import cv2

slim = tf.contrib.slim
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


from nets import ssd_vgg_300, ssd_common, np_methods
from preprocessing import ssd_vgg_preprocessing
# from notebooks import visualization
import visualization
import matplotlib
matplotlib.use('Agg')

###===============================================================================###
# Main image processing routine.
def process_image(img, select_threshold=0.5, nms_threshold=.45, net_shape=(300, 300)):
    # Run SSD network.
    rimg, rpredictions, rlocalisations, rbbox_img = isess.run([image_4d, predictions, localisations, bbox_img],
                                                              feed_dict={img_input: img})

    # Get classes and bboxes from the net outputs.
    rclasses, rscores, rbboxes = np_methods.ssd_bboxes_select(
            rpredictions, rlocalisations, ssd_anchors,
            select_threshold=select_threshold, img_shape=net_shape, num_classes=21, decode=True)

    rbboxes = np_methods.bboxes_clip(rbbox_img, rbboxes)
    rclasses, rscores, rbboxes = np_methods.bboxes_sort(rclasses, rscores, rbboxes, top_k=400)
    rclasses, rscores, rbboxes = np_methods.bboxes_nms(rclasses, rscores, rbboxes, nms_threshold=nms_threshold)
    # Resize bboxes to original image shape. Note: useless for Resize.WARP!
    rbboxes = np_methods.bboxes_resize(rbbox_img, rbboxes)
    return rclasses, rscores, rbboxes
###===================================================================================###

#object detection
def compute_minRect(img):
    src_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(9,9))
    cl1 = clahe.apply(src_gray)
    canny_output = cv2.Canny(cl1, 90,180,5)
    _, contours, _ = cv2.findContours(canny_output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    minRect = [None]*len(contours)
    for i, c in enumerate(contours):
        minRect[i] = cv2.minAreaRect(c)
    return minRect

def non_maximum_suppression(img,minRect):
    rect_candidate = []
    for i in range(len(minRect)):
        if(minRect[i][1][0]>60 or minRect[i][1][0]<40 or minRect[i][1][1]>60 or minRect[i][1][1]<40):
            continue
        rect_candidate.append(minRect[i])

    final_rect = []
    for i in range(len(rect_candidate)):
        temporary_num = 0
        for j in range(len(rect_candidate)):
            if (j == i):
                continue
            dx = abs(rect_candidate[i][0][0]-rect_candidate[j][0][0])
            dy = abs(rect_candidate[i][0][1]-rect_candidate[j][0][1])
            area1 = rect_candidate[i][1][0]*rect_candidate[i][1][1]
            area2 = rect_candidate[j][1][0]*rect_candidate[j][1][1]
            if(dx < 10 and dy <10):
                if(area1 < area2):
                    temporary_num = 1
                    break
        if(temporary_num == 0):
            final_rect.append(rect_candidate[i])
        final_rect = list(set(final_rect))
    drawing = img.copy()
    for i in range(len(final_rect)):
        box = cv2.boxPoints(final_rect[i])
        box = np.intp(box)
        cv2.drawContours(drawing, [box], 0, [0,0,255],4)
    # cv2.imshow('Contours', drawing)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    # plt.imshow(cv2.cvtColor(drawing, cv2.COLOR_BGR2RGB))
    plt.imshow(drawing)
    plt.show()
    return final_rect


def robot_player(color_image, depth_image):
    rclasses, rscores, rbboxes =  process_image(color_image)
    visualization.plt_bboxes(color_image, rclasses, rscores, rbboxes)
    if(len(rclasses)==0):
        return
    count = 0
    while count < len(rclasses):
        ymin = int(rbboxes[count][0]*color_image.shape[0])
        xmin = int(rbboxes[count][1]*color_image.shape[1])
        ymax = int(rbboxes[count][2]*color_image.shape[0])
        xmax = int(rbboxes[count][3]*color_image.shape[1])
        if (xmax < 450 or xmin > 1000):
            count = count + 1
            continue

        # crop_img = color_image[ymin-10:ymax+10,xmin-10:xmax+10,:]
        # minRect = compute_minRect(crop_img)
        # final_rect = []
        # final_rect = non_maximum_suppression(crop_img,minRect)

        # if len(minRect) == 0:
        #     count = count + 1
        #     print 'No Objects'
        #     continue

        # # find the x, y, z of the pick point in the camera coordinate
        z = 0.9927
        x = (0-624.79)/931.694*z
        y = (0-360.52)/931.463*z
        grasp = PoseStamped()
        grasp.pose.position.x = x
        grasp.pose.position.y = y
        grasp.pose.position.z = z
        # #pick = tf2_geometry_msgs.do_transform_pose(grasp, trans)

        #pick.pose.position.z += 0.076 # offset for tool0 to the suction cup 0.13
        pick_point = [(xmin+xmax)/2,(ymin+ymax)/2]
        # angle =  (0 + minRect[0][2])*3.14/180
        angle = 0

        print 'angle: ',angle
        grasp_point = np.array([[pick_point]], dtype=np.float32)
        gp_base = cv2.perspectiveTransform(grasp_point, image2baseMatrix)

        pick = PoseStamped()
        pick.header.frame_id = "world"
        pick.pose.position.z = 0.134
        pick.pose.position.x = gp_base[0][0][0]/1000.0
        pick.pose.position.y = gp_base[0][0][1]/1000.0
        pick.pose.orientation.x = 1
        pick.pose.orientation.y = 0
        pick.pose.orientation.z = 0
        pick.pose.orientation.w = 0
        set_states()
        pick.pose.position.z = pick.pose.position.z + 0.05
        group.set_pose_target(pick, end_effector_link='tool0') #wrist3_Link
        plan = group.plan()
        raw_input('Press enter to go pick position: ')
        group.execute(plan)
        pick.pose.position.z = pick.pose.position.z - 0.04
        pick.pose.position.x = pick.pose.position.x - 0.005
        pick.pose.position.y = pick.pose.position.y - 0.005
        group.set_pose_target(pick, end_effector_link='tool0')
        plan = group.plan()
        group.execute(plan)
        time.sleep(0.2)
        set_digital_out(0, False)
        time.sleep(1)
        pick.pose.position.z = pick.pose.position.z + 0.1
        group.set_pose_target(pick, end_effector_link='tool0')
        plan = group.plan()
        group.execute(plan)
        # raw_input('Press enter to go place position: ')
        time.sleep(1)
        #place joint
        place_position_0 = [24.47/180*3.14, -104.60/180*3.14, -143.36/180*3.14, -22.19/180*3.14, 89.51/180*3.14, -53.94/180*3.14]
        place_position_1 = [31.28/180*3.14, -105.20/180*3.14, -142.30/180*3.14, -22.70/180*3.14, 89.53/180*3.14, -53.94/180*3.14]
        place_position_2 = [21.45/180*3.14, -108.54/180*3.14, -136.41/180*3.14, -25.19/180*3.14, 89.49/180*3.14, -53.94/180*3.14]
        place_position_3 = [27.38/180*3.14, -108.96/180*3.14, -135.62/180*3.14, -25.59/180*3.14, 89.51/180*3.14, -53.94/180*3.14]

        place_xyz_0 = PoseStamped()
        place_xyz_1 = PoseStamped()
        place_xyz_2 = PoseStamped()
        place_xyz_3 = PoseStamped()

        place_xyz_0.pose.position.z = 0.15
        place_xyz_0.pose.position.x = 0.54
        place_xyz_0.pose.position.y = 0.046
        place_xyz_0.pose.orientation.x = 1
        place_xyz_0.pose.orientation.y = 0
        place_xyz_0.pose.orientation.z = 0
        place_xyz_0.pose.orientation.w = 0

        place_xyz_1.pose.position.z = 0.15
        place_xyz_1.pose.position.x = 0.54
        place_xyz_1.pose.position.y = 0.106
        place_xyz_1.pose.orientation.x = 1
        place_xyz_1.pose.orientation.y = 0
        place_xyz_1.pose.orientation.z = 0
        place_xyz_1.pose.orientation.w = 0

        place_xyz_2.pose.position.z = 0.15
        place_xyz_2.pose.position.x = 0.600
        place_xyz_2.pose.position.y = 0.046
        place_xyz_2.pose.orientation.x = 1
        place_xyz_2.pose.orientation.y = 0
        place_xyz_2.pose.orientation.z = 0
        place_xyz_2.pose.orientation.w = 0

        place_xyz_3.pose.position.z = 0.15
        place_xyz_3.pose.position.x = 0.600
        place_xyz_3.pose.position.y = 0.106
        place_xyz_3.pose.orientation.x = 1
        place_xyz_3.pose.orientation.y = 0
        place_xyz_3.pose.orientation.z = 0
        place_xyz_3.pose.orientation.w = 0




        place_position_up = [26.0/180*3.14, -100.42/180*3.14, -139.49/180*3.14, -30.26/180*3.14, 89.51/180*3.14, -23.94/180*3.14]
        place_position_mid = [99.31/180*3.14, -87.55/180*3.14, -135.44/180*3.14, -47.08/180*3.14, 89.0/180*3.14, -23.94/180*3.14]

       # #
       #  place_0 = PoseStamped()
       #  place_0.pose.position.x = 0.577
       #  place_0.pose.position.y = 0.06
       #  place_0.pose.position.z = 0.048
       #  place_0.pose.orientation.x = 1
       #  place_0.pose.orientation.y = 0.0
       #  place_0.pose.orientation.z = 0.0
       #  place_0.pose.orientation.w = 0.0
       #  place_0_up = place_0
       #  place_0_up.pose.position.z += 0.01
       #  place_1 = place_0
       #  place_1.pose.position.y += 0.051
       #  place_1_up = place_1
       #  place_1_up.pose.position.z += 0.01
       #  place_2 = place_0
       #  place_2.pose.position.x += 0.051
       #  place_2_up = place_2
       #  place_2_up.pose.position.z += 0.01
       #  place_3 = place_0
       #  place_3.pose.position.x += 0.051
       #  place_3.pose.position.y += 0.051
       #  place_3_up = place_3
       #  place_3_up.pose.position.z += 0.01
       #
       #  #go to up of place point
       #  group.set_pose_target(place_0_up, end_effector_link='tool0')
       #  plan = group.plan()
       #  raw_input('Press enter to go place position: ')
       #  group.execute(plan)
       #  time.sleep(1)
       #  #go to place
       #  group.set_pose_target(place_0, end_effector_link='tool0')
       #  plan = group.plan()
       #  raw_input('Press enter to go place position: ')
       #  group.execute(plan)
       #  time.sleep(1)
       #  group.set_joint_value_target(home_joint_position)
       #  plan = group.plan()
       #  raw_input('Press enter to go home position: ')
       #  group.execute(plan)

        group.set_joint_value_target(place_position_mid)
        plan = group.plan()
        group.set_joint_value_target(place_position_up)
        plan = group.plan()
        # raw_input('Press enter to go place position: ')
        group.execute(plan)
        time.sleep(0.5)
        jigsaw_type = rclasses[count]
        if jigsaw_type == 1:
            # place_position_0[5]-=angle
            group.set_joint_value_target(place_position_0)
            # group.set_pose_target(place_xyz_0, end_effector_link='tool0')
        elif jigsaw_type == 3:
            # place_position_1[5]-=angle
            group.set_joint_value_target(place_position_1)
            # group.set_pose_target(place_xyz_1, end_effector_link='tool0')
        elif jigsaw_type == 4:
            # place_position_2[5]-=angle
            group.set_joint_value_target(place_position_2)
            # group.set_pose_target(place_xyz_2, end_effector_link='tool0')
        elif jigsaw_type == 2:
            # place_position_3[5]-=angle
            group.set_joint_value_target(place_position_3)
            # group.set_pose_target(place_xyz_3, end_effector_link='tool0')
        plan = group.plan()
        # raw_input('Press enter to go the place position: ')
        group.execute(plan)
        time.sleep(1)
        set_digital_out(0, True)
        time.sleep(1)
        count = count + 1
        group.set_joint_value_target(place_position_up)
        plan = group.plan()
        group.execute(plan)
        time.sleep(0.3)
    group.set_joint_value_target(home_joint_position)
    plan = group.plan()
    raw_input('Press enter to go home position: ')
    group.execute(plan)



#calibration
# xy1 = [849.35,562.26] #from tp
# xy2 = [839.2,183.67]
# xy3 = [407.45,239.78]
# xy4 = [451.36,667.85]
#
# uv1 = [828,540] #from image process
# uv2 = [480,576]
# uv3 = [478,178]
# uv4 = [874,158]
xy1 = [439.76,729.71] #from tp
xy2 = [458.04,497.03]
xy3 = [236.57, 743.48]
xy4 = [260.350,500.420]

uv1 = [896,434] #from image process
uv2 = [678,456]
uv3 = [904,244]
uv4 = [676,274]

#perspective transformation
pts1 = np.float32([uv1,uv2,uv3,uv4])
pts2 = np.float32([xy1,xy2,xy3,xy4])
image2baseMatrix = cv2.getPerspectiveTransform(pts1,pts2)

# TODO: ROS node initialization
rospy.init_node('perception', anonymous=True)

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

# TODO: transform the picking pose in the robot base coordinate
tfBuffer = tf2_ros.Buffer(rospy.Duration(1200.0))
listener = tf2_ros.TransformListener(tfBuffer)
try:
    trans = tfBuffer.lookup_transform('world', 'camera_color_optical_frame', rospy.Time(0), rospy.Duration(1.0))
except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
    print "Find transform failed"

# TODO: Go to the home pose waiting for picking instruction
home_joint_position = [116.1/180*3.14, -87.58/180*3.14, -135.23/180*3.14, -48.19/180*3.14, 90.03/180*3.14, -23.94/180*3.14]
group.set_joint_value_target(home_joint_position)
plan = group.plan()
raw_input('Press enter to go the home position: ')
group.execute(plan)
time.sleep(1)

# initiate realsense
points = rs.points()
pipeline= rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
profile = pipeline.start(config)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
align_to = rs.stream.color
align = rs.align(align_to)

###=====================================================================================###
#SSD_detection
# TensorFlow session: grow memory when needed. TF, DO NOT USE ALL MY GPU MEMORY!!!
gpu_options = tf.GPUOptions(allow_growth=True)
config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
isess = tf.InteractiveSession(config=config)

# Input placeholder.
net_shape = (300, 300)
data_format = 'NHWC'
img_input = tf.placeholder(tf.uint8, shape=(None, None, 3))
# Evaluation pre-processing: resize to SSD net shape.
image_pre, labels_pre, bboxes_pre, bbox_img = ssd_vgg_preprocessing.preprocess_for_eval(
    img_input, None, None, net_shape, data_format, resize=ssd_vgg_preprocessing.Resize.WARP_RESIZE)
image_4d = tf.expand_dims(image_pre, 0)

# Define the SSD model.
reuse = True if 'ssd_net' in locals() else None
ssd_net = ssd_vgg_300.SSDNet()
with slim.arg_scope(ssd_net.arg_scope(data_format=data_format)):
    predictions, localisations, _, _ = ssd_net.net(image_4d, is_training=False, reuse=reuse)

# Restore SSD model.
ckpt_filename = '/home/bionicdl-saber/Documents/code/SSD-Tensorflow/checkpoints/model.ckpt-2234288'
# ckpt_filename = '../checkpoints/VGG_VOC0712_SSD_300x300_ft_iter_120000.ckpt'
isess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(isess, ckpt_filename)

# SSD default anchor boxes.
ssd_anchors = ssd_net.anchors(net_shape)


###=====================================================================================###
time.sleep(0.1)
frames = pipeline.wait_for_frames()
aligned_frames = align.process(frames)
aligned_depth_frame = aligned_frames.get_depth_frame()
color_frame = aligned_frames.get_color_frame()
depth_image_background = np.asanyarray(aligned_depth_frame.get_data())
color_image_background = np.asanyarray(color_frame.get_data())
# cv2.imwrite("1.jpg", color_image_background)
# time.sleep(1)
# color_image_background = mpimg.imread("1.jpg")
robot_player(color_image_background, depth_image_background)
