import pyrealsense2 as rs
import numpy as np
import cv2

points = rs.points()
pipeline= rs.pipeline()

config = rs.config()
config.enable_stream(rs.stream.infrared, 1, 1280, 720, rs.format.y8, 30)
config.enable_stream(rs.stream.infrared, 2, 1280, 720, rs.format.y8, 30)
# config.enable_stream(rs.stream.depth,    3,1280, 720, rs.format.z16, 30)

profile = pipeline.start(config)


try:
    while True:
        frames = pipeline.wait_for_frames()
        irL_frame = frames.get_infrared_frame(1)
        irR_frame = frames.get_infrared_frame(2)
        # depth_frame = frames.get_infrared_frame(3)



        image_L = np.asanyarray(irL_frame.get_data())
        image_R = np.asanyarray(irR_frame.get_data())
        # image_D = np.asanyarray(depth_frame.get_data())

        cv2.namedWindow('IR Example', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('IR Example', image_L)

        cv2.namedWindow('IR Example2', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('IR Example2', image_R)

        key = cv2.waitKey(1)
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break
finally:
    pipeline.stop()
