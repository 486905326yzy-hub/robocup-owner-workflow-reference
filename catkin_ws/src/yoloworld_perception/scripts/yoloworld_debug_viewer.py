#!/usr/bin/env python3
import os

import cv2
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image


class YoloWorldDebugViewer:
    def __init__(self):
        self.bridge = CvBridge()
        self.image_topic = rospy.get_param("~image_topic", "/kinect2/hd/image_color_rect")
        self.debug_image_topic = rospy.get_param("~debug_image_topic", "/perception/yoloworld/debug_image")
        self.show_raw = bool(rospy.get_param("~show_raw", True))
        self.show_debug = bool(rospy.get_param("~show_debug", True))
        self.raw_window = rospy.get_param("~raw_window", "WPR Camera")
        self.debug_window = rospy.get_param("~debug_window", "YOLO-World Person Detection")

        self.gui_enabled = bool(os.environ.get("DISPLAY"))
        if not self.gui_enabled:
            rospy.logwarn("DISPLAY is not set; OpenCV windows will not be shown")
        else:
            if self.show_raw:
                cv2.namedWindow(self.raw_window, cv2.WINDOW_NORMAL)
            if self.show_debug:
                cv2.namedWindow(self.debug_window, cv2.WINDOW_NORMAL)

        if self.show_raw:
            self.raw_sub = rospy.Subscriber(self.image_topic, Image, self.raw_callback, queue_size=1, buff_size=2**24)
            rospy.loginfo("Debug viewer raw image: %s", self.image_topic)
        if self.show_debug:
            self.debug_sub = rospy.Subscriber(self.debug_image_topic, Image, self.debug_callback, queue_size=1, buff_size=2**24)
            rospy.loginfo("Debug viewer YOLO image: %s", self.debug_image_topic)

    def raw_callback(self, msg):
        self.show_image(msg, self.raw_window)

    def debug_callback(self, msg):
        self.show_image(msg, self.debug_window)

    def show_image(self, msg, window_name):
        if not self.gui_enabled:
            return
        try:
            image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except CvBridgeError as exc:
            rospy.logerr_throttle(2.0, "cv_bridge conversion failed: %s", exc)
            return

        cv2.imshow(window_name, image)
        cv2.waitKey(1)

    def close(self):
        if self.gui_enabled:
            cv2.destroyAllWindows()


def main():
    rospy.init_node("yoloworld_debug_viewer")
    viewer = YoloWorldDebugViewer()
    rospy.on_shutdown(viewer.close)
    rospy.spin()


if __name__ == "__main__":
    main()

