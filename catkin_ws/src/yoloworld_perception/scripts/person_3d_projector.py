#!/usr/bin/env python3
import math
import threading

import numpy as np
import rospy
from geometry_msgs.msg import PointStamped
from perception_msgs.msg import Person3D, Person3DArray, Detection2DArray
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2

try:
    import tf2_geometry_msgs  # noqa: F401
    import tf2_ros
except Exception:
    tf2_ros = None


class Person3DProjector:
    def __init__(self):
        self.lock = threading.Lock()
        self.latest_detections = None

        self.detections_topic = rospy.get_param("~detections_topic", "/yoloworld/detections")
        self.pointcloud_topic = rospy.get_param("~pointcloud_topic", "/kinect2/qhd/points")
        self.target_frame = rospy.get_param("~target_frame", "base_footprint")
        self.detection_image_width = float(rospy.get_param("~detection_image_width", 0.0))
        self.detection_image_height = float(rospy.get_param("~detection_image_height", 0.0))
        self.max_detection_age = float(rospy.get_param("~max_detection_age", 0.5))
        self.sample_grid = max(1, int(rospy.get_param("~sample_grid", 7)))
        self.min_valid_points = max(1, int(rospy.get_param("~min_valid_points", 3)))
        self.min_depth = float(rospy.get_param("~min_depth", 0.2))
        self.max_depth = float(rospy.get_param("~max_depth", 8.0))

        self.tf_buffer = None
        self.tf_listener = None
        if self.target_frame and tf2_ros is not None:
            self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(10.0))
            self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        elif self.target_frame:
            rospy.logwarn("tf2 is unavailable; 3D persons will stay in point-cloud frame")

        self.pub = rospy.Publisher("~persons", Person3DArray, queue_size=5)
        self.det_sub = rospy.Subscriber(self.detections_topic, Detection2DArray, self.detections_callback, queue_size=5)
        self.pc_sub = rospy.Subscriber(self.pointcloud_topic, PointCloud2, self.pointcloud_callback, queue_size=1, buff_size=2**26)

        rospy.loginfo("Subscribed detections topic: %s", self.detections_topic)
        rospy.loginfo("Subscribed pointcloud topic: %s", self.pointcloud_topic)
        if self.detection_image_width > 0 and self.detection_image_height > 0:
            rospy.loginfo(
                "Scaling detection pixels from %.0fx%.0f to incoming point-cloud dimensions",
                self.detection_image_width,
                self.detection_image_height,
            )

    def detections_callback(self, msg):
        with self.lock:
            self.latest_detections = msg

    def pointcloud_callback(self, cloud_msg):
        with self.lock:
            det_msg = self.latest_detections

        if det_msg is None or not det_msg.detections:
            return

        age = abs((cloud_msg.header.stamp - det_msg.header.stamp).to_sec())
        if age > self.max_detection_age:
            rospy.logwarn_throttle(3.0, "Skipping stale detections: age %.3fs", age)
            return

        persons = Person3DArray()
        persons.header = cloud_msg.header
        if self.target_frame and self.tf_buffer is not None:
            persons.header.frame_id = self.target_frame

        for det in det_msg.detections:
            point = self.estimate_point(cloud_msg, det)
            if point is None:
                continue
            if self.target_frame and self.tf_buffer is not None:
                point = self.transform_point(point, self.target_frame)
                if point is None:
                    continue

            person = Person3D()
            person.header = persons.header
            person.label = det.class_name
            person.class_id = det.class_id
            person.score = det.score
            person.x = float(point.point.x)
            person.y = float(point.point.y)
            person.z = float(point.point.z)
            person.xmin = det.xmin
            person.ymin = det.ymin
            person.xmax = det.xmax
            person.ymax = det.ymax
            person.center_x = det.center_x
            person.center_y = det.center_y
            persons.persons.append(person)

        self.pub.publish(persons)

    def estimate_point(self, cloud_msg, det):
        if cloud_msg.width == 0 or cloud_msg.height == 0:
            return None

        scale_x = 1.0
        scale_y = 1.0
        if self.detection_image_width > 0 and self.detection_image_height > 0:
            scale_x = float(cloud_msg.width) / self.detection_image_width
            scale_y = float(cloud_msg.height) / self.detection_image_height

        xmin = max(0, min(int(round(det.xmin * scale_x)), cloud_msg.width - 1))
        xmax = max(0, min(int(round(det.xmax * scale_x)), cloud_msg.width - 1))
        ymin = max(0, min(int(round(det.ymin * scale_y)), cloud_msg.height - 1))
        ymax = max(0, min(int(round(det.ymax * scale_y)), cloud_msg.height - 1))
        if xmax < xmin:
            xmin, xmax = xmax, xmin
        if ymax < ymin:
            ymin, ymax = ymax, ymin

        # Bias sampling toward the upper/middle body where Kinect depth is usually cleaner than legs.
        y_mid = int(ymin + 0.45 * max(1, ymax - ymin))
        half_h = max(2, int(0.20 * max(1, ymax - ymin)))
        y0 = max(ymin, y_mid - half_h)
        y1 = min(ymax, y_mid + half_h)

        uvs = []
        for x in self.linspace_int(xmin, xmax, self.sample_grid):
            for y in self.linspace_int(y0, y1, self.sample_grid):
                uvs.append((x, y))

        points = []
        for x, y, z in point_cloud2.read_points(cloud_msg, field_names=("x", "y", "z"), skip_nans=True, uvs=uvs):
            if not all(math.isfinite(v) for v in (x, y, z)):
                continue
            depth = math.sqrt(x * x + y * y + z * z)
            if self.min_depth <= depth <= self.max_depth:
                points.append((x, y, z))

        if len(points) < self.min_valid_points:
            return None

        median = np.median(np.asarray(points, dtype=np.float32), axis=0)
        out = PointStamped()
        out.header = cloud_msg.header
        out.point.x = float(median[0])
        out.point.y = float(median[1])
        out.point.z = float(median[2])
        return out

    def transform_point(self, point, target_frame):
        try:
            return self.tf_buffer.transform(point, target_frame, rospy.Duration(0.05))
        except Exception as exc:
            rospy.logwarn_throttle(2.0, "Point transform to %s failed: %s", target_frame, exc)
            return None

    @staticmethod
    def linspace_int(start, stop, count):
        if count <= 1 or start == stop:
            return [int(round((start + stop) / 2.0))]
        return [int(round(start + (stop - start) * i / float(count - 1))) for i in range(count)]


def main():
    rospy.init_node("person_3d_projector")
    Person3DProjector()
    rospy.spin()


if __name__ == "__main__":
    main()
