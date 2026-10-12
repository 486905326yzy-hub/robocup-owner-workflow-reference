#!/usr/bin/env python3
"""Start the owner enrollment node in the existing Python 3.9 environment."""
import os
import sys

ROS_PYTHON_PATHS = (
    "/home/ubuntu20/catkin_ws/devel/lib/python3/dist-packages",
    "/opt/ros/noetic/lib/python3/dist-packages",
)
source_dir = os.path.dirname(os.path.abspath(__file__))
task_script = os.path.join(source_dir, "owner_enrollment_real.py")
parts = [part for part in os.environ.get("PYTHONPATH", "").split(":") if part]
for path in reversed(ROS_PYTHON_PATHS):
    if path not in parts:
        parts.insert(0, path)
os.environ["PYTHONPATH"] = ":".join(parts)
os.execv("/usr/bin/python3.9", ["python3.9", task_script] + sys.argv[1:])
