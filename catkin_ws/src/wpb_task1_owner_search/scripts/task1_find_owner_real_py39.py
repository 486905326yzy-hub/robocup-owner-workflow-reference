#!/usr/bin/env python3
"""Run the owner-search task with the Python 3.9 CUDA environment.

ROS Noetic's generated node wrappers use Python 3.8, while this machine's
RTX 50-series PyTorch/CUDA build is installed for Python 3.9.  This tiny
relay keeps the ROS launch interface unchanged and replaces itself with the
actual task under Python 3.9 before any ROS modules are imported.
"""

import os
import sys


ROS_PYTHON_PATHS = (
    "/home/ubuntu20/catkin_ws/devel/lib/python3/dist-packages",
    "/opt/ros/noetic/lib/python3/dist-packages",
)
source_dir = os.path.dirname(os.path.abspath(__file__))
task_script = os.path.join(source_dir, "task1_find_owner_real.py")

current_parts = [part for part in os.environ.get("PYTHONPATH", "").split(":") if part]
for path in reversed(ROS_PYTHON_PATHS):
    if path not in current_parts:
        current_parts.insert(0, path)
os.environ["PYTHONPATH"] = ":".join(current_parts)

os.execv("/usr/bin/python3.9", ["python3.9", task_script] + sys.argv[1:])
