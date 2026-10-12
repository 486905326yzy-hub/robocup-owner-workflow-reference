#!/usr/bin/env python3
"""Run the existing offline ASR node explicitly with its Python 3.8 runtime."""

import os

target = "/home/ubuntu20/catkin_ws/src/offline_voice_bridge/scripts/offline_asr_node.py"
runtime = "/usr/bin/python3"
environment = os.environ.copy()
site = "/usr/local/lib/python3.8/dist-packages"
current = [value for value in environment.get("PYTHONPATH", "").split(":") if value]
if site not in current:
    current.append(site)
environment["PYTHONPATH"] = ":".join(current)
os.execve(runtime, [runtime, target], environment)
