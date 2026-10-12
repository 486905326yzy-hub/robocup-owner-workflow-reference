#!/usr/bin/env python3
# coding=utf-8
"""Start sound_play only after ROS and the audio device have settled."""

import os
import time

import rospy


def main():
    rospy.init_node("soundplay_launcher", anonymous=True)
    delay = max(0.0, float(rospy.get_param("~startup_delay", 3.0)))
    device = str(rospy.get_param("~device", "plughw:1,0"))
    rospy.loginfo("Starting sound_play in %.1fs on %s", delay, device)
    time.sleep(delay)
    os.execvp(
        "rosrun",
        [
            "rosrun",
            "sound_play",
            "soundplay_node.py",
            "_device:=" + device,
            "__name:=soundplay_node",
        ],
    )


if __name__ == "__main__":
    main()
