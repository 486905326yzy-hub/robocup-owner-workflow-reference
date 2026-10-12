#!/usr/bin/env python3
import sys
import time

import rospy
from std_msgs.msg import String


if __name__ == "__main__":
    rospy.init_node("offline_voice_say_once")
    topic = rospy.get_param("~topic", rospy.get_param("/tts/subscribe_topic", "/voice/say"))
    text_args = [arg for arg in sys.argv[1:] if not arg.startswith("__") and ":=" not in arg]
    text = " ".join(text_args).strip() or "Hello from offline voice bridge."
    pub = rospy.Publisher(topic, String, queue_size=1)
    deadline = time.time() + 5.0
    while pub.get_num_connections() == 0 and time.time() < deadline and not rospy.is_shutdown():
        rospy.sleep(0.1)
    pub.publish(String(data=text))
    rospy.loginfo("Published one TTS message to %s: %s", topic, text)
    rospy.sleep(1.0)
