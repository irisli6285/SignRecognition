#!/usr/bin/env python
# license removed for brevity
import rospy
from std_msgs.msg import String

def init_publisher():
    pub = rospy.Publisher('distance_chatter', String, queue_size=10)
    rospy.init_node('talker', anonymous=True)
    return pub

def publish_dist(pub, dist_str):
    if not rospy.is_shutdown():
        rospy.loginfo(dist_str)
        pub.publish(dist_str)
        return True
    else:
        return False
