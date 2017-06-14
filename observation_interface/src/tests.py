#!/usr/bin/env python

"""
CnR 2.0 Interface element unit tests
"""

import rospy
import sys

from observation_interface.msg import *


def robot_pull_test():
    rospy.init_node('interface_tester')
    pub = rospy.Publisher("pull_questions",Question,queue_size=10)

    # qids = [1,2,5,4,3]
    # weights = [5,10,2,7,4]
    #
    # msg = Question()
    # msg.qids = qids
    # msg.weights = weights
    #
    # pub.publish(msg)

    rospy.sleep(5)

    qids = [1,2,5,4,3]
    weights = [5,10,2,7,4]

    msg = Question()
    msg.qids = qids
    msg.weights = weights

    pub.publish(msg)


if __name__ == "__main__":
    robot_pull_test()
