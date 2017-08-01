#!/usr/bin/env python

from policy_translator.srv import *
from policy_translator.msg import *
from std_msgs.msg import String, Float64
from belief_handling import rehydrate_msg, dehydrate_msg
from gaussianMixtures import GM
import rospy

def policy_translator_client(msg):
    rospy.wait_for_service('translator')
    try:
        pt = rospy.ServiceProxy('translator',policy_translator_service)
        print(msg)
        print('')
        res = pt(msg)
        return res
    except rospy.ServiceException, e:
        print "Service call failed: %s"%e

# def dehydrate_msg(belief,n):
#     '''
#     INSERT CODE TO GET WEIGHTS MEANS VARIANCES FROM BELIEF HERE
#     '''
#     means = belief
#     # Means - list of 1xn vectors
#     means_length = len(means)
#     total_elements = n*means_length
#     means_dry = []

#     for i in range(0,means_length):
#         for j in range(0,n):
#             means_dry.append(means[i][j])

#     return means_dry

# def rehydrate_msg(res):
#     weights = res.response.weights_updated
#     means = res.response.means_updated
#     variances = res.response.variances_updated
#     size = res.response.size



if __name__ == '__main__':

    rospy.init_node('test_client')

    belief = GM()
    belief.addNewG([-6,3,-3,2.5],[[2,0,0,0],[0,2,0,0],[0,0,2,0],[0,0,0,2]],1) # kitchen
    belief.addNewG([-7,2,-5,0],[[2,0,0,0],[0,2,0,0],[0,0,2,0],[0,0,0,2]],1) # hallway
    belief.addNewG([0,0,0,-2.5],[[2,0,0,0],[0,2,0,0],[0,0,2,0],[0,0,0,2]],1) # library
    belief.addNewG([0,0,2,2.5],[[2,0,0,0],[0,2,0,0],[0,0,2,0],[0,0,0,2]],1) # billiards room
    belief.addNewG([0,0,-5,-2],[[2,0,0,0],[0,2,0,0],[0,0,2,0],[0,0,0,2]],1) # study
    belief.addNewG([0,0,-8,-2],[[2,0,0,0],[0,2,0,0],[0,0,2,0],[0,0,0,2]],1) # dining room
    belief.normalizeWeights()

    msg = PolicyTranslatorRequest()
    # name = String('deckard')
    msg.name = 'roy'
    # msg.weights = [1.1,2.2,3.3,4.4,5.5]
    # msg.means = [1.1,2.2,3.3,4.4,5.5]
    # msg.variances = [1.1,2.2,3.3,4.4,5.5]
    msg.weights, msg.means, msg.variances = dehydrate_msg(belief)

    print('Requesting service')
    res = policy_translator_client(msg)
    print(res)

    for i in range(0,100):
        belief = rehydrate_msg(res.response.weights_updated,res.response.means_updated,res.response.variances_updated)
        msg = PolicyTranslatorRequest()
        msg.weights, msg.means, msg.variances = dehydrate_msg(belief)
        msg.name = 'deckard'
        # msg.weights = res.response.weights_updated
        # msg.means = res.response.means_updated
        # msg.variances = res.response.variances_updated

        print('Requesting service')
        res = policy_translator_client(msg)
        print(res)
        print('--------------------------')
        rospy.sleep(2)

    # mean = [1,2,3,4]
    # means = [mean,mean,mean,mean]
    #
    # variance1 = []
    #
    # means_dry = dehydrate_msg(means,len(mean))
    #
    # print(means_dry)
