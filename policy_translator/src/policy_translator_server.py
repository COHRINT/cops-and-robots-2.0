#!/usr/bin/env python

'''
policy_translator_server.py

A ROS service which interfaces with a PolicyTranslator. Receives a request for
a goal pose, which includes a belief, and responds with a new goal pose and an
updated belief.
'''

__author__ = "Ian Loefgren"
__copyright__ = "Copyright 2017, Cohrint"
__credits__ = ["Ian Loefgren"]
__license__ = "GPL"
__version__ = "1.0"
__maintainer__ = "Ian Loefgren"
__email__ = "ian.loefgren@colorado.edu"
__status__ = "Development"

from policy_translator.srv import *
from policy_translator.msg import *
from observation_interface.msg import *
from observation_interface.srv import *
from std_msgs.msg import String # human_push topic

import rospy
import tf
import numpy as np
import math
import os

import voi # obs_mapping in callbacks
from gaussianMixtures import GM
from PolicyTranslator import PolicyTranslator
from MAPTranslator import MAPTranslator
from belief_handling import rehydrate_msg, dehydrate_msg, discrete_rehydrate, discrete_dehydrate

# Observation Queue #TODO delete in CnR 2.0
from obs_queue import Obs_Queue

class PolicyTranslatorServer(object):

    def __init__(self, check="MAP"):
        if check == 'MAP':
            print("Running MAP Translator!")
            self.pt = MAPTranslator()
            self.trans = "MAP"      # Variable used in wrapper to bypass observation interface
        else:
            args = ['PolicyTranslator.py','-n','D2Diffs','-r','True','-a','1','-s','False','-g','True'];
            self.pt = PolicyTranslator(args)
            self.trans = "POL"

        rospy.init_node('policy_translator_server')
        self.listener = tf.TransformListener()
        s = rospy.Service('translator',discrete_policy_translator_service,self.handle_policy_translator)

        # Observations -> likelihood queue
        rospy.Subscriber("/human_push", String, self.human_push_callback)
        rospy.Subscriber("/answered", Answer, self.robot_pull_callback)
        self.queue = Obs_Queue()

        self.likelihoods = np.load(os.path.dirname(__file__) + "/likelihoods.npy")

        bounds = [-9.6, -3.6, 4, 3.6]
        self.delta = 0.1
        self.shapes = [int((bounds[2]-bounds[0])/self.delta),int((bounds[3]-bounds[1])/self.delta)]

        print('Policy translator service ready.')

        rospy.spin()

    def handle_policy_translator(self,req):
        '''
        Create an observation request, get a new goal and belief and package
        them to respond.
        '''
        name = req.request.name

        if self.trans == "MAP":
            belief = self.translator_wrapper(req.request.name,self.queue.flush(),
                                                flat_belief=req.request.belief)
        else:      # run the observation observance
            if not req.request.weights:
                obs = None
            else:
                obs_msg = ObservationRequest()
                obs = self.obs_server_client(obs_msg)


                belief = self.translator_wrapper(req.request.name,req.request.weights,
                                    req.request.means,req.request.variances,obs)

        if self.trans != "MAP":
            weights_updated = belief[0]
            means_updated = belief[1]
            variances_updated = belief[2]
            goal_pose = belief[3]

            res = self.create_message(req.request.name,
                                goal_pose,
                                weights_updated=weights_updated,
                                means_updated=means_updated,
                                variances_updated=variances_updated)

        else:
            goal_pose = belief[1]
            res = self.create_message(req.request.name,
                                        goal_pose,
                                        flat_belief=belief[0])

        return res

    def obs_server_client(self,msg):
        '''
        Request an observation from the observation server.
        '''
        rospy.wait_for_service('observation_interface')
        try:
            proxy = rospy.ServiceProxy('observation_interface',observation_server)
            res = proxy(msg)
            return res.response.observation
        except rospy.ServiceException, e:
            print "Service call failed: %s"%e

    def tf_update(self,name):
        '''
        Get the pose of the robot making the service request using a ROS
        transform ('tf') lookup and return that pose.
        '''
        name = name.lower()
        ref = "/" + name + "/odom"
        child = "/" + name + "/base_footprint"
        (trans, rot) = self.listener.lookupTransform(ref, child, rospy.Time(0))
        x = trans[0]
        y = trans[1]
        (_, _, theta) = tf.transformations.euler_from_quaternion(rot)
        pose = [x, y, np.rad2deg(theta)]
        return pose

    def translator_wrapper(self,name,obs,weights=None,means=None,variances=None,flat_belief=None):
        '''
        Rehydrate the belief then get the position of the calling robot, update the
        belief and get a new goal pose. Then dehydrate the updated belief.
        '''
        copPoses = []

        if self.trans == "MAP":
            belief = discrete_rehydrate(flat_belief,self.shapes)
        else:
            belief = rehydrate_msg(weights,means,variances)

        position = self.tf_update(name)

        copPoses.append(position)

        # obs = self.queue.flush()
        print("OBSERVATIONS: {}".format(obs))

        (b_updated,goal_pose) = self.pt.getNextPose(belief,obs,copPoses)

        if b_updated is not None:
            if self.trans == "MAP":
                belief = discrete_dehydrate(b_updated)
            else:
                (weights,means,variances) = dehydrate_msg(b_updated)

        orientation = math.atan2(goal_pose[1]-position[1],goal_pose[0]-position[0])
        goal_pose.append(orientation)

        if self.trans == "MAP":
            belief = [belief,goal_pose]
        else:
            belief = [weights,means,variances,goal_pose]
        return belief

    def create_message(self,name,goal_pose,weights=None,means=None,variances=None,flat_belief=None):
        '''
        Create a response message containing the new dehydrated belief and the
        new goal pose.
        '''
        msg = None
        if self.trans == "MAP":
            msg = DiscretePolicyTranslatorResponse()
            msg.name = name
            msg.belief_updated = flat_belief
            msg.goal_pose = goal_pose
        else:
            msg = PolicyTranslatorResponse()
            msg.name = name
            msg.weights_updated = weights
            msg.means_updated = means
            msg.variances_updated = variances
            msg.goal_pose = goal_pose
        return msg


    def human_push_callback(self, human_push):
        """
        Mapping of human push observations to a likelihood index and pos_neg value
        """
        # strip the space from message
        #print("Origingal String:" + human_push.data)
        question = human_push.data.lstrip()
        #print("Now String:"+question)
        (lkhd_question, ans) = voi.obs_mapping[question]
        print("HUMAN PUSH OBS ADDED")

        try:
            # lhs = np.load('likelihoods.npy')
            item = np.where(self.likelihoods['question']==lkhd_question)
            index = item[0][0]
            self.queue.add(index, ans)
            print("Question added: {}".format(self.likelihoods[index][0]))
        except IOError as ioerr:
            print(ioerr)


    def robot_pull_callback(self, data):
        """"
        Mapping of human response observations to likelihood index and pos_neg value
        """
        self.queue.add(data.qid, data.ans)
        print("ROBOT PULL OBS ADDED")

# Comment out rospy.spin() in init function of policy_translator_server
def Test_Callbacks():
    # Test human_push_callback
    a = String()
    a.data = "    I know Roy is right of the dining table" # q_id 5
    b = String()
    b.data = " I know Roy is not near the bookcase"  # q_id 27

    server = PolicyTranslatorServer()

    # send test data
    server.human_push_callback(a)
    server.human_push_callback(b)

    print("Printing Current Queue")
    server.Obs_Queue.print_queue()

    # Test robot_pull_callback
    c = Answer()
    c.qid = 17
    c.ans = False

    d = Answer()
    d.qid = 65
    d.ans = True

    # send test data
    server.robot_pull_callback(c)
    server.robot_pull_callback(d)

    print("Printing Current Queue")
    server.Obs_Queue.print_queue()

if __name__ == "__main__":
    # Test_Callbacks()
    PolicyTranslatorServer()
