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
__version__ = "2.0"
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
from belief_handling import rehydrate_msg, dehydrate_msg

# Observation Queue #TODO delete in CnR 2.0
from obs_queue import Obs_Queue

class PolicyTranslatorServer(object):

    def __init__(self, check="MAP"):
        if check == 'MAP': # Allow for use of different translators
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

        # self.likelihoods = np.load(os.path.dirname(__file__) + "/likelihoods.npy")

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

        if not req.request.weights:
            obs = None
        else:
            obs = self.queue.flush()

            belief = self.translator_wrapper(req.request.name,obs,req.request.weights,
                                req.request.means,req.request.variances)


        weights_updated = belief[0]
        means_updated = belief[1]
        variances_updated = belief[2]
        goal_pose = belief[3]

        res = self.create_message(req.request.name,
                            goal_pose,
                            weights_updated=weights_updated,
                            means_updated=means_updated,
                            variances_updated=variances_updated)

        return res

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

    def translator_wrapper(self,name,obs,weights=None,means=None,variances=None):
        '''
        Rehydrate the belief then get the position of the calling robot, update the
        belief and get a new goal pose. Then dehydrate the updated belief.
        '''
        copPoses = []

        belief = rehydrate_msg(weights,means,variances)

        position = self.tf_update(name)

        copPoses.append(position)

        (b_updated,goal_pose) = self.pt.getNextPose(belief,obs,copPoses)

        if b_updated is not None:
            (weights,means,variances) = dehydrate_msg(b_updated)

        orientation = math.atan2(goal_pose[1]-position[1],goal_pose[0]-position[0])
        goal_pose.append(orientation)
        belief = [weights,means,variances,goal_pose]
        return belief

    def create_message(self,name,goal_pose,weights=None,means=None,variances=None):
        '''
        Create a response message containing the new dehydrated belief and the
        new goal pose.
        '''
        msg = None
        msg = PolicyTranslatorResponse()
        msg.name = name
        msg.weights_updated = weights
        msg.means_updated = means
        msg.variances_updated = variances
        msg.goal_pose = goal_pose
        return msg

    def human_push_callback(self, human_push):
        """
        -Maps "human push" observations to a sofmax model and class, with a
            positive/negative value
        -Adds the mapped observation's model, class, and sign to the observation
            queue (self.queue)

        ----------
        Parameters
        ----------
        data : std_msgs.msg.String
        """
        # strip the space from message
        question = human_push.data.lstrip()
        model, class_idx, sign = self.pt.obs2models(question)
        self.queue.add(model, class_idx, sign)
        print("HUMAN PUSH OBS ADDED")

    def robot_pull_callback(self, data):
        """
        -Maps "robot pull" question responses to a sofmax model and class, with a
            positive/negative value
        -Adds the mapped observation's model, class, and sign to the observation
            queue (self.queue)

        Parameters
        ----------
        data : Answer.msg , Custom Message
        """
        question = [data.text,data.ans]
        model, class_idx, sign = self.pt.obs2models(question)
        self.queue.add(model, class_idx, sign)
        print("ROBOT PULL OBS ADDED")

def Test_Callbacks():
    """ Test human_push_callback and robot_pull_callback
        Comment out rospy.spin() in the init function of policy_translator_server
    """
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
