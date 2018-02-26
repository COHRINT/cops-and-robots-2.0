#!/usr/bin/env python

'''
policy_translator_server.py

A ROS service which interfaces with a PolicyTranslator. Receives a request for
a goal pose, which includes a belief, and responds with a new goal pose and an
updated belief.
'''

__author__ = "Ian Loefgren, Luke Barbier"
__copyright__ = "Copyright 2017, Cohrint"
__credits__ = ["Ian Loefgren"]
__license__ = "GPL"
__version__ = "2.1"
__maintainer__ = "Ian Loefgren"
__email__ = "ian.loefgren@colorado.edu"
__status__ = "Development"

from pdb import set_trace

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
import time

# import voi # obs_mapping in callbacks
from pose import Pose
from gaussianMixtures import GM
from POMDPTranslator import POMDPTranslator
from belief_handling import rehydrate_msg, dehydrate_msg, discrete_rehydrate, discrete_dehydrate
from belief_handling import rehydrate_msg, dehydrate_msg

# Observation Queue 
from obs_queue import Obs_Queue

# For publishing the belief image over ROS
import cv2
from sensor_msgs.msg import CompressedImage, Image
from cv_bridge import CvBridge, CvBridgeError


class PolicyTranslatorServer(object):

    def __init__(self, check="MAP"):

        #Late hack to gather data
        self.allObs = [];
        self.fileName = 'obs_{}.npy'.format(time.clock()); 

        #Yes this is probably the worst way to adhoc a timer: Luke
        self.questionCounter = 0; 
        #The interface will only recieve questions every "questionDivide" updates
        #Effectively,the interface will update at a rate 'questionDivide' times slower for robot pull
        #But everything else will work normally
        self.questionDivide = 10; 

        self.pt = POMDPTranslator()

        rospy.init_node('policy_translator_server')
        self.cop_pose = None
#        self.listener = tf.TransformListener()
#        cop_name = rospy.get_param('cop')
#        self.cop_pose = Pose(cop_name)
        s = rospy.Service('translator',policy_translator_service,self.handle_policy_translator)

        # Observations -> likelihood queue
        rospy.Subscriber("/human_push", String, self.human_push_callback)
        rospy.Subscriber("/answered", Answer, self.robot_pull_callback)
        self.q_pub = rospy.Publisher("/robot_questions",Question,queue_size=10)
        self.queue = Obs_Queue()

        # self.likelihoods = np.load(os.path.dirname(__file__) + "/likelihoods.npy")

        bounds = [-5, -2.5, 5, 2.5]
        self.delta = 0.1
        self.shapes = [int((bounds[2]-bounds[0])/self.delta),int((bounds[3]-bounds[1])/self.delta)]

        self.call_count = 0

        print('Policy translator service ready.')

        # Initialize cv image pub objects
        self.bridge = CvBridge()
        self.image_pub = rospy.Publisher("/interface_map", Image, queue_size=10)

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

            print(req.request.weights)
            belief = self.translator_wrapper(req.request.name,obs,req.request.weights,
                                req.request.means,req.request.variances)


        weights_updated = belief[0]
        means_updated = belief[1]
        variances_updated = belief[2]
        goal_pose = belief[3]

        res = self.create_message(req.request.name,
                            goal_pose,
                            weights=weights_updated,
                            means=means_updated,
                            variances=variances_updated)
        
        #Hack observations into npy file
        qs = []
        if obs is not None:
            for observation in obs:
                if observation is str:
                    qs.append(observation[-1]+'\n')
                else:
                    observation = ''.join(str(e) for e in observation[-1])
                    qs.append(observation+'\n')
            qs = ''.join(qs)
        else:
            qs = 'no observations'
        allObs.append(qs); 
        np.save(open(self.fileName),self.allObs);

        # write observations for update to text file
        qs = []
        if obs is not None:
            for observation in obs:
                if observation is str:
                    qs.append(observation[-1]+'\n')
                else:
                    observation = ''.join(str(e) for e in observation[-1])
                    qs.append(observation+'\n')
            qs = ''.join(qs)
        else:
            qs = 'no observations'
        with open(os.path.dirname(__file__) + "/../tmp/obs_{}.txt".format(time.time()),'a+') as f:
            f.write(qs)

        return res

    # def tf_update(self,name):
    #     '''
    #     Get the pose of the robot making the service request using a ROS
    #     transform ('tf') lookup and return that pose.
    #     '''
    #     # return (0,0,0)
    #     name = name.lower()
    #     ref = "/" + name + "/odom"
    #     child = "/" + name + "/base_footprint"
    #     (trans, rot) = self.listener.lookupTransform(ref, child, rospy.Time(0))
    #     x = trans[0]
    #     y = trans[1]
    #     (_, _, theta) = tf.transformations.euler_from_quaternion(rot)
    #     pose = [x, y, np.rad2deg()]
    #     return pose

    def translator_wrapper(self,name,obs,weights=None,means=None,variances=None):
        '''
        Rehydrate the belief then get the position of the calling robot, update the
        belief and get a new goal pose. Then dehydrate the updated belief.
        '''
        goal_pose = None
        copPoses = []

        belief = rehydrate_msg(weights,means,variances)
        if self.cop_pose is None:
            self.cop_pose = Pose(name)
        position = self.cop_pose.pose
        position[2] = (position[2] * 180) / np.pi
        
#        position = self.tf_update(name)

        copPoses.append(position)

        # if (self.call_count % 4 == 0):
        (b_updated,goal_pose,questions) = self.pt.getNextPose(belief,obs,copPoses)
        
        q_msg = Question()
        # q_msg.qids = questions[1]
        q_msg.qids = [0 for x in range(0,len(questions[0]))]
        q_msg.questions = questions[0]
        q_msg.weights = [0 for x in range(0,len(questions[0]))]
        if(self.questionCounter%self.questionDivide == 0):
            self.q_pub.publish(q_msg)
        self.questionCounter += 1; 

        # Publish the saved image to a rosptopic
        print("Reading image")
        try:
            cv_image = cv2.imread(os.path.dirname(__file__) + "/../tmp/tmpBelief.png")
            imageMsg = self.bridge.cv2_to_imgmsg(cv_image, "bgr8")
            self.image_pub.publish(imageMsg)
        except CvBridgeError as e:
            print(e)

        # else:
        #     b_updated = self.pt.beliefUpdate(belief,obs,copPoses)
        #     goal_pose = [position[0],position[1]]

        self.call_count += 1

        orientation = math.atan2(goal_pose[1]-position[1],goal_pose[0]-position[0])
        goal_pose[2] = orientation

        if b_updated is not None:
            (weights,means,variances) = dehydrate_msg(b_updated)

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
        room_num, model, class_idx, sign = self.pt.obs2models(question,self.cop_pose.pose)
        self.queue.add(question, room_num, model, class_idx, sign)
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
        question = [data.question,data.ans]
        room_num, model, class_idx, sign = self.pt.obs2models(question,self.cop_pose.pose)
        self.queue.add(question, room_num, model, class_idx, sign)
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
