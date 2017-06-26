#!/usr/bin/env python

from __future__ import division


"""
Compute VOI of all possible questions in experiment based on the entropy of the
current belief.
"""

import numpy as np
import rospy
import os

from gaussianMixtures import GM

from policy_translator.msg import *

obs_mapping = {'I know Roy is right of the dining table': ['Is Roy right of the dining table?', True],
'I know Roy is near the dining room': ['Is Roy near the dining room?', True],
'I know Roy is behind the checkers table': ['Is Roy behind the checkers table?', True],
'I know Roy is in front of the cassini poster': ['Is Roy in front of the cassini poster?', True],
'I know Roy is not left of the mars poster': ['Is Roy left of the mars poster?', False],
'I know Roy is right of the desk': ['Is Roy right of the desk?', True],
'I know Roy is not near the fridge': ['Is Roy near the fridge?', False],
'I know Roy is near the chair': ['Is Roy near the chair?', True],
'I know Roy is not near the fern': ['Is Roy near the fern?', False],
'I know Roy is not near the kitchen': ['Is Roy near the kitchen?', False],
'I know Roy is not in front of the desk': ['Is Roy in front of the desk?', False],
'I know Roy is not right of the mars poster': ['Is Roy right of the mars poster?', False],
'I know Roy is near the checkers table': ['Is Roy near the checkers table?', True],
'I know Roy is right of the chair': ['Is Roy right of the chair?', True],
'I know Roy is behind the bookcase': ['Is Roy behind the bookcase?', True],
'I know Roy is not in front of the cassini poster': ['Is Roy in front of the cassini poster?', False],
'I know Roy is not outside the study': ['Is Roy outside the study?', False],
'I know Roy is not in front of the bookcase': ['Is Roy in front of the bookcase?', False],
'I know Roy is behind the mars poster': ['Is Roy behind the mars poster?', True],
'I know Roy is not inside the study': ['Is Roy inside the study?', False],
'I know Roy is in front of the bookcase': ['Is Roy in front of the bookcase?', True],
'I know Roy is left of the chair': ['Is Roy left of the chair?', True],
'I know Roy is not outside the kitchen': ['Is Roy outside the kitchen?', False],
'I know Roy is inside the kitchen': ['Is Roy inside the kitchen?', True],
'I know Roy is behind the dining table': ['Is Roy behind the dining table?', True],
'I know Roy is not left of the fridge': ['Is Roy left of the fridge?', False],
'I know Roy is not near the billiard room': ['Is Roy near the billiard room?', False],
'I know Roy is not near the chair': ['Is Roy near the chair?', False],
'I know Roy is right of the filing cabinet': ['Is Roy right of the filing cabinet?', True],
'I know Roy is not left of the bookcase': ['Is Roy left of the bookcase?', False],
'I know Roy is right of the cassini poster': ['Is Roy right of the cassini poster?', True],
'I know Roy is near the study': ['Is Roy near the study?', True],
'I know Roy is inside the dining room': ['Is Roy inside the dining room?', True],
'I know Roy is behind the chair': ['Is Roy behind the chair?', True],
'I know Roy is not in front of the dining table': ['Is Roy in front of the dining table?', False],
'I know Roy is behind the fridge': ['Is Roy behind the fridge?', True],
'I know Roy is right of the checkers table': ['Is Roy right of the checkers table?', True],
'I know Roy is not behind the mars poster': ['Is Roy behind the mars poster?', False],
'I know Roy is not left of the cassini poster': ['Is Roy left of the cassini poster?', False],
'I know Roy is near the fern': ['Is Roy near the fern?', True],
'I know Roy is not outside the hallway': ['Is Roy outside the hallway?', False],
'I know Roy is not near the library': ['Is Roy near the library?', False],
'I know Roy is not inside the library': ['Is Roy inside the library?', False],
'I know Roy is not behind the desk': ['Is Roy behind the desk?', False],
'I know Roy is not near the hallway': ['Is Roy near the hallway?', False],
'I know Roy is outside the hallway': ['Is Roy outside the hallway?', True],
'I know Roy is not left of the fern': ['Is Roy left of the fern?', False],
'I know Roy is not behind the checkers table': ['Is Roy behind the checkers table?', False],
'I know Roy is right of the fern': ['Is Roy right of the fern?', True],
'I know Roy is inside the hallway': ['Is Roy inside the hallway?', True],
'I know Roy is left of the checkers table': ['Is Roy left of the checkers table?', True],
'I know Roy is behind the desk': ['Is Roy behind the desk?', True],
'I know Roy is not near the dining room': ['Is Roy near the dining room?', False],
'I know Roy is not right of the checkers table': ['Is Roy right of the checkers table?', False],
'I know Roy is in front of the dining table': ['Is Roy in front of the dining table?', True],
'I know Roy is right of the bookcase': ['Is Roy right of the bookcase?', True],
'I know Roy is not left of the dining table': ['Is Roy left of the dining table?', False],
'I know Roy is not right of the desk': ['Is Roy right of the desk?', False],
'I know Roy is not right of the filing cabinet': ['Is Roy right of the filing cabinet?', False],
'I know Roy is left of the bookcase': ['Is Roy left of the bookcase?', True],
'I know Roy is near the bookcase': ['Is Roy near the bookcase?', True],
'I know Roy is not in front of the fern': ['Is Roy in front of the fern?', False],
'I know Roy is inside the library': ['Is Roy inside the library?', True],
'I know Roy is in front of the checkers table': ['Is Roy in front of the checkers table?', True],
'I know Roy is not near the filing cabinet': ['Is Roy near the filing cabinet?', False],
'I know Roy is not right of the fridge': ['Is Roy right of the fridge?', False],
'I know Roy is near the billiard room': ['Is Roy near the billiard room?', True],
'I know Roy is outside the study': ['Is Roy outside the study?', True],
'I know Roy is not inside the billiard room': ['Is Roy inside the billiard room?', False],
'I know Roy is left of the mars poster': ['Is Roy left of the mars poster?', True],
'I know Roy is right of the fridge': ['Is Roy right of the fridge?', True],
'I know Roy is near the mars poster': ['Is Roy near the mars poster?', True],
'I know Roy is outside the library': ['Is Roy outside the library?', True],
'I know Roy is not in front of the chair': ['Is Roy in front of the chair?', False],
'I know Roy is not near the dining table': ['Is Roy near the dining table?', False],
'I know Roy is not left of the filing cabinet': ['Is Roy left of the filing cabinet?', False],
'I know Roy is near the filing cabinet': ['Is Roy near the filing cabinet?', True],
'I know Roy is not right of the bookcase': ['Is Roy right of the bookcase?', False],
'I know Roy is not near the cassini poster': ['Is Roy near the cassini poster?', False],
'I know Roy is not right of the cassini poster': ['Is Roy right of the cassini poster?', False],
'I know Roy is not near the desk': ['Is Roy near the desk?', False],
'I know Roy is near the cassini poster': ['Is Roy near the cassini poster?', True],
'I know Roy is not behind the bookcase': ['Is Roy behind the bookcase?', False],
'I know Roy is not left of the desk': ['Is Roy left of the desk?', False],
'I know Roy is not outside the dining room': ['Is Roy outside the dining room?', False],
'I know Roy is near the library': ['Is Roy near the library?', True],
'I know Roy is left of the fridge': ['Is Roy left of the fridge?', True],
'I know Roy is not right of the chair': ['Is Roy right of the chair?', False],
'I know Roy is not near the study': ['Is Roy near the study?', False],
'I know Roy is not inside the hallway': ['Is Roy inside the hallway?', False],
'I know Roy is left of the fern': ['Is Roy left of the fern?', True],
'I know Roy is left of the filing cabinet': ['Is Roy left of the filing cabinet?', True],
'I know Roy is right of the mars poster': ['Is Roy right of the mars poster?', True],
'I know Roy is behind the cassini poster': ['Is Roy behind the cassini poster?', True],
'I know Roy is outside the kitchen': ['Is Roy outside the kitchen?', True],
'I know Roy is not behind the fern': ['Is Roy behind the fern?', False],
'I know Roy is not inside the dining room': ['Is Roy inside the dining room?', False],
'I know Roy is not near the bookcase': ['Is Roy near the bookcase?', False],
'I know Roy is inside the study': ['Is Roy inside the study?', True],
'I know Roy is not behind the cassini poster': ['Is Roy behind the cassini poster?', False],
'I know Roy is not in front of the filing cabinet': ['Is Roy in front of the filing cabinet?', False],
'I know Roy is not behind the chair': ['Is Roy behind the chair?', False],
'I know Roy is not right of the fern': ['Is Roy right of the fern?', False],
'I know Roy is not behind the dining table': ['Is Roy behind the dining table?', False],
'I know Roy is not near the mars poster': ['Is Roy near the mars poster?', False],
'I know Roy is in front of the filing cabinet': ['Is Roy in front of the filing cabinet?', True],
'I know Roy is in front of the chair': ['Is Roy in front of the chair?', True],
'I know Roy is behind the filing cabinet': ['Is Roy behind the filing cabinet?', True],
'I know Roy is not behind the filing cabinet': ['Is Roy behind the filing cabinet?', False],
'I know Roy is not inside the kitchen': ['Is Roy inside the kitchen?', False],
'I know Roy is in front of the mars poster': ['Is Roy in front of the mars poster?', True],
'I know Roy is not behind the fridge': ['Is Roy behind the fridge?', False],
'I know Roy is not outside the library': ['Is Roy outside the library?', False],
'I know Roy is not right of the dining table': ['Is Roy right of the dining table?', False],
'I know Roy is left of the desk': ['Is Roy left of the desk?', True],
'I know Roy is in front of the fern': ['Is Roy in front of the fern?', True],
'I know Roy is left of the dining table': ['Is Roy left of the dining table?', True],
'I know Roy is not outside the billiard room': ['Is Roy outside the billiard room?', False],
'I know Roy is outside the billiard room': ['Is Roy outside the billiard room?', True],
'I know Roy is in front of the desk': ['Is Roy in front of the desk?', True],
'I know Roy is behind the fern': ['Is Roy behind the fern?', True],
'I know Roy is near the kitchen': ['Is Roy near the kitchen?', True],
'I know Roy is not near the checkers table': ['Is Roy near the checkers table?', False],
'I know Roy is not in front of the checkers table': ['Is Roy in front of the checkers table?', False],
'I know Roy is not in front of the fridge': ['Is Roy in front of the fridge?', False],
'I know Roy is in front of the fridge': ['Is Roy in front of the fridge?', True],
'I know Roy is left of the cassini poster': ['Is Roy left of the cassini poster?', True],
'I know Roy is inside the billiard room': ['Is Roy inside the billiard room?', True],
'I know Roy is not in front of the mars poster': ['Is Roy in front of the mars poster?', False],
'I know Roy is not left of the chair': ['Is Roy left of the chair?', False],
'I know Roy is outside the dining room': ['Is Roy outside the dining room?', True],
'I know Roy is near the desk': ['Is Roy near the desk?', True],
'I know Roy is near the hallway': ['Is Roy near the hallway?', True],
'I know Roy is near the fridge': ['Is Roy near the fridge?', True],
'I know Roy is not left of the checkers table': ['Is Roy left of the checkers table?', False],
'I know Roy is near the dining table': ['Is Roy near the dining table?', True]}

class Questioner(object):

    def __init__(self,human_sensor,target_order,target_weights,bounds,delta,
                    repeat_annoyance=0.5, repeat_time_penalty=60):
        # rospy.init_node('questioner')

        self.all_likelihoods = np.load(os.path.dirname(__file__) + '/likelihoods.npy')
        self.all_questions = self.all_likelihoods[0:len(self.all_likelihoods)]['question']
        self.target_order = target_order
        self.target_weights = target_weights
        self.repeat_annoyance = repeat_annoyance
        self.repeat_time_penalty = repeat_time_penalty
        self.bounds = bounds
        self.delta = delta

        self.topic = 'robot_questions'
        self.pub = rospy.Publisher(self.topic,Question)

    def weigh_questions(self, priors):
        q_weights = np.empty_like(self.all_questions, dtype=np.float64)
        for prior_name, prior in priors.iteritems():
            if type(prior) is not np.ndarray:
                discretized_prior = prior.discretize2D(low=bounds[0:2],high=bounds[2:4], delta=0.1)
            else:
                discretized_prior = prior
            prior_entropy = self.entropy_calc(discretized_prior)
            # print(prior_entropy.sum())
            # prior_entropy = np.divide(prior_entropy,prior_entropy.sum())
            # print(prior_entropy.sum())
            # flat_prior_pdf = self.flatten(discretized_prior)
            # print(discretized_prior.shape)
            flat_prior_pdf = discretized_prior.flatten()
            # print(type(flat_prior_pdf))
            # print(flat_prior_pdf)

            for i, likelihood_obj in enumerate(self.all_likelihoods):
                question = likelihood_obj['question']
                if prior_name.lower() not in question.lower():
                    continue

                # Use positive and negative answers for VOI
                likelihood = likelihood_obj['probability']
                q_weights[i] = self.calculate_VOI(likelihood, flat_prior_pdf, prior_entropy)

                # Add heuristic question cost based on target weight
                for j, target in enumerate(self.target_order):
                    if target.lower() in question.lower():
                        q_weights[i] *= self.target_weights[j]

                # Add heuristic question cost based on number of times asked
                tla = likelihood_obj['time_last_answered']
                if tla == -1:
                    continue

                dt = time.time() - tla
                if dt > self.repeat_time_penalty:
                    self.all_likelihoods[i]['time_last_answered'] = -1
                elif tla > 0:
                    q_weights[i] *= (dt / self.repeat_time_penalty + 1)\
                         * self.repeat_annoyance

        # Re-order questions by their weights
        q_ids = range(len(self.all_questions))
        self.weighted_questions = zip(q_weights, q_ids, self.all_questions[:])
        self.weighted_questions.sort(reverse=True)
        # for q in self.weighted_questions:
            # print(q)


    def calculate_VOI(self, likelihood, flat_prior_pdf, prior_entropy=None):
        """Calculates the value of a specific question's information.

        VOI is defined as:

        .. math::

            VOI(i) = \\sum_{j \\in {0,1}} P(D_i = j)
                \\left(-\\int p(x \\vert D_i=j) \\log{p(x \\vert D_i=j)}dx\\right)
                +\\int p(x) \\log{p(x)}dx
        """
        if prior_entropy is None:
            prior_entropy = prior.entropy()

        VOI = 0
        grid_spacing = 0.1
        # alpha = self.human_sensor.false_alarm_prob / 2  # only for binary
        alpha = 0.2 / 2
        pos_likelihood = alpha + (1 - alpha) * likelihood
        neg_likelihood = np.ones_like(pos_likelihood) - pos_likelihood
        neg_likelihood = alpha + (1 - alpha) * neg_likelihood
        likelihoods = [neg_likelihood, pos_likelihood]
        for likelihood in likelihoods:
            post_unnormalized = likelihood * flat_prior_pdf
            sensor_marginal = np.sum(post_unnormalized) * grid_spacing ** 2
            log_post_unnormalized = np.log(post_unnormalized)
            log_sensor_marginal = np.log(sensor_marginal)

            VOI += -np.sum(post_unnormalized * (log_post_unnormalized -
                log_sensor_marginal)) * grid_spacing ** 2

        VOI += -prior_entropy
        return - VOI  # keep value positive

    def entropy_calc(self,prior):
        """
        Computes entropy of prior belief. From Cops and Robots 1.0 Questioner
        class in /fusion/question.py written by Nick Sweet.
        """
        # prod = np.multiply(prior,np.log(prior))
        # p_sum = -np.sum(prod * self.delta ** 2)
        H = -np.sum(prior * np.log(prior)) * self.delta ** 2
        return H

    def flatten(self,discretized_prior):
        return np.ndarray.tolist(discretized_prior)

    def get_questions(self,prior):
        """
        Weighs questions given priors, calculates VOI for each question and
        sends questions in ROS message.
        """
        self.weigh_questions({"Roy":prior})
        self.transmit_questions()

    def transmit_questions(self):
        """
        Publishes list of quesitons to ROS topic /robot_questions ordered sorted
        high to low VOI.
        """
        msg = Question()
        msg.weights = [q[0] for q in self.weighted_questions]
        msg.qids = [q[1] for q in self.weighted_questions]
        msg.questions = [q[2] for q in self.weighted_questions]
        # print(msg)
        self.pub.publish(msg)

if __name__ == "__main__":
    prior = GM([[-9, 3],
                 [-8, 3],
                 [-7, 3.5]
                 ],
                 [[[1.5, 1.0],
                   [1.0, 1.5]],
                  [[0.5, -0.3],
                   [-0.3, 0.5]],
                  [[2.5, -0.3],
                   [-0.3, 2.5]]],
                 [0.2, 0.6, 0.2]
                 )

    bounds = [-9.6, -3.6, 4, 3.6]
    delta = 0.1
    prior.plot2D(low=bounds[0:2],high=bounds[2:4])
    q = Questioner(human_sensor=None, target_order=['Pris','Roy'],
                   target_weights=[11., 10.],bounds=bounds,delta=delta)

    q.weigh_questions({'Roy':prior})

    rospy.sleep(3)

    # for qu in q.weighted_questions:
        # print qu

    q.transmit_questions()
