#!/usr/bin/env python

from __future__ import division


"""
Compute VOI of all possible questions in experiment based on the entropy of the
current belief.
"""

import numpy as np
import rospy

from gaussianMixtures import GM

from policy_translator.msg import *

class Questioner(object):

    def __init__(self,human_sensor,target_order,target_weights,bounds,delta,
                    repeat_annoyance=0.5, repeat_time_penalty=60):
        rospy.init_node('questioner')

        self.all_likelihoods = np.load('likelihoods.npy')
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
            discretized_prior = prior.discretize2D(low=bounds[0:2],high=bounds[2:4], delta=0.1)
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

    def transmit_questions(self):
        """
        Publishes list of quesitons to ROS topic /robot_questions ordered sorted
        high to low VOI.
        """
        msg = Question()
        # msg.weights = self.weighted_questions[0:len(self.weighted_questions)][0]
        msg.weights = [q[0] for q in self.weighted_questions]
        # print([q[0] for q in self.weighted_questions])
        msg.qids = [q[1] for q in self.weighted_questions]
        # msg.qids = self.weighted_questions[0:len(self.weighted_questions)][1]
        # msg.questions = self.weighted_questions[0:len(self.weighted_questions)][2]
        msg.questions = [q[2] for q in self.weighted_questions]
        print(msg)
        self.pub.publish(msg)

if __name__ == "__main__":
    prior = GM([[1, 1],
                 [0, 0],
                 [-4, -2]
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
