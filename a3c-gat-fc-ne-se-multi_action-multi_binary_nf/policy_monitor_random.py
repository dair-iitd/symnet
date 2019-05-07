import sys
import os
import copy
import itertools
import collections
import numpy as np
import networkx as nx
import scipy.sparse as sp
import tensorflow as tf
import time

from inspect import getsourcefile
current_path = os.path.dirname(os.path.abspath(getsourcefile(lambda: 0)))
import_path = os.path.abspath(os.path.join(current_path, ".."))

if import_path not in sys.path:
    sys.path.append(import_path)

from gym.wrappers import Monitor
import gym

from estimators import PolicyEstimator
from worker import make_copy_params_op
from parse_instance import InstanceParser
from gat.utils import process
from utils.utilmodule import get_adj_mat_from_list


class PolicyMonitor(object):
    def __init__(self, envs, num_valid_actions_list, domain, instances):

        self.domain = domain
        self.instances = instances
        self.N = len(instances)
        self.num_valid_actions_list = num_valid_actions_list

        self.envs = envs

    def test_eval(self, all_env=-1):

        num_episodes = 200
        mean_total_rewards = []
        mean_episode_lengths = []

        if all_env == -1:
            all_env = self.N
        for i in range(all_env):
            rewards_i = []
            episode_lengths_i = []

            for _ in range(num_episodes):
                # Run an episode
                initial_state, done = self.envs[i].reset()
                state = initial_state
                episode_reward = 0.0
                episode_length = 0
                while not done:
                    action = np.random.randint(0,
                                               self.num_valid_actions_list[i])
                    next_state, reward, done, _ = self.envs[i].test_step(
                        action)
                    episode_reward += reward
                    episode_length += 1
                    state = next_state
                rewards_i.append(episode_reward)
                episode_lengths_i.append(episode_length)

            mean_total_reward = sum(rewards_i) / float(len(rewards_i))
            mean_episode_length = sum(episode_lengths_i) / float(
                len(episode_lengths_i))

            mean_total_rewards.append(mean_total_reward)
            mean_episode_lengths.append(mean_episode_length)

        print("mean total rewards are: {}".format(mean_total_rewards))

        return mean_total_rewards, mean_episode_lengths
