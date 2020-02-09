# Murugeswari
# RDDL Environment

import sys
import os
import random
import ctypes
import numpy as np

import gym

from gym import Env
from gym.utils import seeding

# For instance parser
curr_dir_path = os.path.dirname(os.path.realpath(__file__))
parser_path = os.path.abspath(os.path.join(curr_dir_path, "../../../utils"))
if parser_path not in sys.path:
    sys.path = [parser_path] + sys.path

from parse_instance import InstanceParser


class RDDLEnv(Env):
    def __init__(self, domain, instance):

        self.domain = domain + '_mdp'
        self.problem = domain + '_inst_mdp__' + instance
        self.instance_parser = InstanceParser(domain, instance)

        # Seed Random number generator
        self._seed()

        f = open(
            os.path.abspath(
                os.path.join(
                    os.path.dirname(__file__), './rddl/parsed/',
                    self.problem)))

        p = "##"  # Values of p are hard-coded in PROST. Should not be changed.
        for l in f:
            if (p == "## horizon\n"):
                h = int(l)
            elif (p == "## number of actions\n"
                  and self.domain != 'academic_advising_mdp'):
                num_act = int(l)
            elif (p == "## number of action fluents\n"
                  and self.domain == 'academic_advising_mdp'):
                num_act = int(l) + 1
            elif (p == "## number of det state fluents\n"):
                num_det = int(l)
            elif (p == "## number of prob state fluents\n"):
                num_prob = int(l)
            elif (p == "## initial state\n"):
                init = [int(i) for i in l.split()]
                break
            p = l
        f.close()

        # Problem parameters
        self.num_state_vars = num_det + num_prob  # number of state variables
        self.num_action_vars = self.instance_parser.get_num_actions(
        )  # number of action variables
        self.initial_state = init
        self.state_type = type(self.initial_state)
        self.state = np.array(self.initial_state)  # current state
        self.horizon = h  # episode horizon
        self.tstep = 1  # current time step
        self.done = False  # end_of_episode flag

        # Set up RDDL Simulator clibxx.so

        qwwe = str(instance).split(".|_")
        print(("clibxx{}".format(qwwe[0])))
        self.rddlsim = ctypes.CDLL(
            os.path.abspath(
                os.path.join(
                    os.path.dirname(__file__),
                    './rddl/lib/clibxx{}.so'.format(qwwe[0]))))
        self.rddlsim.step.restype = ctypes.c_double

        # Better without the explicit encoding
        parsed_file_name = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__), './rddl/parsed/', self.problem))
        parsed_file_name_byteobject = parsed_file_name.encode()
        parsed_file_name_ctype = ctypes.create_string_buffer(
            parsed_file_name_byteobject, len(parsed_file_name_byteobject))
        self.rddlsim.parse(parsed_file_name_ctype.value)

    # Do not understand this yet. Almost all other sample environments have it, so we have it too.
    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    # Take a real step in the environment. Current state changes.
    def _step(self, action_var):

        # Convert state and action to c-types
        s = self.state
        ss = s.tolist()
        sss = (ctypes.c_double * len(ss))(*ss)
        action = (ctypes.c_int)(action_var)

        # Call Simulator
        reward = self.rddlsim.step(sss, len(ss), action)
        self.state = np.array(sss, dtype=np.int8)
        # if self.domain == "academic_advising_mdp":
        #     pass
        #     if action_var > 0:
        #         if reward == -5:
        #             pass
        #         elif reward == -6:
        #             if self.instance_parser.prog_requirement[action_var -
        #                                                      1] == 1:
        #                 reward = -3
        #             else:
        #                 reward = -5

        #             flag_change = False
        #             flag_all = True
        #             for i in range(int(len(s) / 2), len(s)):
        #                 if self.instance_parser.prog_requirement[
        #                         i - int(len(s) / 2)] == 1:
        #                     if s[i] == 0 and self.state[i] == 1:
        #                         flag_change = True

        #                     if self.state[i] == 0:
        #                         flag_all = False

        #             if flag_all and flag_change:
        #                 reward = 400

        #         elif reward == -7:
        #             if self.instance_parser.prog_requirement[action_var -
        #                                                      1] == 0:
        #                 reward = -9

        #             flag_change = False
        #             flag_all = True
        #             for i in range(int(len(s) / 2), len(s)):
        #                 if self.instance_parser.prog_requirement[
        #                         i - int(len(s) / 2)] == 1:
        #                     if s[i] == 0 and self.state[i] == 1:
        #                         flag_change = True

        #                     if self.state[i] == 0:
        #                         flag_all = False

        #             if flag_all and flag_change:
        #                 reward = 200

        #         elif reward == -1:
        #             pass
        #         elif reward == -2:
        #             pass
        #         elif reward == 0:
        #             pass
        # elif self.domain == "navigation_mdp":
        #     if reward == 0:
        #         pass
        #     else:
        #         if not np.any(self.state == 1):
        #             reward = -10
        #         elif (not np.all(s == self.state)
        #               ) and np.all(self.state == np.array(
        #                   self.instance_parser.goals, dtype=np.int32)):
        #             reward = 100

        # Advance time step
        self.tstep = self.tstep + 1
        if self.domain == "navigation_mdp":
            if self.tstep > 2 * self.horizon:
                self.done = True
        else:
            if self.tstep > self.horizon:
                self.done = True

        # # Handle episode end in case of navigation
        # # Not able to check if robot's position is same as goal state
        # if self.domain == "navigation_mdp" and not(np.any(self.state)):
        #     self.done = True

        return self.state, reward, self.done, {}

    def test_step(self, action_var):

        # Convert state and action to c-types
        s = self.state
        ss = s.tolist()
        sss = (ctypes.c_double * len(ss))(*ss)
        action = (ctypes.c_int)(action_var)

        # Call Simulator
        reward = self.rddlsim.step(sss, len(ss), action)
        self.state = np.array(sss, dtype=np.int8)

        # Advance time step
        self.tstep = self.tstep + 1
        if self.domain == "navigation_mdp":
            if self.tstep > 2 * self.horizon:
                self.done = True
        else:
            if self.tstep > self.horizon:
                self.done = True

        # # Handle episode end in case of navigation
        # # Not able to check if robot's position is same as goal state
        # if self.domain == "navigation_mdp" and not(np.any(self.state)):
        #     self.done = True

        return self.state, reward, self.done, {}

    # Take an imaginary step to get the next state and reward. Current state does not change.
    def pseudostep(self, curr_state, action_var):

        # Convert state and action to c-types
        s = np.array(curr_state)
        ss = s.tolist()
        sss = (ctypes.c_double * len(ss))(*ss)
        action = (ctypes.c_int)(action_var)

        # Call Simulator
        reward = self.rddlsim.step(sss, len(ss), action)
        next_state = np.array(sss, dtype=np.int8)

        return next_state, reward

    def _reset(self):
        self.state = np.array(self.initial_state)
        self.tstep = 1
        self.done = False
        return self.state, self.done

    def random_reset(self):
        self.state = np.array(self.initial_state)

        for i in range(random.randint(0, 5)):
            ran_act = random.randint(0, self.num_action_vars - 1)
            self.step(ran_act)

        return self.state, self.done

    def random_state(self):
        if self.domain == "sysadmin_mdp":
            return np.random.choice(2, self.num_state_vars)
        elif self.domain == "game_of_life_mdp":
            return np.random.choice(2, self.num_state_vars)
        elif self.domain == "navigation_mdp":
            pass

    def _set_state(self, state):
        self.state = state

    def _close(self):
        print("Environment Closed")

    def get_fluent_features(self, state):
        return np.array(self.instance_parser.get_fluent_features(state))

    def get_action_details(self):
        return self.instance_parser.get_action_details()

    def get_nf_features(self):
        return np.array(self.instance_parser.get_nf_features())

    def get_adjacency_list(self):
        return self.instance_parser.get_adjacency_list()

    def get_num_adjacency_list(self):
        return self.instance_parser.get_num_adjacency_list()

    def get_feature_dims(self):
        return self.instance_parser.get_feature_dims()

    def get_num_actions(self):
        return self.instance_parser.get_num_actions()

    def get_graph_fluent_features(self, state):
        return np.array(self.instance_parser.get_graph_fluent_features(state))

    def get_num_nodes(self):
        return self.instance_parser.get_num_nodes()

    def get_num_graph_fluents(self):
        return self.instance_parser.get_num_graph_fluents()

    def get_num_type_actions(self):
        return self.instance_parser.get_num_type_actions()

    def print_domain(self, state):
        self.instance_parser.print_domain(state)

    def print_action_probs(self, action_probs, action):
        self.instance_parser.print_action_probs(action_probs, action)


if __name__ == '__main__':
    ENV = gym.make('RDDL-v1')
    ENV.seed(0)

    NUM_EPISODES = 1

    for i in range(NUM_EPISODES):
        reward = 0  # epsiode reward
        rwd = 0  # step reward
        curr, done = ENV.reset()  # current state and end-of-episode flag
        while not done:
            action = random.randint(
                0, ENV.num_action_vars)  # choose a random action
            # action = 0
            nxt, rwd, done, _ = ENV.step(action)  # next state and step reward
            print(('state: {}  action: {}  reward: {} next: {}'.format(
                curr, action, rwd, nxt)))
            curr = nxt
            reward += rwd
        print(('Episode Reward: {}'.format(reward)))
        print()

    ENV.close()
