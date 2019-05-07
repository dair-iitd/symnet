import better_exceptions

import unittest
import sys
import os
import numpy as np
import tensorflow as tf
import itertools
import shutil
import threading
import multiprocessing
from pprint import pprint

curr_dir_path = os.path.dirname(os.path.realpath(__file__))
gym_path = os.path.abspath(os.path.join(curr_dir_path, "../.."))
if gym_path not in sys.path:
    sys.path = [gym_path] + sys.path
parser_path = os.path.abspath(os.path.join(curr_dir_path, "../../utils"))
if parser_path not in sys.path:
    sys.path = [parser_path] + sys.path
import gym

from policy_monitor_random import PolicyMonitor
from parse_instance import InstanceParser

# os.environ['CUDA_VISIBLE_DEVICES'] = ''

tf.flags.DEFINE_string("restore_dir", None,
                       "Directory to write Tensorboard summaries to.")
tf.flags.DEFINE_string("model_dir", "./test",
                       "Directory to write Tensorboard summaries to.")
tf.flags.DEFINE_string("domain", None, "Name of domain")
tf.flags.DEFINE_string("train_instance", None, "Name of instance")
tf.flags.DEFINE_string("test_instance", None, "Name of test instance")
tf.flags.DEFINE_integer("num_instances", None, "Name of number of instances")
tf.flags.DEFINE_integer("num_features", 3, "Number of features in graph CNN")
tf.flags.DEFINE_integer("num_action_dim", 20, "num action dim")
tf.flags.DEFINE_integer("num_decoder_dim", 20, "num decoder dim")
tf.flags.DEFINE_integer("neighbourhood", 1, "Number of features in graph CNN")
tf.flags.DEFINE_string("activation", "elu", "Activation function")
tf.flags.DEFINE_string("gpuid", None, "Activation function")
tf.flags.DEFINE_float("lr", 5e-5, "Learning rate")
tf.flags.DEFINE_integer("t_max", 20,
                        "Number of steps before performing an update")
tf.flags.DEFINE_integer(
    "max_global_steps", None,
    "Stop training after this many steps in the environment. Defaults to running indefinitely."
)
tf.flags.DEFINE_boolean("lr_decay", False, "If set, decay learning rate")
tf.flags.DEFINE_boolean(
    "use_pretrained", False,
    "If set, load weights from pretrained model (if found in checkpoint dir).")
tf.flags.DEFINE_integer("eval_every", 300,
                        "Evaluate the policy every N seconds")
tf.flags.DEFINE_boolean(
    "reset", False,
    "If set, delete the existing model directory and start training from scratch."
)
tf.flags.DEFINE_integer(
    "parallelism", None,
    "Number of threads to run. If not set we run [num_cpu_cores] threads.")

FLAGS = tf.flags.FLAGS

instances = []
for instance_num in FLAGS.train_instance.strip().split(","):
    if FLAGS.num_instances > 0:
        for i in range(FLAGS.num_instances):
            instance = "{}.{}".format(instance_num, (i + 1))
            instances.append(instance)
    else:
        instance = "{}".format(instance_num)
        instances.append(instance)

N_train_instances = len(instances)

for instance_num in FLAGS.test_instance.strip().split(","):
    if FLAGS.num_instances > 0:
        for i in range(FLAGS.num_instances):
            instance = "{}.{}".format(instance_num, (i + 1))
            instances.append(instance)
    else:
        instance = "{}".format(instance_num)
        instances.append(instance)

FLAGS.num_instances = len(instances)


def make_envs():
    envs = []
    for i in range(FLAGS.num_instances):
        env_name = "RDDL-{}{}-v1".format(FLAGS.domain, instances[i])
        env = gym.make(env_name)
        envs.append(env)
    return envs


# action space
print("Finding state and action parameters from env")

envs_ = make_envs()

num_nodes_list = []
num_valid_actions_list = []
action_details_list = []
num_graph_fluent_list = []
num_type_actions = None

for env_ in envs_:
    num_type_actions = env_.get_num_type_actions()
    num_nodes_list.append(env_.get_num_nodes())
    num_valid_actions_list.append(env_.get_num_actions())
    action_details_list.append(env_.get_action_details())
    num_graph_fluent_list.append(env_.get_num_graph_fluents())

# nn hidden layer parameters
policy_num_gcn_hidden = FLAGS.num_features
value_num_gcn_hidden = FLAGS.num_features

# Number of input features
fluent_feature_dims, nonfluent_feature_dims = envs_[0].get_feature_dims()

print('fluent features')
pprint(fluent_feature_dims)
print('non fluent features')
pprint(nonfluent_feature_dims)
print('nodes list')
pprint(num_nodes_list)
print('num valid actions list')
pprint(num_valid_actions_list)
print('action details')
pprint(action_details_list)
print('graph fluent')
pprint(num_graph_fluent_list)

for e in envs_:
    e.close()

    # Used to occasionally write episode rewards to Tensorboard
pe = PolicyMonitor(
    envs=make_envs(),
    num_valid_actions_list=num_valid_actions_list,
    domain=FLAGS.domain,
    instances=instances)

pe.test_eval()