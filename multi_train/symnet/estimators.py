import pickle as pkl
import numpy as np
import networkx as nx
import scipy.sparse as sp
import tensorflow as tf

from gat.models import GAT


class PolicyEstimator():
    """
    Policy Function approximator. Given a observation, returns probabilities
    over all possible actions.

    Args:
      num_outputs: Size of the action space.
      reuse: If true, an existing shared network will be re-used.
      trainable: If true we add train ops to the network.
        Actor threads that don't update their local models and don't need
        train ops would set this to false.
    """

    def __init__(self,
                 num_nodes_list,
                 fluent_feature_dims,
                 nonfluent_feature_dims,
                 N,
                 num_valid_actions_list,
                 action_details_list,
                 num_graph_fluent_list,
                 num_gcn_hidden,
                 num_action_dim,
                 num_decoder_dim,
                 num_adjacency_list,
                 num_gat_layers=1,
                 activation="lrelu",
                 learning_rate=0.001,
                 reuse=False,
                 trainable=True):

        self.num_nodes_list = num_nodes_list

        self.fluent_feature_dims = fluent_feature_dims
        self.nonfluent_feature_dims = nonfluent_feature_dims
        self.feature_dims = fluent_feature_dims + nonfluent_feature_dims

        self.num_gcn_hidden = num_gcn_hidden
        self.num_action_dim = num_action_dim
        self.num_decoder_dim = num_decoder_dim
        self.activation = activation
        self.num_gat_layers = num_gat_layers
        self.num_adjacency_list = num_adjacency_list

        self.num_valid_actions_list = num_valid_actions_list
        self.action_details_list = action_details_list

        self.num_graph_fluent_list = num_graph_fluent_list

        if activation == "relu":
            self.activation_fn = tf.nn.relu
        if activation == "lrelu":
            self.activation_fn = tf.nn.leaky_relu
        if activation == "elu":
            self.activation_fn = tf.nn.elu

        self.N = N
        self.learning_rate = learning_rate
        self.lambda_entropy = 1.0

        # Placeholders for our input

        self.instance = tf.placeholder(
            shape=[None], dtype=tf.int32, name="instance")

        self.batch_size = tf.placeholder(tf.int32, name="batch_size")
        self.env_num = tf.placeholder(tf.int32, name="env_num")

        tf.summary.scalar("env_num", self.env_num)

        # The TD target value
        self.targets = tf.placeholder(shape=[None], dtype=tf.float32, name="y")
        # Integer id of which action was selected
        self.actions = tf.placeholder(
            shape=[None], dtype=tf.int32, name="actions")

        self.inputs_list = [None] * self.N
        self.graph_input_fluent_list = [None] * self.N
        self.adj_biases_placeholder_list = [None] * self.N

        for i in range(self.N):
            self.inputs_list[i] = tf.placeholder(
                dtype=tf.float32,
                shape=(None, self.num_nodes_list[i], self.feature_dims),
                name="inputs_{}".format(i))

            self.graph_input_fluent_list[i] = tf.placeholder(
                dtype=tf.float32,
                shape=(None, self.num_graph_fluent_list[i]),
                name="input_graph_fluent_{}".format(i))

            self.adj_biases_placeholder_list[i] = [
                tf.placeholder(
                    dtype=tf.float32,
                    shape=(None, self.num_nodes_list[i],
                           self.num_nodes_list[i]),
                    name="adj_biases_placeholder_{}_{}".format(i, zz))
                for zz in range(self.num_adjacency_list[i])
            ]

        self.is_train = tf.placeholder(
            dtype=tf.bool, shape=(), name="is_train")

        self.gcn_hidden_list = [None] * self.N
        self.action_embedding1_list = [None] * self.N
        self.graph_embedding_list = [None] * self.N
        self.predicted_action_list = [None] * self.N

        self.probs_list = [None] * self.N
        self.predictions_list = [None] * self.N
        self.entropy_list = [None] * self.N
        self.entropy_mean_list = [None] * self.N
        self.picked_action_probs_list = [None] * self.N
        self.losses_list = [None] * self.N
        self.loss_list = [None] * self.N
        self.grads_and_vars_list = [None] * self.N
        self.train_op_list = [None] * self.N

        with tf.variable_scope("policy_net", reuse=tf.AUTO_REUSE):

            for env_i in range(self.N):
                with tf.variable_scope("gat1", reuse=tf.AUTO_REUSE):
                    gat1 = [None] * self.num_adjacency_list[env_i]

                    for gat_i in range(self.num_adjacency_list[env_i]):
                        gat1[gat_i] = GAT.inference(
                            self.inputs_list[env_i],
                            self.num_gcn_hidden,
                            self.num_nodes_list[env_i],
                            self.is_train,
                            0.0,
                            0.0,
                            bias_mat=self.adj_biases_placeholder_list[env_i][
                                gat_i],
                            hid_units=[8],
                            n_heads=[8, 1],
                            residual=False,
                            activation=self.activation_fn)

                # with tf.variable_scope("gat2", reuse=tf.AUTO_REUSE):
                #     gat2 = [None] * self.num_gat_layers
                #     gat2[0] = GAT.inference(
                #         self.inputs_list[env_i],
                #         self.num_gcn_hidden,
                #         self.num_nodes_list[env_i],
                #         self.is_train,
                #         0.0,
                #         0.0,
                #         bias_mat=tf.transpose(
                #             self.adj_biases_placeholder_list[env_i],
                #             perm=[0, 2, 1]),
                #         hid_units=[8],
                #         n_heads=[8, 1],
                #         residual=False,
                #         activation=self.activation_fn)

                #     for gat_i in range(1, self.num_gat_layers):
                #         gat2[gat_i] = GAT.inference(
                #             gat2[gat_i - 1],
                #             self.num_gcn_hidden,
                #             self.num_nodes_list[env_i],
                #             self.is_train,
                #             0.0,
                #             0.0,
                #             bias_mat=tf.transpose(
                #                 self.adj_biases_placeholder_list[env_i],
                #                 perm=[0, 2, 1]),
                #             hid_units=[8],
                #             n_heads=[8, 1],
                #             residual=False,
                #             activation=self.activation_fn)

                self.gcn_hidden_list[env_i] = tf.concat(gat1, axis=-1)

                self.action_embedding1_list[env_i] = tf.layers.dense(
                    inputs=self.gcn_hidden_list[env_i],
                    units=self.num_action_dim,
                    activation=self.activation_fn,
                    name="action_embedding1",
                    reuse=tf.AUTO_REUSE)

                action_embedding1_flat = tf.reshape(
                    self.action_embedding1_list[env_i], [
                        self.batch_size, self.num_nodes_list[env_i],
                        self.num_action_dim
                    ])

                self.graph_embedding_list[env_i] = tf.concat(
                    [
                        tf.reduce_max(action_embedding1_flat, axis=1),
                        self.graph_input_fluent_list[env_i]
                    ],
                    axis=1)

                graph_embedding_repeat = tf.reshape(
                    tf.tile(self.graph_embedding_list[env_i],
                            [1, self.num_nodes_list[env_i]]),
                    tf.shape(self.action_embedding1_list[env_i]))

                action_embedding2_list = [
                    None
                ] * self.num_valid_actions_list[env_i]

                action_embedding3_list = [
                    None
                ] * self.num_valid_actions_list[env_i]

                # graph action

                detailed_action = self.action_details_list[env_i]

                for k in range(self.num_valid_actions_list[env_i]):

                    action_template = detailed_action[k][0]
                    input_nodes = list(detailed_action[k][1])

                    if len(input_nodes) == 0:
                        action_embedding2_list[k] = tf.layers.dense(
                            inputs=self.graph_embedding_list[env_i],
                            units=self.num_decoder_dim,
                            activation=self.activation_fn,
                            name="action_embedding2_{}".format(
                                action_template),
                            reuse=tf.AUTO_REUSE)

                    else:
                        temp_embedding_list = [
                            tf.reshape(
                                self.action_embedding1_list[env_i][:, inp, :],
                                [self.batch_size, self.num_action_dim])
                            for inp in input_nodes
                        ]

                        node_state_embedding_concat = tf.concat(
                            temp_embedding_list, axis=1)

                        node_state_embedding_reshape = tf.reshape(
                            node_state_embedding_concat, [
                                self.batch_size,
                                len(input_nodes), self.num_action_dim
                            ])

                        node_state_embedding_pooled = tf.reshape(
                            tf.reduce_max(
                                node_state_embedding_reshape, axis=1),
                            [self.batch_size, self.num_action_dim])

                        action_embedding2_list[k] = tf.layers.dense(
                            inputs=node_state_embedding_pooled,
                            units=self.num_decoder_dim,
                            activation=self.activation_fn,
                            name="action_embedding2_{}".format(
                                action_template),
                            reuse=tf.AUTO_REUSE)

                    action_embedding3_list[k] = tf.layers.dense(
                        inputs=action_embedding2_list[k],
                        units=1,
                        activation=self.activation_fn,
                        name="action_embedding3_{}".format(action_template),
                        reuse=tf.AUTO_REUSE)

                reshaped_action_list = [None
                                        ] * self.num_valid_actions_list[env_i]

                for j in range(self.num_valid_actions_list[env_i]):
                    reshaped_action_list[j] = tf.reshape(
                        action_embedding3_list[j], [self.batch_size, 1])

                self.predicted_action_list[env_i] = tf.concat(
                    reshaped_action_list,
                    axis=1,
                    name="predicted_action_{}".format(i))

                self.probs_list[env_i] = tf.nn.softmax(
                    self.predicted_action_list[env_i]) + 1e-8

                self.predictions_list[env_i] = {
                    "logits": self.predicted_action_list[env_i],
                    "probs": self.probs_list[env_i]
                }

                # We add entropy to the loss to encourage exploration
                self.entropy_list[env_i] = - \
                    tf.reduce_sum(self.probs_list[env_i] * tf.log(self.probs_list[env_i]),
                                1, name="entropy_{}".format(env_i))
                self.entropy_mean_list[env_i] = tf.reduce_mean(
                    self.entropy_list[env_i],
                    name="entropy_mean_{}".format(env_i))

                # Get the predictions for the chosen actions only
                gather_indices = tf.range(self.batch_size) * \
                    tf.shape(self.probs_list[env_i])[1] + self.actions
                self.picked_action_probs_list[env_i] = tf.gather(
                    tf.reshape(self.probs_list[env_i], [-1]), gather_indices)

                self.losses_list[env_i] = -(
                    tf.log(self.picked_action_probs_list[env_i]) * self.targets
                    + self.lambda_entropy * self.entropy_list[env_i])
                self.loss_list[env_i] = tf.reduce_sum(
                    self.losses_list[env_i], name="loss_{}".format(env_i))

                if trainable:
                    # self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
                    self.optimizer = tf.train.RMSPropOptimizer(
                        self.learning_rate, 0.99, 0.0, 1e-6)
                    self.grads_and_vars_list[
                        env_i] = self.optimizer.compute_gradients(
                            self.loss_list[env_i])
                    self.grads_and_vars_list[env_i] = [[
                        grad, var
                    ] for grad, var in self.grads_and_vars_list[env_i]
                                                       if grad is not None]
                    self.train_op_list[env_i] = self.optimizer.apply_gradients(
                        self.grads_and_vars_list[env_i],
                        global_step=tf.train.get_global_step())

        # Merge summaries from this network and the shared network (but not the value net)
        var_scope_name = tf.get_variable_scope().name
        summary_ops = tf.get_collection(tf.GraphKeys.SUMMARIES)
        sumaries = [
            s for s in summary_ops
            if "policy_net" in s.name or "shared" in s.name
        ]
        sumaries = [s for s in summary_ops if var_scope_name in s.name]
        self.summaries = tf.summary.merge(sumaries)


class ValueEstimator():
    def __init__(self,
                 num_nodes_list,
                 fluent_feature_dims,
                 nonfluent_feature_dims,
                 N,
                 num_graph_fluent_list,
                 num_gcn_hidden,
                 num_action_dim,
                 num_decoder_dim,
                 num_adjacency_list,
                 num_gat_layers=1,
                 activation="lrelu",
                 learning_rate=0.001,
                 reuse=False,
                 trainable=True):
        self.num_nodes_list = num_nodes_list
        self.fluent_feature_dims = fluent_feature_dims
        self.nonfluent_feature_dims = nonfluent_feature_dims
        self.feature_dims = fluent_feature_dims + nonfluent_feature_dims

        self.num_gcn_hidden = num_gcn_hidden
        self.num_action_dim = num_action_dim
        self.num_decoder_dim = num_decoder_dim
        self.num_outputs = 1
        self.activation = activation
        self.num_gat_layers = num_gat_layers
        self.num_adjacency_list = num_adjacency_list

        self.num_graph_fluent_list = num_graph_fluent_list

        if activation == "relu":
            self.activation_fn = tf.nn.relu
        if activation == "lrelu":
            self.activation_fn = tf.nn.leaky_relu
        if activation == "elu":
            self.activation_fn = tf.nn.elu

        self.N = N
        self.learning_rate = learning_rate

        # The TD target value
        self.targets = tf.placeholder(shape=[None], dtype=tf.float32, name="y")
        self.batch_size = tf.placeholder(tf.int32, name="batch_size")
        self.env_num = tf.placeholder(tf.int32, name="env_num")
        # Build network
        # TODO: add support

        self.inputs_list = [None] * self.N
        self.graph_input_fluent_list = [None] * self.N
        self.adj_biases_placeholder_list = [None] * self.N

        for i in range(self.N):
            self.inputs_list[i] = tf.placeholder(
                dtype=tf.float32,
                shape=(None, self.num_nodes_list[i], self.feature_dims),
                name="inputs_{}".format(i))
            self.graph_input_fluent_list[i] = tf.placeholder(
                dtype=tf.float32,
                shape=(None, self.num_graph_fluent_list[i]),
                name="input_graph_fluent_{}".format(i))

            self.adj_biases_placeholder_list[i] = [
                tf.placeholder(
                    dtype=tf.float32,
                    shape=(None, self.num_nodes_list[i],
                           self.num_nodes_list[i]),
                    name="adj_biases_placeholder_{}_{}".format(i, zz))
                for zz in range(self.num_adjacency_list[i])
            ]

        self.is_train = tf.placeholder(
            dtype=tf.bool, shape=(), name="is_train")

        self.hidden_list = [None] * self.N
        self.logits_list = [None] * self.N
        self.predictions_list = [None] * self.N
        self.losses_list = [None] * self.N
        self.loss_list = [None] * self.N
        self.grads_and_vars_list = [None] * self.N
        self.train_op_list = [None] * self.N

        # Build network
        with tf.variable_scope("value_net", reuse=tf.AUTO_REUSE):

            for i in range(self.N):
                with tf.variable_scope("gat1", reuse=tf.AUTO_REUSE):
                    gat1 = [None] * self.num_adjacency_list[i]

                    for gat_i in range(self.num_adjacency_list[i]):
                        gat1[gat_i] = GAT.inference(
                            self.inputs_list[i],
                            self.num_gcn_hidden,
                            self.num_nodes_list[i],
                            self.is_train,
                            0.0,
                            0.0,
                            bias_mat=self.adj_biases_placeholder_list[i][
                                gat_i],
                            hid_units=[8],
                            n_heads=[8, 1],
                            residual=False,
                            activation=self.activation_fn)

                # with tf.variable_scope("gat2", reuse=tf.AUTO_REUSE):
                #     gat2 = [None] * self.num_gat_layers
                #     gat2[0] = GAT.inference(
                #         self.inputs_list[i],
                #         self.num_gcn_hidden,
                #         self.num_nodes_list[i],
                #         self.is_train,
                #         0.0,
                #         0.0,
                #         bias_mat=tf.transpose(
                #             self.adj_biases_placeholder_list[i],
                #             perm=[0, 2, 1]),
                #         hid_units=[8],
                #         n_heads=[8, 1],
                #         residual=False,
                #         activation=self.activation_fn)

                #     for gat_i in range(1, self.num_gat_layers):
                #         gat2[gat_i] = GAT.inference(
                #             gat2[gat_i - 1],
                #             self.num_gcn_hidden,
                #             self.num_nodes_list[i],
                #             self.is_train,
                #             0.0,
                #             0.0,
                #             bias_mat=tf.transpose(
                #                 self.adj_biases_placeholder_list[i],
                #                 perm=[0, 2, 1]),
                #             hid_units=[8],
                #             n_heads=[8, 1],
                #             residual=False,
                #             activation=self.activation_fn)

                gcn_hidden = tf.concat(gat1, axis=-1)

                embedding1 = tf.layers.dense(
                    inputs=gcn_hidden,
                    units=self.num_action_dim,
                    activation=self.activation_fn,
                    name="embedding1",
                    reuse=tf.AUTO_REUSE)

                embedding1_flat = tf.reshape(embedding1, [
                    self.batch_size, self.num_nodes_list[i],
                    self.num_action_dim
                ])

                graph_embedding = tf.concat(
                    [
                        tf.reduce_max(embedding1_flat, axis=1),
                        self.graph_input_fluent_list[i]
                    ],
                    axis=1)

                graph_embedding_repeat = tf.reshape(
                    tf.tile(graph_embedding, [1, self.num_nodes_list[i]]), [
                        self.batch_size, self.num_nodes_list[i],
                        self.num_action_dim + self.num_graph_fluent_list[i]
                    ])

                node_state_embedding_concat = tf.concat(
                    values=[embedding1, graph_embedding_repeat], axis=2)

                embedding2 = tf.layers.dense(
                    inputs=node_state_embedding_concat,
                    units=self.num_decoder_dim,
                    activation=self.activation_fn,
                    name="embedding2",
                    reuse=tf.AUTO_REUSE)

                embedding3 = tf.layers.dense(
                    inputs=embedding2,
                    units=1,
                    activation=self.activation_fn,
                    name="embedding3",
                    reuse=tf.AUTO_REUSE)

                # Common summaries
                prefix = tf.get_variable_scope().name
                # tf.contrib.layers.summarize_activation(self.gcn_hidden1)
                # tf.summary.scalar("{}/reward_max".format(prefix),
                #                   tf.reduce_max(self.targets))
                # tf.summary.scalar("{}/reward_min".format(prefix),
                #                   tf.reduce_min(self.targets))
                if self.N > 1:
                    tf.summary.scalar("{}/reward_mean".format(prefix),
                                      tf.reduce_mean(self.targets))
                # tf.summary.histogram("{}/reward_targets".format(prefix),
                #                      self.targets)

                self.hidden_list[i] = tf.reshape(
                    embedding3, [self.batch_size, self.num_nodes_list[i]])

                self.logits_list[i] = tf.reduce_sum(
                    self.hidden_list[i], axis=1, name="logits_{}".format(i))

                self.losses_list[i] = tf.squared_difference(
                    self.logits_list[i], self.targets)
                self.loss_list[i] = tf.reduce_mean(
                    self.losses_list[i], name="loss_{}".format(i))

                self.predictions_list[i] = {"logits": self.logits_list[i]}

                # Summaries
                if self.N == 1:
                    tf.summary.scalar(self.loss_list[i].name,
                                      self.loss_list[i])

                if trainable:
                    # self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
                    self.optimizer = tf.train.RMSPropOptimizer(
                        self.learning_rate, 0.99, 0.0, 1e-6)
                    self.grads_and_vars_list[
                        i] = self.optimizer.compute_gradients(
                            self.loss_list[i])
                    self.grads_and_vars_list[i] = [[
                        grad, var
                    ] for grad, var in self.grads_and_vars_list[i]
                                                   if grad is not None]
                    self.train_op_list[i] = self.optimizer.apply_gradients(
                        self.grads_and_vars_list[i],
                        global_step=tf.train.get_global_step())

        var_scope_name = tf.get_variable_scope().name
        summary_ops = tf.get_collection(tf.GraphKeys.SUMMARIES)
        sumaries = [
            s for s in summary_ops
            if "value_net" in s.name or "shared" in s.name
        ]
        sumaries = [s for s in summary_ops if var_scope_name in s.name]
        self.summaries = tf.summary.merge(sumaries)
