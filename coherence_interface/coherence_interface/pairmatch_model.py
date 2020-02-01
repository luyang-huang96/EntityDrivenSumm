# File: sentence pair matching model for coherence measurement
# -*- coding: utf-8 -*-
# @Time    : 4/12/2019 2:25 PM
# @Author  : Derek Hu

from collections import namedtuple
import numpy as np
import tensorflow as tf
from tensorflow.contrib.cudnn_rnn.python.ops import cudnn_rnn_ops
import re
import argparse

max_length = 50  # hypothesis
l2_lambda = 0.0001


def model_para(parser):
    parser.add_argument('--sm_conv1d_filter', default=128, type=int, help='filter num for conv1d')
    parser.add_argument('--sm_conv1d_width', default=3, type=int, help='filter width for conv1d')
    parser.add_argument('--sm_conv_filters', default=256, type=int, help='filter num for conv2d')
    parser.add_argument('--sm_conv_heights', default=3, type=int, help='filter height for conv2d')
    parser.add_argument('--sm_conv_widths', default=3, type=int, help='filter width for conv2d')
    parser.add_argument('--sm_maxpool_width', default=2, type=int, help='width for maxpooling')
    parser.add_argument('--sm_fc_num_units', default=256, type=int, help='Number of units in FC layers.')
    parser.add_argument('--sm_margin', default=1.0, type=int, help='Margin for ranking loss')

def parse_list_str(value):
  l = [int(value)]
  if not l:
    raise ValueError("List is empty.")
  return l

class SeqMatchNet(object):
    def __init__(self, trained_wordvec, word_vector, vocab_size, args=None):
        # parser = argparse.ArgumentParser()
        # model_para(parser)
        # model_params = parser.parse_args()
  
        self._device_0 = None
        self.trained_wordvec = trained_wordvec
        self.word_vector = word_vector
        self.vocab_szie = vocab_size
        #self.emb_size = args.embedding_size
        self.emb_size = 300
        # params: will be re-ordered in the future
        #self.learning_rate = args.learning_rate
        self.learning_rate = 1e-3
        # self.sm_conv1d_filter = model_params.sm_conv1d_filter
        # self.sm_conv1d_width = model_params.sm_conv1d_width
        # self.sm_conv_filters = parse_list_str(model_params.sm_conv_filters)
        # self.sm_conv_heights = parse_list_str(model_params.sm_conv_heights)
        # self.sm_conv_widths = parse_list_str(model_params.sm_conv_widths)
        # self.sm_maxpool_widths = parse_list_str(model_params.sm_maxpool_width)
        # self.sm_fc_num_units = parse_list_str(model_params.sm_fc_num_units)
        # self.sm_margin = model_params.sm_margin
        self.sm_conv1d_filter = 128
        self.sm_conv1d_width = 3
        self.sm_conv_filters = parse_list_str(256)
        self.sm_conv_heights = parse_list_str(3)
        self.sm_conv_widths = parse_list_str(3)
        self.sm_maxpool_widths = parse_list_str(2)
        self.sm_fc_num_units = parse_list_str(256)
        self.sm_margin = 1.0

        self._build_model()
        self._add_train_op()

    def _build_inference_graph(self, device=None):

        self._sents_A = tf.placeholder(tf.int32, [None, max_length])
        self._sents_B = tf.placeholder(tf.int32, [None, max_length])
        self._lengths_A = tf.placeholder(tf.int32, [None])
        self._lengths_B = tf.placeholder(tf.int32, [None])

        with tf.variable_scope("seq_match"), tf.device(device):
            with tf.name_scope("embedding"):
                if self.trained_wordvec == True:
                    init_embeddings = tf.constant(self.word_vector, dtype=tf.float32)
                    embeddings_matrix = tf.get_variable("embeddings", 
                                               initializer=init_embeddings, trainable=True)
                else:
                    embeddings_matrix = tf.get_variable("embeddings", 
                                          [self.vocab_szie, self.emb_size], trainable=True)

                print("embed matrix: ", embeddings_matrix)
                sent_A_embed = tf.nn.embedding_lookup(embeddings_matrix, self._sents_A)
                sent_B_embed = tf.nn.embedding_lookup(embeddings_matrix, self._sents_B)

            self._output = self._add_conv_match(sent_A_embed, sent_B_embed,
                                                self._lengths_A, self._lengths_B)

    def _build_model(self):
        self._sents_A_pos = tf.placeholder(tf.int32, [None, max_length])
        self._lengths_A_pos = tf.placeholder(tf.int32, [None])

        self._sents_B_pos = tf.placeholder(tf.int32, [None, max_length])
        self._lengths_B_pos = tf.placeholder(tf.int32, [None])

        self._sents_A_neg = tf.placeholder(tf.int32, [None, max_length])
        self._lengths_A_neg = tf.placeholder(tf.int32, [None])

        self._sents_B_neg = tf.placeholder(tf.int32, [None, max_length])
        self._lengths_B_neg = tf.placeholder(tf.int32, [None])

        self.dropout_rate_prob = tf.placeholder(tf.float32, name="dropout_rate_prob")
        self.is_train = tf.placeholder(tf.bool, name="is_train")

        with tf.variable_scope("seq_match"), tf.device(self._device_0):
            with tf.variable_scope('embeddings'):
                if self.trained_wordvec == True:
                    init_embeddings = tf.constant(self.word_vector, dtype=tf.float32)
                    embeddings_matrix = tf.get_variable("embeddings", initializer=init_embeddings, 
                                                        trainable=True)
                else:
                    embeddings_matrix = tf.get_variable("embeddings", [self.vocab_szie, self.emb_size],
                                                        trainable=True)

                print("embed matrix: ", embeddings_matrix)
                sent_A_pos_embed = tf.nn.embedding_lookup(embeddings_matrix, self._sents_A_pos)
                sent_A_neg_embed = tf.nn.embedding_lookup(embeddings_matrix, self._sents_A_neg)
                sent_B_pos_embed = tf.nn.embedding_lookup(embeddings_matrix, self._sents_B_pos)
                sent_B_neg_embed = tf.nn.embedding_lookup(embeddings_matrix, self._sents_B_neg)

            self._output_pos = self._add_conv_match(sent_A_pos_embed, sent_B_pos_embed,
                                                    self._lengths_A_pos,
                                                    self._lengths_B_pos)
            self._output_neg = self._add_conv_match(sent_A_neg_embed, sent_B_neg_embed,
                                                    self._lengths_A_neg,
                                                    self._lengths_B_neg, True)

        with tf.variable_scope("loss"), tf.device(self._device_0):
            # Implements ranking triplet loss
            batch_loss = tf.nn.relu(self.sm_margin + self._output_neg -
                                    self._output_pos)
            loss = tf.reduce_mean(batch_loss)
            # Accuracy: correct if pos > neg
            accuracy = tf.reduce_mean(
                tf.to_float(self._output_pos > self._output_neg))

            tf.summary.scalar("loss", loss)
            tf.summary.scalar("accuracy", accuracy)
            self.global_step = tf.Variable(0, name="global_step", trainable=False)
            self._loss = loss
            self._accuracy = accuracy

            # optimizer = tf.train.AdamOptimizer(self.learning_rate)
            # grads_and_vars = optimizer.compute_gradients(self._loss)
            # self.grads_and_vars = grads_and_vars
            # self.train_op = optimizer.apply_gradients(grads_and_vars, global_step=self.global_step)

            # self.max_grad_norm = 1.0
            # optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
            # tvars = tf.trainable_variables()
            # grads, global_norm = tf.clip_by_global_norm(
            #     tf.gradients(self._loss, tvars), self.max_grad_norm)
            # self.train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step)

    def _add_train_op(self, optimizer_class=tf.train.GradientDescentOptimizer):
        self.min_lr = 0.001
        self.decay_rate = 0.1
        self.decay_step = 30000
        self.max_grad_norm = 1.0
        self._lr_rate = tf.maximum(
            self.min_lr,  # minimum learning rate.
            tf.train.exponential_decay(self.learning_rate, self.global_step, self.decay_step,
                                       self.decay_rate))
        tf.summary.scalar("learning_rate", self._lr_rate)

        tvars = tf.trainable_variables()
        with tf.device(self._device_0):
            # Compute gradients
            grads, global_norm = tf.clip_by_global_norm(
                tf.gradients(self._loss, tvars), self.max_grad_norm)
            tf.summary.scalar("global_norm", global_norm)

            # Create optimizer and train ops
            optimizer = optimizer_class(self._lr_rate)
            #optimizer = tf.train.AdamOptimizer(learning_rate=0.0005)
            self.train_op = optimizer.apply_gradients(
                zip(grads, tvars), global_step=self.global_step, name="train_step")

    def _add_conv_match(self,
                        sent_A_embed,
                        sent_B_embed,
                        lengths_A,
                        lengths_B,
                        reuse=None):
        """ This method implements the following work:
      Hu, B., Lu, Z., Li, H., & Chen, Q. (2014). Convolutional neural network
      architectures for matching natural language sentences. In Advances in neural
      information processing systems (pp. 2042-2050).
    """
        with tf.variable_scope('conv_match', reuse=reuse):
            # Part 1: conv1d
            with tf.variable_scope('conv1d'):
                # First sentence with conv-1D in layer 1
                sent_A_mask = tf.expand_dims(
                    tf.sequence_mask(lengths_A, max_length, tf.float32),
                    2)  # [?, max_sent_len, 1]
                sent_A_embed *= sent_A_mask

                sent_A_conv1d_out = tf.layers.conv1d(
                    sent_A_embed,
                    self.sm_conv1d_filter,
                    self.sm_conv1d_width,
                    padding="same",
                    activation=tf.nn.relu,
                    kernel_initializer=tf.random_uniform_initializer(-0.1, 0.1),
                    name="sent_A_conv1d_layer1")
                sent_A_conv1d_out = tf.layers.dropout(
                    sent_A_conv1d_out, self.dropout_rate_prob, training=self.is_train)

                # Second sentence with conv-1D in layer 1
                sent_B_mask = tf.expand_dims(
                    tf.sequence_mask(lengths_B, max_length, tf.float32),2)  # [?, max_sent_len, 1]
                sent_B_embed *= sent_B_mask

                sent_B_conv1d_out = tf.layers.conv1d(
                    sent_B_embed,
                    self.sm_conv1d_filter,
                    self.sm_conv1d_width,
                    padding="same",
                    activation=tf.nn.relu,
                    kernel_initializer=tf.random_uniform_initializer(-0.1, 0.1),
                    name="sent_B_conv1d_layer1")
                sent_B_conv1d_out = tf.layers.dropout(
                    sent_B_conv1d_out, self.dropout_rate_prob, training=self.is_train)

                # Extend and concat the feature maps into 2D
                sent_B_conv1d_out_list = tf.unstack(sent_B_conv1d_out, axis=1)
                sent_AB_concat_feats = []
                for x in sent_B_conv1d_out_list:  # [?, sm_conv1d_filter]
                    tiled_x = tf.tile(tf.expand_dims(x, 1), [1, max_length, 1])
                    concat_feats = tf.concat(
                        [sent_A_conv1d_out, tiled_x],
                        axis=2)  # [?, max_sent_len, sm_conv1d_filter*2]
                    sent_AB_concat_feats.append(concat_feats)
                sent_AB_2D_feats = tf.stack(
                    sent_AB_concat_feats,
                    axis=2)  # [?, max_sent_len, max_sent_len, sm_conv1d_filter*2]

            # Part 2: conv2d
            with tf.variable_scope('conv2d'):
                conv2d_feats = sent_AB_2D_feats
                for i, (cf, ch, cw, mw) in enumerate(
                        zip(self.sm_conv_filters, self.sm_conv_heights, self.sm_conv_widths,
                            self.sm_maxpool_widths)):
                    conv2d_feats = tf.layers.conv2d(
                        conv2d_feats,
                        cf, (ch, cw),
                        padding="valid",
                        activation=tf.nn.relu,
                        kernel_initializer=tf.random_uniform_initializer(-0.1, 0.1),
                        name="conv2d_layer_%d" % (i + 1))

                    conv2d_feats = tf.layers.dropout(
                        conv2d_feats, self.dropout_rate_prob, training=self.is_train)

                    if mw:
                        conv2d_feats = tf.layers.max_pooling2d(
                            conv2d_feats, mw, mw, name="maxpool_layer_%d" % (i + 1))

            # Part 3: fully-connected
            with tf.variable_scope('fc'):
                mlp_hidden = tf.contrib.layers.flatten(conv2d_feats)

                for i, n in enumerate(self.sm_fc_num_units):
                    mlp_hidden = tf.contrib.layers.fully_connected(
                        mlp_hidden,
                        n,
                        activation_fn=tf.nn.relu,  # tf.tanh/tf.sigmoid/tf.nn.relu
                        weights_initializer=tf.random_uniform_initializer(-0.1, 0.1),
                        scope="fc_layer_%d" % (i + 1))

                prob = tf.squeeze(
                    tf.contrib.layers.fully_connected(
                        mlp_hidden,
                        1,
                        activation_fn=tf.tanh,  # tf.sigmoid
                        weights_initializer=tf.random_uniform_initializer(-0.1, 0.1),
                        scope="fc_layer_output"),
                    axis=1)  # [?]
                # NB: use tanh instead of sigmoid because it centers at 0

        return prob

    def compute_coherence(self, sess, sents_A, sents_B, lengths_A, lengths_B):
        result = sess.run(
            self._output,
            feed_dict={
                self._sents_A: sents_A,
                self._sents_B: sents_B,
                self._lengths_A: lengths_A,
                self._lengths_B: lengths_B
            })

        return result

