import inspect
import math
import numpy as np
import os
import pandas as pd
import random
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
from tqdm import tqdm


class Model(object):
    def __init__(self, config):
        """
            Creates a RNN model.

            Attributes
            ----------
            config: json
                Configuration file specifying the model's parameters

        """
        self.config = config

        # Define the number of hidden states in the LSTM cell
        self.n_hidden = config['model']['b_hidden']
        # Define the directory where the results will be saved
        # TODO: Define a better convention for directories names
        self.directory = config['model']['out_dir']
        # Define global step and learning rate decay
        self.global_step = tf.Variable(0, trainable=False)
        # Learning rate with exponential decay
        # decayed_learning_rate = learning_rate * decay_rate ^ (global_step / decay_steps)
        # TODO: Add decay parameters to config
        initial_learning_rate = config['model']['learning_rate']
        self.learning_rate = tf.train.exponential_decay(learning_rate=initial_learning_rate,
                                                        global_step=self.global_step,
                                                        decay_steps=1000,
                                                        decay_rate=0.96,
                                                        staircase=True)
        # Define the model's hyperparameters
        self.batch_size = config['model']['batch_size']
        # Number of words in the vocabulary
        self.vocabulary_size = config['model']['vocabulary_size']
        # Word embedding size, assumed to be equal to the GloVe vector size
        self.word_emb_size = config['glove']['vec_size']

        # Define placeholders
        self.is_training = tf.placeholder(tf.bool, name='is_training')
        self.x = tf.placeholder('int32', [self.batch_size, None, None], name='x')
        # self.cx = tf.placeholder('int32', [self.batch_size, None, None, W], name='cx')
        self.x_mask = tf.placeholder('bool', [self.batch_size, None, None], name='x_mask')
        self.q = tf.placeholder('int32', [self.batch_size, None], name='q')
        # self.cq = tf.placeholder('int32', [self.batch_size, None, W], name='cq')
        self.q_mask = tf.placeholder('bool', [self.batch_size, None], name='q_mask')
        self.y = tf.placeholder('bool', [self.batch_size, None, None], name='y')
        self.y2 = tf.placeholder('bool', [self.batch_size, None, None], name='y2')
        self.is_train = tf.placeholder('bool', [], name='is_train')
        self.new_emb_mat = tf.placeholder('float', [None, self.word_emb_size], name='new_emb_mat')

        # Look up embeddings for inputs.
        embeddings = tf.Variable(
            tf.random_uniform([self.vocabulary_size, self.embedding_size], -1.0, 1.0), name='product_id')
        embed = tf.nn.embedding_lookup(embeddings, self.sequence)

        cell = tf.contrib.rnn.LSTMCell(self.n_hidden, state_is_tuple=True)
        output, _ = tf.nn.dynamic_rnn(cell, embed, dtype=tf.float32)

        # Get only the last output in the sequence
        self.output = output[:, -1, :]
        # Add a dense layer on top of the RNN to convert the output into a softmax
        softmax_w = tf.get_variable("softmax_w",
                                    [n_hidden, self.vocabulary_size],
                                    dtype=tf.float32)
        softmax_b = tf.get_variable("softmax_b",
                                    [self.vocabulary_size],
                                    dtype=tf.float32)
        logits = tf.matmul(self.output, softmax_w) + softmax_b
        predictions = tf.nn.softmax(logits)

        # Define the cross entropy loss
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                                        logits=logits,
                                        labels=one_hot_out))

        # Estimate the accuracy of the model
        correct_prediction = tf.equal(self.target,
                                      tf.argmax(predictions, axis=1))
        accuracy = tf.reduce_mean(
            tf.cast(correct_prediction, tf.float32))
        # Add Tensorboard summaries
        accuracy_summary = tf.summary.scalar('accuracy', accuracy)
        loss_summary = tf.summary.scalar('loss', tf.reduce_mean(self.loss))

        # Define the optimizer
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self._train_op = optimizer.minimize(self.loss, global_step=self.global_step)

        self.sess = tf.Session()
        init = tf.global_variables_initializer()
        # Launch the tensorflow graph
        self.sess.run(init)

        self.merged = tf.summary.merge_all()
        # Create summary writers for train and test set
        self.writer = tf.summary.FileWriter(self.directory + '/train',
                                            self.sess.graph)

        config = projector.ProjectorConfig()
        embedding = config.embeddings.add()
        embedding.tensor_name = 'product_id'
        # Link this tensor to its metadata file (e.g. labels).
        embedding.metadata_path = os.path.join('Product.tsv')
        projector.visualize_embeddings(self.writer, config)

    def _build_forward(self):
        """
            Builds the model's feedforward network.

        """
        config = self.config

        # TODO: Think of a better way to define these dimensions
        N = self.batch_size
        M = config['model']['max_num_sents']
        JX = config['model']['max_sent_size']
        JQ = config['model']['max_ques_size']
        VW = self.vocabulary_size
        VC = config['model']['char_vocabulary_size']
        d = config['model']['n_hidden']
        W = config['model']['max_word_size']

        JX = tf.shape(self.x)[2]
        JQ = tf.shape(self.q)[1]
        M = tf.shape(self.x)[1]
        dc, dw, dco = config.char_emb_size, config.word_emb_size, config.char_out_size

        with tf.variable_scope("word_emb"), tf.device("/cpu:0"):
            # TODO: I am not sure that having a config variable for this is the best solution
            # TODO: Save the embedding matrix somewhere other than the config file
            if config['model']['is_training']:
                word_emb_mat = tf.get_variable("word_emb_mat",
                                               dtype='float',
                                               shape=[VW, dw],
                                               initializer=get_initializer(config.emb_mat))
            else:
                word_emb_mat = tf.get_variable("word_emb_mat",
                                               shape=[VW, dw],
                                               dtype='float')
            if config.use_glove_for_unk:
                word_emb_mat = tf.concat(0, [word_emb_mat, self.new_emb_mat])

            with tf.name_scope("word"):
                Ax = tf.nn.embedding_lookup(word_emb_mat, self.x)  # [N, M, JX, d]
                Aq = tf.nn.embedding_lookup(word_emb_mat, self.q)  # [N, JQ, d]
                self.tensor_dict['x'] = Ax
                self.tensor_dict['q'] = Aq

        # Build the LSTM cell with dropout
        cell = BasicLSTMCell(self.n_hidden, state_is_tuple=True)
        d_cell = SwitchableDropoutWrapper(cell, self.is_training, input_keep_prob=config.input_keep_prob)
        # Calculate the number of used values in the each matrix
        x_len = tf.reduce_sum(tf.cast(self.x_mask, 'int32'), 2)  # [N, M]
        q_len = tf.reduce_sum(tf.cast(self.q_mask, 'int32'), 1)  # [N]

        with tf.variable_scope("prepro"):
            (fw_u, bw_u), ((_, fw_u_f), (_, bw_u_f)) = tf.nn.bidirectional_dynamic_rnn(d_cell, d_cell, Aq, q_len, dtype='float', scope='u1')  # [N, J, d], [N, d]
            u = tf.concat(2, [fw_u, bw_u])
            if config['model']['share_lstm_weights']:
                tf.get_variable_scope().reuse_variables()
                (fw_h, bw_h), _ = tf.nn.bidirectional_dynamic_rnn(cell, cell, Ax, x_len, dtype='float', scope='u1')  # [N, M, JX, 2d]
                h = tf.concat(3, [fw_h, bw_h])  # [N, M, JX, 2d]
            else:
                (fw_h, bw_h), _ = bidirectional_dynamic_rnn(cell, cell, Ax, x_len, dtype='float', scope='h1')  # [N, M, JX, 2d]
                h = tf.concat(3, [fw_h, bw_h])  # [N, M, JX, 2d]
            self.tensor_dict['u'] = u
            self.tensor_dict['h'] = h

        with tf.variable_scope("main"):
            if config.dynamic_att:
                p0 = h
                u = tf.reshape(tf.tile(tf.expand_dims(u, 1), [1, M, 1, 1]), [N * M, JQ, 2 * d])
                q_mask = tf.reshape(tf.tile(tf.expand_dims(self.q_mask, 1), [1, M, 1]), [N * M, JQ])
                first_cell = AttentionCell(cell, u, mask=q_mask, mapper='sim',
                                           input_keep_prob=self.config.input_keep_prob, is_train=self.is_train)
            else:
                p0 = attention_layer(config, self.is_train, h, u, h_mask=self.x_mask, u_mask=self.q_mask, scope="p0", tensor_dict=self.tensor_dict)
                first_cell = d_cell

            (fw_g0, bw_g0), _ = bidirectional_dynamic_rnn(first_cell, first_cell, p0, x_len, dtype='float', scope='g0')  # [N, M, JX, 2d]
            g0 = tf.concat(3, [fw_g0, bw_g0])
            (fw_g1, bw_g1), _ = bidirectional_dynamic_rnn(first_cell, first_cell, g0, x_len, dtype='float', scope='g1')  # [N, M, JX, 2d]
            g1 = tf.concat(3, [fw_g1, bw_g1])

            logits = get_logits([g1, p0], d, True, wd=config.wd, input_keep_prob=config.input_keep_prob,
                                mask=self.x_mask, is_train=self.is_train, func=config.answer_func, scope='logits1')
            a1i = softsel(tf.reshape(g1, [N, M * JX, 2 * d]), tf.reshape(logits, [N, M * JX]))
            a1i = tf.tile(tf.expand_dims(tf.expand_dims(a1i, 1), 1), [1, M, JX, 1])

            (fw_g2, bw_g2), _ = bidirectional_dynamic_rnn(d_cell, d_cell, tf.concat(3, [p0, g1, a1i, g1 * a1i]),
                                                          x_len, dtype='float', scope='g2')  # [N, M, JX, 2d]
            g2 = tf.concat(3, [fw_g2, bw_g2])
            logits2 = get_logits([g2, p0], d, True, wd=config.wd, input_keep_prob=config.input_keep_prob,
                                 mask=self.x_mask,
                                 is_train=self.is_train, func=config.answer_func, scope='logits2')

            flat_logits = tf.reshape(logits, [-1, M * JX])
            flat_yp = tf.nn.softmax(flat_logits)  # [-1, M*JX]
            yp = tf.reshape(flat_yp, [-1, M, JX])
            flat_logits2 = tf.reshape(logits2, [-1, M * JX])
            flat_yp2 = tf.nn.softmax(flat_logits2)
            yp2 = tf.reshape(flat_yp2, [-1, M, JX])

            self.tensor_dict['g1'] = g1
            self.tensor_dict['g2'] = g2

            self.logits = flat_logits
            self.logits2 = flat_logits2
            self.yp = yp
            self.yp2 = yp2
