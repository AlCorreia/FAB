import inspect
import math
import numpy as np
import os
import pandas as pd
import random
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
from tensorflow.python.ops.rnn_cell_impl import _linear as get_logits
from tqdm import tqdm

from read_data import get_batch_idxs

VERY_LOW_NUMBER=-1e30

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
        # Define the directory where the results will be saved
        # TODO: Define a better convention for directories names
        self.directory = config['directories']['out_dir']
        # Define global step and learning rate decay
        self.global_step = tf.Variable(0, trainable=False)
        # Learning rate with exponential decay
        # decayed_learning_rate = learning_rate * decay_rate ^ (global_step / decay_steps)
        initial_learning_rate = config['model']['learning_rate']
        self.learning_rate = tf.train.exponential_decay(learning_rate=initial_learning_rate,
                                                        global_step=self.global_step,
                                                        decay_steps=config['model']['decay_steps'],
                                                        decay_rate=config['model']['decay_rate'],
                                                        staircase=True)

        # TODO: Think of a better way to define these dimensions
        # 1. Capitals letters: indicates type of element.
        #    B: batch, P: paragraph, Q: question, W: word, V: vocabulary, C:char,
        #    E: embedding, H: hidden
        # 2. Lowercase letters: indicates a measure
        #    m: maximum, n: number, s: size, o: out
        self.Bs = config['model']['batch_size']
        #self.Ps = config['model']['max_par_size']
        #self.Qs = config['model']['max_ques_size']
        #self.Wm = config['model']['max_word_size']
        self.WVs = config['model']['vocabulary_size']
        self.WEs = int(config['glove']['vec_size']) # Assumed to be equal to the GloVe vector size
        # self.CVs = config['model']['char_vocabulary_size']
        # self.CEs = config['model']['char_emb_size']
        # self.Co = config['model']['char_out_size']
        self.Hn = config['model']['n_hidden'] # Number of hidden units in the LSTM cell

        # Define placeholders
        # TODO: Include characters
        self.is_training = tf.placeholder(tf.bool, name='is_training')
        self.x = tf.placeholder('int32', [self.Bs, None], name='x') #number of batches and number of words
        # self.cx = tf.placeholder('int32', [self.Bs, None, None, W], name='cx')
        self.x_mask = tf.placeholder('bool', [self.Bs, None], name='x_mask')
        self.q = tf.placeholder('int32', [self.Bs, None], name='q')
        # self.cq = tf.placeholder('int32', [self.Bs, None, W], name='cq')
        self.y = tf.placeholder('bool', [self.Bs, None], name='y')
        self.y2 = tf.placeholder('bool', [self.Bs, None], name='y2')
        self.is_train = tf.placeholder('bool', [], name='is_train')
        self.new_emb_mat = tf.placeholder('float', [None, self.WEs], name='new_emb_mat')

        #Masks
        self.x_mask = tf.sign(self.x)
        self.q_mask = tf.sign(self.q)

        # Redefine some parameters based on the actual tensor dimensions
        self.Ps = tf.shape(self.x)[1]
        self.Qs = tf.shape(self.q)[1]

        # Define a dictionary to hold references to the model's tensors
        self.tensor_dict = {}

        # Forward outputs / loss inputs
        self.logits = None
        self.yp = None
        self.var_list = None

        # Loss outputs
        self.loss = None

        self._build_forward()
        self._build_loss()

        # Define optimizer and train step
        # TODO: We could add the optimizer option to the config file. ADAM for now.
        self.optimizer = tf.train.AdamOptimizer()
        self.train_step = self.optimizer.minimize(self.loss, global_step=self.global_step)
        # TODO: Understand the need for the moving average function
        # self.var_ema = None
        # if rep:
        #     self._build_var_ema()
        # if config.mode == 'train':
        #     self._build_ema()

        self.summary = tf.merge_all_summaries()
        self.summary = tf.merge_summary(tf.get_collection("summaries", scope=self.scope))
        # Add a writer object to log the models's progress in the "train" folder
        self.writer = tf.summary.FileWriter(self.directory + '/train',
                                            self.sess.graph)

        # Define a session for the model
        self.sess = tf.Session()

    def train(self, batch_idxs, dataset):
        """ Runs a train step given the input X and the correct label y.

            Parameters
            ----------
            batch_idxs : list of the idxs of each example
            dataset : the correspondent json file

        """
        # Combine the input dictionaries for all the features models
        feed_dict = self.get_feed_dict(batch_idxs, is_training=True)
        # Run the training step
        summary, _ = self.sess.run([self.summary, self.train_step],
                                   feed_dict=feed_dict)
        # Write the results to Tensorboard
        self.writer.add_summary(summary, global_step=self.sess.run(self.global_step))
        # Regularly save the models parameters
        if self.sess.run(self.global_step) % 1000 == 0:
            self.saver.save(self.sess, self.directory + '/model.ckpt')

    def _build_forward(self):
        """
            Builds the model's feedforward network.

        """
        config = self.config
        with tf.variable_scope("word_emb"), tf.device("/cpu:0"):
            # TODO: I am not sure that having a config variable for this is the best solution
            # TODO: Save the embedding matrix somewhere other than the config file
            if config['model']['is_training']:
                word_emb_mat = tf.get_variable("word_emb_mat",
                                               dtype='float',
                                               shape=[self.WVs, self.WEs],
                                               initializer=get_initializer(config.emb_mat))
            else:
                word_emb_mat = tf.get_variable("word_emb_mat",
                                               shape=[self.WVs, self.WEs],
                                               dtype='float')
            if config.use_glove_for_unk:
                word_emb_mat = tf.concat(0, [word_emb_mat, self.new_emb_mat])

            with tf.name_scope("word"):
                Ax = tf.nn.embedding_lookup(word_emb_mat, self.x)  # [Bs, Ps, Hn]
                Aq = tf.nn.embedding_lookup(word_emb_mat, self.q)  # [Bs, Qs, Hn]
                self.tensor_dict['x'] = Ax
                self.tensor_dict['q'] = Aq

        # Build the LSTM cell with dropout
        cell = BasicLSTMCell(self.n_hidden, state_is_tuple=True)
        d_cell = SwitchableDropoutWrapper(cell, self.is_training, input_keep_prob=config.input_keep_prob)
        # Calculate the number of used values in the each matrix
        x_len = tf.reduce_sum(tf.cast(self.x_mask, 'int32'), 1)  # [Bs]
        q_len = tf.reduce_sum(tf.cast(self.q_mask, 'int32'), 1)  # [Bs]

        with tf.variable_scope("prepro"):
            # [Bs, Qs, 2Hn], [Bs, Hn]
            (fw_u, bw_u), ((_, fw_u_f), (_, bw_u_f)) = \
                tf.nn.bidirectional_dynamic_rnn(cell_fw=d_cell,
                                                cell_bw=d_cell,
                                                inputs=Aq,
                                                sequence_length=q_len,
                                                dtype='float',
                                                scope='u1')
            u = tf.concat(2, [fw_u, bw_u])
            if config['model']['share_lstm_weights']:
                tf.get_variable_scope().reuse_variables()
                # [Bs, Ps, 2Hn]
                (fw_h, bw_h), _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell,
                                                                  cell_bw=cell,
                                                                  inputs=Ax,
                                                                  sequence_length=x_len,
                                                                  dtype='float',
                                                                  scope='u1')
                h = tf.concat(3, [fw_h, bw_h])  # [Bs, Sn, Ss, 2Hn]
            else:
                # [Bs, Ps, 2Hn]
                (fw_h, bw_h), _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell,
                                                                  cell_bw=cell,
                                                                  inputs=Ax,
                                                                  sequence_length=x_len,
                                                                  dtype='float',
                                                                  scope='h1')
                h = tf.concat(3, [fw_h, bw_h])  # [Bs, Ps, 2Hn]
            self.tensor_dict['u'] = u
            self.tensor_dict['h'] = h

        with tf.variable_scope("main"):

			#AttentionLayer

            # TODO: Implement attention model
            p0 =  attention_layer(x,q, h,u,scope='p0') #[Bs, Ps, 8Hn]
            first_cell = tf.reshape(tf.tile(tf.expand_dims(u, 1),
                                            [1, Sn, 1, 1]),
                                    [self.Bs * self.Sn, self.Qs, 2 * self.Hn])

            # [Bs, Sn, Ss, 2Hn]
            (fw_g0, bw_g0), _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=first_cell,
                                                                cell_bw=first_cell,
                                                                inputs=p0,
                                                                sequence_length=x_len,
                                                                dtype='float',
                                                                scope='g0')
            g0 = tf.concat(3, [fw_g0, bw_g0])

            # [Bs, Sn, Ss, 2Hn]
            (fw_g1, bw_g1), _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=first_cell,
                                                                cell_bw=first_cell,
                                                                inputs=g0,
                                                                sequence_length=x_len,
                                                                dtype='float',
                                                                scope='g1')
            g1 = tf.concat(3, [fw_g1, bw_g1])

            # TODO: Rewrite the get_logits function
            logits = get_logits(args=[g1, p0],
                                output_size=1,
                                bias=True)
            # TODO: Rewrite the softsel function
            a1i = softsel(tf.reshape(g1, [self.Bs, self.Sn * self.Ss, 2 * self.Hn]),
                          tf.reshape(logits, [self.Bs, self.Sn * self.Ss]))
            a1i = tf.tile(tf.expand_dims(tf.expand_dims(a1i, 1), 1),
                          [1, self.Sn, self.Ss, 1])

            # [Bs, Sn, Ss, 2Hn]
            (fw_g2, bw_g2), _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=d_cell,
                                                                cell_bw=d_cell,
                                                                inputs=tf.concat(3, [p0, g1, a1i, g1 * a1i]),
                                                                sequence_length=x_len,
                                                                dtype='float',
                                                                scope='g2')
            g2 = tf.concat(3, [fw_g2, bw_g2])
            # TODO: Rewrite the get_logits function
            logits2 = get_logits(args=[g2, p0],
                                 output_size=1,
                                 bias=True)

            flat_logits = tf.reshape(logits, [-1, M * JX])
            flat_yp = tf.nn.softmax(flat_logits)  # [-1, Sn * Ss]
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

    def _build_loss(self):
        """
            Defines the model's loss function.

        """
        loss_mask = tf.reduce_max(tf.cast(self.q_mask, 'float'), 1)
        losses = tf.nn.softmax_cross_entropy_with_logits(
            self.logits, tf.cast(tf.reshape(self.y, [-1, self.Sn * self.Ss]), 'float'))
        ce_loss = tf.reduce_mean(loss_mask * losses)
        tf.add_to_collection('losses', ce_loss)
        ce_loss2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            self.logits2, tf.cast(tf.reshape(self.y2, [-1, self.Sn * self.Ss]), 'float')))
        tf.add_to_collection("losses", ce_loss2)

        self.loss = tf.add_n(tf.get_collection('losses', scope=self.scope), name='loss')
        tf.scalar_summary(self.loss.op.name, self.loss)
        tf.add_to_collection('ema/scalar', self.loss)

    # TODO: Better define this function.
    # Don't know whether it's better to implement it on the model or on in the
    # Dataset class
    #
    # Options:
    # 1. Implement it on model have the get_batch_idxs on Dataset pass the full
    # dictionary.
    # 2. Implement it on the Dataset class and return the whole feed_dict. In
    # that case, variable names must be well defined.
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    def get_feed_dict(self, batch_idxs, is_training):
        feed_dict = {}

        # TODO: Add characters
        # TODO: Actually read the data from the json files
        x = np.zeros([self.Bs, self.Sn, self.Ss], dtype='int32')
        # cx = np.zeros([self.Bs, self.Sn, self.Ss, self.Ws], dtype='int32')
        x_mask = np.zeros([self.Bs, self.Sn, self.Ss], dtype='bool')
        q = np.zeros([self.Bs, self.Qs], dtype='int32')
        # cq = np.zeros([self.Bs, self.Qs, self.Ws], dtype='int32')
        q_mask = np.zeros([self.Bs, self.Qs], dtype='bool')

        feed_dict[self.x] = x
        feed_dict[self.x_mask] = x_mask
        # feed_dict[self.cx] = cx
        feed_dict[self.q] = q
        # feed_dict[self.cq] = cq
        feed_dict[self.q_mask] = q_mask
        feed_dict[self.is_training] = is_training
        if config.use_glove_for_unk:
            feed_dict[self.new_emb_mat] = batch.shared['new_emb_mat']

        return feed_dict


def attention_layer(x, q, h, u, scope=None):
    """
        Define attention mechanism between vectors h (question) and u (paragraph):
             att(m,n)  = vec1 . h_m + vec2 . u_n + (vec3 * h_m) . u_n
        where '.' is the dot product and '*' the elementwise product
    """
    mask_matrix=tf.sign(tf.matmul(tf.expand_dims(x,-1),tf.expand_dims(x_q,-1),transpose_b=True))
    with tf.variable_scope(scope):
        vec1_att = tf.get_variable("att1",dtype='float',shape=[hidden_size,1]) #[Hs, 1]
        vec2_att = tf.get_variable("att2",dtype='float',shape=[hidden_size,1]) #[Hs, 1]
        vec3_att = tf.get_variable("att3",dtype='float',shape=[hidden_size]) #[Hs, 1]

    #shaping all the vectors into a matrix to compute attention values in one matrix multiplication
    shaped_h=tf.concat(
        tf.unstack(
            value=h,
            axis=0),
        axis=0) #[Hs, Ps * Bs]

    shaped_u = tf.concat(
        tf.unstack(
            value=u,
            axis=0),
        axis=0) #[Hs, Qs * Bs]

    #computation of vec1 * h + vec2 * u
    att_1_Product = tf.reshape(
        tf.matmul(
            shaped_h,
            vec1_att),
        shape=[batch_size,1,-1]) # [Bs, 1, Ps]

    att_2_Product = tf.reshape(
        tf.matmul(
            shaped_u,
            vec2_att),
        shape=[batch_size,-1,1]) # [Bs, Qs, 1]

    att_1_Product = tf.tile(
        att_1_Product,
        [1,tf.reduce_max(length(x_q_input)),1]) #[Bs, Qs, Ps]

    att_2_Product = tf.tile(
        att_2_Product,
        [1,1,tf.reduce_max(length(x_input))]) #[Bs, Qs, Ps]

    #  of (vec3 * h)  . u

    h_vectorized = tf.multiply(h,att_3_vec) #vect 3 * h
    att_3_Product = tf.matmul(u,h_vectorized,transpose_b=True) # ((vec 3 * h) . u [Bs, Qs, Ps]
    att_final=tf.transpose(
        att_1_Product+att_2_Product+att_3_Product,
        perm=[0,2,1]) #[Bs, Ps, Qs]

    att_final_masked = att_final + tf.multiply(tf.cast(1 - mask_matrix, tf.float32), -3e15) #masking

    # paragraph to question attention
    p2q = tf.multiply(
        tf.nn.softmax(logits=att_final_masked,dim=-1),
        tf.cast(mask_matrix,'float')
    ) # computing logits, taking into account mask

    U_a=tf.matmul(p2q,u)

    q2p = tf.nn.softmax(
        tf.reduce_max(att_final_masked,axis=-1))

    H_a = tf.tile(
	    tf.matmul(
            tf.expand_dims(q2p,1),
            h),
	    [1,tf.reduce_max(length(x_input)),1]
    )

    G = tf.concat([h,U_a,tf.multiply(h,U_a),tf.multiply(h,H_a)],axis=-1)
    return G

def softsel(target, logits, mask=None, scope=None):
    """
    :param target: [ ..., J, d] dtype=float
    :param logits: [ ..., J], dtype=float
    :param mask: [ ..., J], dtype=bool
    :param scope:
    :return: [..., d], dtype=float
    """
    with tf.name_scope(scope or "Softsel"):
        a = softmax(logits, mask=mask)
        target_rank = len(target.get_shape().as_list())
        out = tf.reduce_sum(tf.expand_dims(a, -1) * target, target_rank - 2)
        return out
