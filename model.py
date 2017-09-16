import math
import numpy as np
import os
import pandas as pd
import random
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
from tqdm import tqdm
from utils import plot, send_mail, EM_and_F1
import pdb

from my_tf import optimize_loss
from read_data import get_batch_idxs

VERY_LOW_NUMBER = -1e30


class Model(object):
    def __init__(self, config):
        """
            Creates a base model and defines the hyperparameters.

            Attributes
            ----------
            config: json
                Configuration file specifying the model's parameters

        """
        self.config = config
        # Define the directory where the results will be saved
        # TODO: Define a better convention for directories names
        self.directory = config['directories']['dir']
        self.dir_plots = config['directories']['plots']
        # Define global step and learning rate decay
        self.global_step = tf.Variable(0, trainable=False)
        # Define a placeholder for the dropout
        self.keep_prob = tf.placeholder_with_default(1.0, shape=(), name='dropout')
        # Learning rate with exponential decay
        # decayed_learning_rate = learning_rate * decay_rate ^ (global_step / decay_steps)

        # TODO: Think of a better way to define these dimensions
        # 1. Capitals letters: indicates type of element.
        #    B: batch, P: paragraph, Q: question, W: word, V: vocabulary, C:char,
        #    E: embedding, H: hidden
        # 2. Lowercase letters: indicates a measure
        #    m: maximum, n: number, s: size, o: out
        self.Bs = config['train']['batch_size']
        self.MHs = config['model']['multi_head_size']  # Multi-Head
        self.FFHs = config['model']['FeedForward_Hidden_Size']
        # self.Ps = config['model']['max_par_size']
        # self.Qs = config['model']['max_ques_size']
        # self.Wm = config['model']['max_word_size']
        self.WVs = config['model']['vocabulary_size']
        self.WEs = int(config['glove']['vec_size'])  # Assumed to be equal to the GloVe vector size
        self.WEAs = config['model']['attention_emb_size']  # Word-embedding attention size
        self.WEPs = config['model']['process_emb_size']  # Word-embedding attention size for attention and feed-forward sublayers
        # self.CVs = config['model']['char_vocabulary_size']
        # self.CEs = config['model']['char_emb_size']
        # self.Co = config['model']['char_out_size']
        self.Hn = config['model']['n_hidden']  # Number of hidden units in the LSTM cell

        # Define placeholders
        # TODO: Include characters
        self.is_training = tf.placeholder(tf.bool, name='is_training')
        self.x = tf.placeholder('int32', [self.Bs, None], name='x')  # number of batches and number of words
        # self.cx = tf.placeholder('int32', [self.Bs, None, None, W], name='cx')
        self.q = tf.placeholder('int32', [self.Bs, None], name='q')
        # self.cq = tf.placeholder('int32', [self.Bs, None, W], name='cq')
        self.y = tf.placeholder('float32', [self.Bs, None], name='y')
        self.y2 = tf.placeholder('float32', [self.Bs, None], name='y2')
        self.is_train = tf.placeholder('bool', [], name='is_train')
        self.new_emb_mat = tf.placeholder(tf.float32,
                                          [None, self.WEs],
                                          name='new_emb_mat')

        # Masks
        self.x_mask = tf.sign(self.x)
        self.q_mask = tf.sign(self.q)

        self.max_size_x = tf.shape(self.x)
        self.max_size_q = tf.shape(self.q)

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

        # Values for computing EM and F1 for dev
        self.EM_dev = []
        self.F1_dev = []
        self.EM_train = []
        self.F1_train = []

        if config['model']['is_Attention_Model']:
            self._build_forward_Attention()
        else:
            self._build_forward()
        self._build_loss()

        # Define optimizer and train step
        if config['train']['type'] == "Adadelta":
            self.learning_rate = tf.train.exponential_decay(
                learning_rate=config['train']['Adadelta']['learning_rate'],
                global_step=self.global_step,
                decay_steps=config['train']['Adadelta']['decay_steps'],
                decay_rate=config['train']['Adadelta']['decay_rate'],
                staircase=True)
            self.optimizer = tf.train.AdadeltaOptimizer(
                learning_rate=self.learning_rate)

        elif config['train']['type'] == "Adam":
		    # Decay_Rate is positive and therefore the - sign in this equation.
            self.learning_rate = config['train']['Adam']['learning_rate']*tf.multiply(
                                    tf.reduce_min([tf.pow(tf.cast(self.global_step, tf.float32), -self.config['train']['Adam']['decay_rate']), tf.multiply(tf.cast(self.global_step,tf.float32), tf.pow(tf.cast(config['train']['Adam']['WarmupSteps'], tf.float32),-config['train']['Adam']['decay_rate']-1.0))]),
                                    tf.pow(tf.cast(self.WEAs, tf.float32), -0.5))
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate if config['train']['Adam']['constant_LR'] else self.config['train']['Adam']['learning_rate'], beta1 = config['train']['Adam']['beta1'], beta2=config['train']['Adam']['beta2'], epsilon = config['train']['Adam']['epsilon'])

        # not sure if tf.contrib.layers.optimize_loss better than self.optimizer
        # Using contrib.layers to automatically log the gradients
        # self.train_step = tf.contrib.layers.optimize_loss(
        #    self.loss, global_step=self.global_step, learning_rate=self.learning_rate, optimizer='Adam',
        #    summaries=["gradients"], name='TIBINO')
        self.train_step = optimize_loss(self.loss, global_step=self.global_step, learning_rate=self.learning_rate, optimizer=self.config['train']['type'],
        summaries=["gradients"])

        # TODO: Understand the need for the moving average function
        # self.var_ema = None
        # if rep:
        #     self._build_var_ema()
        # if config.mode == 'train':
        #     self._build_ema()

        self.summary = tf.summary.merge_all()
        # Delete if not useful:
        # self.summary = tf.summary.merge(tf.get_collection("summaries", scope=self.scope))

        # Define a session for the model
        self.sess = tf.Session()
        # Add ops to save and restore all the variables.
        self.saver = tf.train.Saver(max_to_keep=100) # not to delete previous checkpoints
        # Initialize all variables
        if not config['model']['load_checkpoint']:
                self.sess.run(tf.global_variables_initializer())
        else:
                self._load()
        # Add a writer object to log the models's progress in the "train" folder
        self.writer = tf.summary.FileWriter(self.dir_plots + 'train',
                                            self.sess.graph)

        self.dev_writer = tf.summary.FileWriter(self.dir_plots + 'dev',
                                                self.sess.graph)

    def train(self, batch_idxs, dataset):
        """ Runs a train step given the input X and the correct label y.

            Parameters
            ----------
            batch_idxs : list of the idxs of each example
            dataset : the correspondent json file

        """
        # Combine the input dictionaries for all the features models
        feed_dict = self.get_feed_dict(batch_idxs,
                                       is_training=True,
                                       dataset=dataset)
        feed_dict['dropout:0'] = self.config['model']['input_keep_prob']

        if self.sess.run(self.global_step) % self.config['train']['steps_to_save'] == 0:
            summary, _, loss_val, global_step, max_x, max_q, Start_Index, End_Index = self.sess.run([self.summary, self.train_step, self.loss, self.global_step, self.max_size_x, self.max_size_q, self.Start_Index, self.End_Index],
                                       feed_dict=feed_dict)

            # Write the results to Tensorboard
            EM, F1, _, _ = EM_and_F1(self.answer, [Start_Index, End_Index])
            self.EM_train.append(EM)
            self.F1_train.append(F1)
            summary_EM = tf.Summary(value=[tf.Summary.Value(tag='EM', simple_value=EM)])
            summary_F1 = tf.Summary(value=[tf.Summary.Value(tag='F1', simple_value=F1)])
            self.writer.add_summary(summary, global_step)
            self.writer.add_summary(summary_F1, global_step)
            self.writer.add_summary(summary_EM, global_step)
            # Regularly save the models parameters

            self.saver.save(self.sess,
                            self.directory + 'ckpt/'+str(round(global_step/1000)) + 'k/model.ckpt')
            self.saver.save(self.sess, self.directory + 'model.ckpt')
        else:
            Start_Index, End_Index, _ = self.sess.run([self.Start_Index, self.End_Index,self.train_step], feed_dict=feed_dict)
            EM, F1 = EM_and_F1(self.answer, [Start_Index, End_Index])

        #To plot averaged EM and F1 during training
        self.EM_train.append(EM)
        self.F1_train.append(F1)

    def evaluate(self, batch_idxs, dataset):
        """ Compute F1 and EM for the dev dataset

            Parameters
            ----------
            batch_idxs : list of the idxs of each example
            dataset : the correspondent json file

        """
        # Combine the input dictionaries for all the features models
        feed_dict = self.get_feed_dict(batch_idxs, is_training=False, dataset=dataset)

        summary, max_x, max_q, Start_Index, End_Index, global_step = self.sess.run([self.summary, self.max_size_x, self.max_size_q, self.Start_Index, self.End_Index, self.global_step], feed_dict=feed_dict)
        # Write the results to Tensorboard
        EM_dev, F1_dev, y1_correct, y2_correct = EM_and_F1(self.answer, [Start_Index, End_Index])
        self.EM_dev.append(EM_dev)
        self.F1_dev.append(F1_dev)
        self.y1_correct_dev.append(y1_correct)
        self.y1_correct_dev.append(y2_correct)
        self.dev_writer.add_summary(summary, global_step=global_step)

    def _load(self):  # To load a checkpoint
        # TODO: Add structure to save/load different checkpoints.
        self.saver.restore(self.sess, self.directory + 'model.ckpt')

    def _embed_scaling(self, X):
        length_X = X.get_shape()[1]  # number of words in the passage
        # If the word2vec vector is scaled by a matrix
        if self.config['model_options']['word2vec_matrix_scaling']:
            X = tf.expand_dims(X, 2)
            X.set_shape([self.Bs, length_X, 1, self.WEs])
            X = tf.squeeze(tf.layers.conv2d(X,
                                            filters=self.WEAs,
                                            kernel_size=1,
                                            strides=1,
                                            use_bias=True,
                                            reuse=True,
                                            name="conv2d"))  # XW+B
        # If the word2vec vector is scaled by a vector
        elif self.config['model_options']['word2vec_vector_scaling']:
            weights = tf.get_variable('weights',
                                      shape=[self.WEAs],
                                      initializer=tf.ones_initializer())
            bias = tf.get_variable('bias',
                                   shape=[self.WEAs],
                                   initializer=tf.zeros_initializer())
            # slice the first WEAs columns
            X = tf.slice(X, [0, 0, 0], [self.Bs, tf.shape(X)[1], self.WEAs])
            X = tf.add(tf.multiply(X, weights), bias)  # xi = xi*w+b
        return X

    def _encoder(self, X, Q):
        # Compute the number of words in passage and question
        size_x = tf.shape(X)[-2]
        size_q = tf.shape(Q)[-2]
        if self.config['model']['full_trainable_encoder']:
            #Trainable encoder has the size of the biggest paragraph in training
            pos_emb_mat = tf.get_variable(
                "pos_emb_mat",
                shape=[self.config['pre']['max_paragraph_size'], self.WEAs],
                dtype=tf.float32,
                initializer=tf.contrib.layers.xavier_initializer())

            encoder_x = tf.nn.embedding_lookup(pos_emb_mat, tf.range(size_x))
            encoder_q = tf.nn.embedding_lookup(pos_emb_mat, tf.range(size_q))

        else:
            low_frequency = self.config['model']['encoder_low_freq']
            high_frequency = self.config['model']['encoder_high_freq']
            # Create a row vector with range(0,n) = [0,1,2,n-1], where n is the greatest size between x and q.
            pos = tf.cast(tf.expand_dims(tf.range(tf.cond(tf.greater(size_x, size_q), lambda: size_x, lambda: size_q)), 1), tf.float32)
            # Create a vector with all the exponents
            exponents = tf.multiply(tf.log(high_frequency/low_frequency), tf.divide(tf.range(self.WEAs/2), self.WEAs/2-1))
            # Power the base frequency by exponents
            freq = tf.expand_dims(tf.multiply(1/low_frequency, tf.exp(-exponents)), 0)
            if self.config['model']['encoder_learn_freq']:  # Encoder frequencies are trained
                freq_PG = tf.get_variable('wave_length', dtype=tf.float32, initializer=freq)
            else:  # Encoder frequencies are not trained
                freq_PG = freq

            freq_PG_scalar = tf.summary.scalar('wave_length', tf.reduce_mean(freq_PG))

            # Compute the encoder values
            encoder_angles = tf.matmul(pos, freq_PG)

            # Compute the encoder values
            encoder_sin = tf.sin(tf.matmul(pos, freq_PG))
            encoder_cos = tf.cos(tf.matmul(pos, freq_PG))

            # Concatenate both values
            encoder = tf.concat([encoder_sin, encoder_cos], axis=1)

            # Computes the encoder values for x and q
            encoder_x = tf.slice(encoder, [0, 0], [size_x, self.WEAs])

            if self.config['model']['encoder_no_cross']:
                # If no cross attention between encoders is desired
                freq_q_sum = tf.add(
                                    tf.multiply(self.config['model']['encoder_step_skip_size'],
                                                freq),
                                    (np.pi/2))
                encoder_q_angles = tf.add(
                                            encoder_angles,
                                            freq_q_sum)
                encoder_sin_q = tf.sin(encoder_q_angles)
                encoder_cos_q = tf.cos(encoder_q_angles)
                encoder_q =  tf.slice(
                                        tf.concat([encoder_sin_q,encoder_cos_q], axis=1),
                                        [0, 0],
                                        [size_q, self.WEAs])
            else:
                # If encoder in x and q are the same
                encoder_q = tf.slice(encoder,[0,0],[size_q,self.WEAs])

        # Encoding x and q
        x_encoded = tf.add(X, encoder_x)
        q_encoded = tf.add(Q, encoder_q)
        x_encoded = tf.nn.dropout(x_encoded, keep_prob=1.0 - tf.to_float(self.is_training)*self.config['train']['dropout_att_encoder'])
        q_encoded = tf.nn.dropout(q_encoded,keep_prob=1.0 - tf.to_float(self.is_training)*self.config['train']['dropout_att_encoder'])
        return x_encoded, q_encoded

    def _attention_layer(self, X1, mask, X2=None, scope=None):
        # Q = X1*WQ, K = X2*WK, V=X1*WV, X2 = X1 if X1 is None
        with tf.variable_scope(scope):
            length_X1 = X1.get_shape()[1]
            X1 = tf.expand_dims(X1, 2)
            X1.set_shape([self.Bs, length_X1, 1, self.WEAs])
            if X2 is None:
                length_X2 = length_X1
                # (SELF ATTENTION)
                # If X2 is None Compute Q = X1*WQ, K = X1*WK, V=X1*WV
                QKV = tf.squeeze(tf.layers.conv2d(X1,
                                                  filters=self.WEPs*3,
                                                  kernel_size=1,
                                                  strides=1,
                                                  name='QKV_Comp'))
                Q, K, V = tf.split(
                    QKV,
                    num_or_size_splits=[self.WEPs, self.WEPs, self.WEPs],
                    axis=2)
            else:
                # (CROSS ATTENTION)
                # If X2 is not none, compute Q = X1*WQ, K = X2*WK, V=X1*WV
                length_X2 = X2.get_shape()[1]
                KV = tf.squeeze(tf.layers.conv2d(X1,
                                                 filters=self.WEPs*2,
                                                 kernel_size=1,
                                                 strides=1,
                                                 name='KV_Comp'))
                K, V = tf.split(KV,
                                num_or_size_splits=[self.WEPs, self.WEPs],
                                axis=2)
                X2 = tf.expand_dims(X2, 2)
                X2.set_shape([self.Bs, length_X2, 1, self.WEAs])
                Q = tf.squeeze(tf.layers.conv2d(X2,
                                                filters=self.WEPs,
                                                kernel_size=1,
                                                strides=1,
                                                name='Q_Comp'))
                X2 = tf.squeeze(X2)
                X2.set_shape([self.Bs, length_X2, self.WEAs])
            X1 = tf.squeeze(X1)
            X1.set_shape([self.Bs, length_X1, self.WEAs])

            # Split Q, K, V for multi-head attention
            Q = tf.split(Q, num_or_size_splits=self.MHs, axis=2)
            K = tf.split(K, num_or_size_splits=self.MHs, axis=2)
            V = tf.split(V, num_or_size_splits=self.MHs, axis=2)
            # Compute transpose of K for multiplyting Q*K^T
            logits = tf.matmul(Q, tf.transpose(K, [0, 1, 3, 2]))

            # Sofmax in each head of the splitted Q and K softmax(Q*K^T):
            softmax = tf.nn.softmax(
                tf.add(
                    tf.divide(logits, tf.sqrt(tf.cast(self.WEAs, tf.float32))),
                    tf.multiply(1.0 - mask, VERY_LOW_NUMBER)),
                dim=-1)
            # Final mask is applied
            softmax = tf.multiply(mask, softmax)

            # Multihead attention
            # WV must be split into multi_head_size smaller matrices
            x_attention = tf.matmul(softmax, V)  # softmax(Q*K^T)*V
            # Concatenate everything together
            x_attention_concat = tf.concat(
                tf.unstack(x_attention,
                           axis=0,
                           num=self.MHs),
                axis=2)
            x_attention_concat.set_shape([self.Bs, length_X2, self.WEPs])
            # Compute softmax(Q*K^T)*V*WO
            x_final = tf.squeeze(
                tf.layers.conv2d(tf.expand_dims(x_attention_concat, 2),
                                 filters=self.WEAs,
                                 kernel_size=1,
                                 strides=1,
                                 name='Att_Comp'))
            # Add Dropout
            x_final_dropout = tf.nn.dropout(
                x_final,
                keep_prob=1.0-tf.to_float(self.is_training)*self.config['train']['dropout_att_sublayer'])
        return x_final_dropout

    def _layer_normalization(self, x, gain=1.0, scope=None):
        with tf.variable_scope(scope):
            # Compute variance and means
            mean, var = tf.nn.moments(x, axes=[-1])
            var += 1e-6  # to avoid NaN, if variance = 0
            normalized_x = tf.transpose(
                                tf.multiply(
                                    tf.add(mean,
                                           tf.transpose(x, [2, 0, 1])),
                                    tf.divide(gain, var)),
                                [1, 2, 0])
        # In Google Attention Model original code, there are these weights.
        # By now, they were turned off in FAB.
            if self.config['model_options']['norm_layer']:
                W_Scale = tf.get_variable('weight',
                                          shape=[self.WEAs],
                                          dtype=tf.float32,
                                          initializer=tf.ones_initializer())
                b_Scale = tf.get_variable('bias',
                                          shape=[self.WEAs],
                                          dtype=tf.float32,
                                          initializer=tf.zeros_initializer())
                normalized_x = normalized_x * W_Scale + b_Scale
        return normalized_x

    def _FeedForward_NN(self, X, scope=None):
        # Starting variables
        with tf.variable_scope(scope):
            length_X = X.get_shape()[1]
            X = tf.expand_dims(X, 2)
            X.set_shape([self.Bs, length_X, 1, self.WEAs])
            # Affine operation followed by Relu.
            # It is done by a convolution and it is the same as X*W+b
            affine_op = tf.layers.conv2d(X,
                                         filters=self.FFHs,
                                         kernel_size=1,
                                         strides=1,
                                         use_bias=True,
                                         activation=tf.nn.relu,
                                         name='affine_op_1')
            X = tf.squeeze(X)
            X.set_shape([self.Bs, length_X, self.WEAs])
            # Second affine oepration.
            # It is done by a convolution and it is the same as X*W+b
            output = tf.squeeze(tf.layers.conv2d(affine_op,
                                                 filters=self.WEAs,
                                                 kernel_size=1,
                                                 strides=1,
                                                 use_bias=True,
                                                 name='affine_op_2'))
            # Apply Dropout
            output = tf.nn.dropout(
                output,
                keep_prob=1.0 - tf.to_float(self.is_training)*self.config['train']['dropout_att_sublayer'])
        return output

    def _one_layer(self, Q, X, mask, scope, switch=False):
        # Defining masks and scopes
        if switch:
            X1 = X
            X2 = Q
            X1X1, X2X2, X2X1, X1X2 = 'xx', 'qq', 'qx', 'xq'
        else:
            X1 = Q
            X2 = X
            X1X1, X2X2, X2X1, X1X2 = 'qq', 'xx', 'xq', 'qx'
        with tf.variable_scope(scope):
            att_layer_X1X1 = self._layer_normalization(
                                tf.add(X1,
                                       self._attention_layer(X1=X1,
                                                       mask=mask[X1X1],
                                                       scope=X1X1)),
                                scope='norm_'+X1X1)

            output_1 = FF_X1X1 = self._layer_normalization(
                                    tf.add(att_layer_X1X1,
                                           self._FeedForward_NN(att_layer_X1X1,
                                                          'FF' + X1X1)),
                                    scope='norm_FF_'+X1X1)

            att_layer_X2X2 = self._layer_normalization(
                                tf.add(X2,
                                       self._attention_layer(X1=X2,
                                                       mask=mask[X2X2],
                                                       scope=X2X2)),
                                scope='norm_' + X2X2)

            att_layer_X1X2 = self._layer_normalization(
                                tf.add(att_layer_X2X2,
                                       self._attention_layer(
                                                       X1=FF_X1X1,
                                                       X2=att_layer_X2X2,
                                                       mask=mask[X2X1],
                                                       scope=X2X1)),
                                scope='norm_'+X2X1)

            output_2 = FF_X2X2 = self._layer_normalization(
                                    tf.add(att_layer_X1X2,
                                           self._FeedForward_NN(att_layer_X1X2,
                                                                'FF_' + X2X2)),
                                    scope='norm_FF_' + X2X2)
            if switch:
                return output_2, output_1
            else:
                return output_1, output_2

    def _one_layer_symmetric(self, Q, X, mask, scope, switch=False):  # Although switch input is not used here, it was added for compatibility with one layer function.
        with tf.variable_scope(scope):
            # Self-Atttention Layer Q
            att_layer_QQ = self._layer_normalization(
                                tf.add(Q,
                                       self._attention_layer(
                                                       X1=Q,
                                                       mask=mask['qq'],
                                                       scope='QQ')),
                                scope='norm_QQ')
            # FF neural network Q_Layer
            FF_QQ = self._layer_normalization(
                        tf.add(att_layer_QQ,
                               self._FeedForward_NN(att_layer_QQ,
                                                    'FF_QQ')),
                        scope='norm_FF_QQ')
            # Self-Atttention Layer X
            att_layer_XX = self._layer_normalization(
                                tf.add(X,
                                       self._attention_layer(
                                                       X1=X,
                                                       mask=mask['xx'],
                                                       scope='XX')),
                                scope='norm_XX')
            # FF neural network X_Layer
            FF_XX = self._layer_normalization(
                                tf.add(att_layer_XX,
                                       self._FeedForward_NN(att_layer_XX,
                                                            'FF_XX')),
                                scope='norm_FF_XX')
            # Cross attention of X and Q:
            att_layer_XQ = self._layer_normalization(
                                tf.add(att_layer_QQ,
                                       self._attention_layer(
                                                       X1=FF_XX,
                                                       X2=att_layer_QQ,
                                                       mask=mask['qx'],
                                                       scope='QX')),
                                scope='norm_QX')
            att_layer_QX = self._layer_normalization(
                                tf.add(att_layer_XX,
                                       self._attention_layer(
                                                       X1=FF_QQ,
                                                       X2=att_layer_XX,
                                                       mask=mask['xq'],
                                                       scope='XQ')),
                                scope='norm_XQ')
            # Output of X and Q:
            output_Q = self._layer_normalization(
                            tf.add(att_layer_XQ,
                                   self._FeedForward_NN(att_layer_XQ,
                                                        'FF_Q_out')),
                            scope='norm_FF_Q_out')
            output_X = self._layer_normalization(
                            tf.add(att_layer_QX,
                                   self._FeedForward_NN(att_layer_QX,
                                                        'FF_X_out')),
                            scope='norm_FF_X_out')
            return output_Q, output_X

    def _linear_sel(self, X, mask, scope):
        """ Select one vector among n vectors by max(w*X) """
        with tf.variable_scope(scope):
            W = tf.get_variable('W',
                                shape=[self.WEAs, 1],
                                dtype=tf.float32)
            summ = tf.summary.histogram('linear_sel', W)
            summm = tf.summary.scalar('linear_sel_sparsity', tf.nn.zero_fraction(W))
            logits = tf.reshape(
                        tf.matmul(tf.reshape(X, [-1, self.WEAs]), W),
                        [self.Bs, -1])  # [Bs, , 1]
            output = tf.nn.softmax(
                        tf.add(logits,
                               tf.multiply(1.0 - mask['x'], VERY_LOW_NUMBER)))
        return output, logits

    def _split_layer_sel(self, Q, X, mask, scope):
        """ Compute a self_attention, cross_attention
            and estimate the answer by linear_selec. """
        with tf.variable_scope(scope):
            self_attention_Q = self._layer_normalization(
                                    tf.add(Q,
                                           self._attention_layer(
                                                           X1=Q,
                                                           mask=mask['qq'],
                                                           scope='qq')),
                                    scope='norm_qq')
            FF_QQ = self._layer_normalization(
                        tf.add(self_attention_Q,
                               self._FeedForward_NN(self_attention_Q, 'FF_qq')),
                        scope='norm_FF_qq')
            self_attention_X = self._layer_normalization(
                                    self._attention_layer(
                                                    X1=X,
                                                    mask=mask['xx'],
                                                    scope='xx'),
                                    scope='norm_xx')
            cross_attention_X = self._layer_normalization(
                                    self._attention_layer(
                                                    X1=FF_QQ,
                                                    X2=X,
                                                    mask=mask['xq'],
                                                    scope='xq'),
                                    scope='norm_xq')
            FF_self_X = self._layer_normalization(
                                    self._FeedForward_NN(self_attention_X,
                                                         'FF_xx_self'),
                                    scope='norm_FF_xx_self')
            FF_cross_X = self._layer_normalization(
                                    self._FeedForward_NN(cross_attention_X,
                                                         'FF_xx_cross'),
                                    scope='norm_FF_xx_cross')
            W = tf.get_variable('W',
                                shape=[self.WEAs*3, 1],
                                dtype=tf.float32)  # [WEAs*3, 1]
            concat_all = tf.concat([X, FF_self_X, FF_cross_X], axis=2)
            logits = tf.reshape(
                        tf.matmul(
                            tf.reshape(concat_all, [-1, self.WEAs*3]), W),
                        [self.Bs, -1])  # W*x #[Bs, , 1]
            output = tf.nn.softmax(
                        tf.add(logits,
                               tf.multiply(1.0 - mask['x'], VERY_LOW_NUMBER)))
        return output, logits

    def _y_selection(self, Q, X, mask, scope, method="linear", y1_sel=None):
        if method == "linear":
            output, logits = self._linear_sel(X, mask, scope)
        elif method == "split_layer":
            output, logits = self._split_layer_sel(Q, X, mask, scope)
        return output, logits

    def _build_forward_Attention(self):
        config = self.config
        # Mask matrices
        mask = {}
        mask['q'] = tf.cast(tf.sign(self.q), tf.float32)
        mask['x'] = tf.cast(tf.sign(self.x), tf.float32)
        mask['qq'] = tf.cast(
                        tf.sign(
                            tf.matmul(tf.expand_dims(self.q, -1),
                                      tf.expand_dims(self.q, 1))), tf.float32)
        mask['xx'] = tf.cast(
                        tf.sign(
                            tf.matmul(tf.expand_dims(self.x, -1),
                                      tf.expand_dims(self.x, 1))), tf.float32)
        mask['xq'] = tf.cast(
                        tf.sign(
                            tf.matmul(tf.expand_dims(self.x, -1),
                                      tf.expand_dims(self.q, 1))), tf.float32)
        mask['qx'] = tf.cast(
                        tf.sign(
                            tf.matmul(tf.expand_dims(self.q, -1),
                                      tf.expand_dims(self.x, 1))), tf.float32)
        with tf.variable_scope("word_emb"):
            # TODO: Having a config variable for this is not the best solution
            # TODO: Save the embedding matrix somewhere else
            word_emb_mat = tf.get_variable(
                "word_emb_mat",
                dtype=tf.float32,
                initializer=config['model']['emb_mat_unk_words'])  # [WVs,WEAs]
            if config['pre']['use_glove_for_unk']:
                word_emb_mat = tf.concat([word_emb_mat, self.new_emb_mat],
                                         axis=0)
            with tf.name_scope("word"):
                Ax = tf.nn.embedding_lookup(word_emb_mat, self.x)  # [Bs,Ps,Hn]
                Aq = tf.nn.embedding_lookup(word_emb_mat, self.q)  # [Bs,Qs,Hn]

        # Scaling word2vec matrices before adding encoder
        with tf.variable_scope('Scaling') as scope:
            if self.config['model_options']['word2vec_vector_scaling']:
                weigths = tf.get_variable('weights', shape=self.WEAs)
                bias = tf.get_variable('bias',
                                       shape=self.WEAs,
                                       initializer=tf.zeros_initializer())
            elif self.config['model_options']['word2vec_matrix_scaling']:
                with tf.variable_scope('conv2d'):
                    # If the scaling matrix was previously trained
                    if self.config['weights_init']['pre_trained_scaling_matrix']:
                        weigths = tf.get_variable(
                                    'kernel',
                                    initializer=np.load('./kernel.npy'),
                                    trainable=False)
                    else:  # If the scaling matrix was not previously trained
                        # In order to be orthonormal
                        weights_init = np.random.random((1, 1, self.WEs, self.WEAs)).astype(np.float32)  # might not work properly if WEs different from WEAs.
                        _, _, U = np.linalg.svd(weights_init, full_matrices=False)
                        weigths = tf.get_variable(
                                    'kernel',
                                    initializer=U)
                    bias = tf.get_variable('bias',
                                           shape=[self.WEAs],
                                           initializer=tf.zeros_initializer())
            scope.reuse_variables()
            x_scaled = self._embed_scaling(Ax)
            q_scaled = self._embed_scaling(Aq)

        # Encoding Variables
        if config['model']['time_encoding']:
            with tf.variable_scope("Encoding"):
                x_scaled, q_scaled = self._encoder(x_scaled, q_scaled)

        # Defining functions according to config.json
        # They are used later in the final model
        # Number of layers until computation of y1
        num_layers_pre = config['model']['n_pre_layer']
        # Layers after computation of y1 to compute y2
        num_layers_post = config['model']['n_post_layer']
        switch = lambda i: (i%2 == 1) if config['model_options']['switching_model'] else lambda i: False
        layer_func = self._one_layer_symmetric if config['model_options']['symmetric'] else self._one_layer

        # Computing following layers after encoder
        q = [self._layer_normalization(q_scaled, scope='norm_q_scaled')] if config['model_options']['encoder_normalization'] else [q_scaled]
        x = [self._layer_normalization(x_scaled, scope='norm_x_scaled')] if config['model_options']['encoder_normalization'] else [x_scaled]
        for i in range(num_layers_pre+num_layers_post):
            q_i, x_i = layer_func(q[i], x[i], mask, 'layer_'+str(i), switch=switch(i))
            q.append(q_i)
            x.append(x_i)

        # Computing outputs
        self.yp, self.logits_y1 = self._y_selection(
                                              Q=q[-1-num_layers_post],
                                              X=x[-1-num_layers_post],
                                              mask=mask,
                                              scope='y1_sel',
                                              method=config['model']['y1_sel'])
        self.yp2, self.logits_y2 = self._y_selection(
                                               Q=q[-1],
                                               X=x[-1],
                                               scope='y2_sel',
                                               mask=mask,
                                               method=config['model']['y2_sel'])
        self.Start_Index = tf.argmax(self.logits_y1, axis=-1)
        self.End_Index = tf.argmax(self.logits_y2, axis=-1)

    def _build_forward(self):
        """
            Builds the model's feedforward network.

        """
        config = self.config
        with tf.variable_scope("word_emb"), tf.device("/cpu:0"):
            # TODO: Having a config variable for this is the best solution
            # TODO: Save the embedding matrix somewhere else
            word_emb_mat = tf.get_variable(
                "word_emb_mat",
                dtype=tf.float32,
                initializer=['model']['emb_mat_unk_words'])  # [WVs,WEs]
            if config['pre']['use_glove_for_unk']:
                word_emb_mat = tf.concat([word_emb_mat, self.new_emb_mat],
                                         axis=0)
            with tf.name_scope("word"):
                Ax = tf.nn.embedding_lookup(word_emb_mat, self.x)  # [Bs,Ps,Hn]
                Aq = tf.nn.embedding_lookup(word_emb_mat, self.q)  # [Bs,Qs,Hn]
                self.tensor_dict['x'] = Ax
                self.tensor_dict['q'] = Aq
        # Build the LSTM cell with dropout
        cell = tf.contrib.rnn.BasicLSTMCell(
            self.Hn,
            state_is_tuple=True,
            forget_bias=config['model']['forget_bias'])
        dropout_cell = tf.contrib.rnn.DropoutWrapper(
            cell,
            input_keep_prob=self.keep_prob)

        # Calculate the number of used values in the each matrix
        x_len = tf.reduce_sum(tf.cast(self.x_mask, 'int32'), 1)  # [Bs]
        q_len = tf.reduce_sum(tf.cast(self.q_mask, 'int32'), 1)  # [Bs]

        with tf.variable_scope("prepro"):
            # [Bs, Qs, 2Hn], [Bs, Hn]
            (fw_u, bw_u), ((_, fw_u_f), (_, bw_u_f)) = \
                tf.nn.bidirectional_dynamic_rnn(cell_fw=dropout_cell,
                                                cell_bw=dropout_cell,
                                                inputs=Aq,
                                                sequence_length=q_len,
                                                dtype='float',
                                                scope='u1')
            u = tf.concat([fw_u, bw_u], axis = 2)
            if config['model']['share_lstm_weights']:
                tf.get_variable_scope().reuse_variables()
                # [Bs, Ps, 2Hn] Originally these cells didnt have dropout
                (fw_h, bw_h), _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=dropout_cell,
                                                                  cell_bw=dropout_cell,
                                                                  inputs=Ax,
                                                                  sequence_length=x_len,
                                                                  dtype='float',
                                                                  scope='u1')
            else:
                # [Bs, Ps, 2Hn] Originally these cells didnt have dropout
                (fw_h, bw_h), _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=dropout_cell,
                                                                  cell_bw=dropout_cell,
                                                                  inputs=Ax,
                                                                  sequence_length=x_len,
                                                                  dtype='float',
                                                                  scope='h1')

            h = tf.concat([fw_h, bw_h], axis=2)  # [Bs, Ps, 2Hn]
            self.tensor_dict['u'] = u
            self.tensor_dict['h'] = h

        with tf.variable_scope("main"):
            # AttentionLayer
            p0 = attention_layer(self.x, self.q, Ax, Aq, x_len, q_len, self.Hn*2, self.Bs, h, u, scope='p0') # [Bs, Ps, 8Hn]
            # Hidden size multiplied by two because of bidirectional layer

            # [Bs, Ps, 8Hn]
            cell_after_att = tf.contrib.rnn.BasicLSTMCell(self.Hn, state_is_tuple=True, forget_bias=config['model']['forget_bias'])
            dropout_cell_after_att = tf.contrib.rnn.DropoutWrapper(
                cell_after_att,
                input_keep_prob=self.keep_prob)
            (fw_g0, bw_g0), _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=dropout_cell_after_att,
                                                                cell_bw=dropout_cell_after_att,
                                                                inputs=p0,
                                                                sequence_length=x_len,
                                                                dtype=tf.float32,
                                                                scope='g0')
            g0 = tf.concat([fw_g0, bw_g0], axis=2)

            # [Bs, Ps, 8Hn]
            cell_after_att_2 = tf.contrib.rnn.BasicLSTMCell(
                self.Hn,
                state_is_tuple=True,
                forget_bias=config['model']['forget_bias'])
            dropout_cell_after_att_2 = tf.contrib.rnn.DropoutWrapper(
                cell_after_att_2,
                input_keep_prob=self.keep_prob)
            (fw_g1, bw_g1), _ = tf.nn.bidirectional_dynamic_rnn(
                                    cell_fw=dropout_cell_after_att_2,
                                    cell_bw=dropout_cell_after_att_2,
                                    inputs=g0,
                                    sequence_length=x_len,
                                    dtype='float',
                                    scope='g1')

            g1 = tf.concat([fw_g1, bw_g1, p0], axis=2)

            w_y1 = tf.get_variable('w_y1',
                                   shape=[10*self.Hn, 1],
                                   dtype=tf.float32)
            logits_y1 = tf.reshape(
                tf.matmul(
                    tf.concat(tf.unstack(value=g1, axis=0), axis=0),
                    w_y1),
                [self.Bs, -1]) + tf.multiply(tf.cast(1 - self.x_mask, tf.float32),
                        VERY_LOW_NUMBER)  # mask

            smax = tf.nn.softmax(logits_y1, 1)
            a1i = tf.matmul(tf.expand_dims(smax, 1), g1)  # softsel
            a1i = tf.tile(a1i, [1, self.Ps, 1])

            # [Bs, Sn, Ss, 2Hn]
            cell_y2 = tf.contrib.rnn.BasicLSTMCell(
                self.Hn,
                state_is_tuple=True,
                forget_bias=config['model']['forget_bias'])
            dropout_cell_y2 = tf.contrib.rnn.DropoutWrapper(
                cell_y2,
                input_keep_prob=self.keep_prob)
            (fw_g2, bw_g2), _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=dropout_cell_y2,
                cell_bw=dropout_cell_y2,
                inputs=tf.concat([p0, g1, a1i, tf.multiply(g1, a1i)], axis=2),
                sequence_length=x_len,
                dtype='float',
                scope='g2')
            g2 = tf.concat([fw_g2, bw_g2], axis=2)
            # TODO: Rewrite the get_logits function
            w_y2 = tf.get_variable('w_y2',
                                   shape=[2*self.Hn, 1],
                                   dtype=tf.float32)
            logits_y2 = tf.reshape(
                tf.matmul(
                    tf.concat(tf.unstack(value=g2,axis=0),axis=0),
                    w_y2),
                [self.Bs,-1]) + tf.multiply(tf.cast(1-self.x_mask, tf.float32), VERY_LOW_NUMBER) #mask

            yp = smax  # [Bs, Ps]
            yp2 = tf.nn.softmax(logits_y2)  # [Bs,Ps]

            self.tensor_dict['g1'] = g1
            self.tensor_dict['g2'] = g2

            self.logits_y1 = logits_y1
            self.logits_y2 = logits_y2
            self.yp = yp
            self.yp2 = yp2
            self.Start_Index = tf.argmax(self.logits_y1, axis=-1)
            self.End_Index = tf.argmax(self.logits_y2, axis=-1)

    def _build_loss(self):
        """
            Defines the model's loss function.
        """
        # TODO: add collections if useful. Otherwise delete them.
        ce_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=self.logits_y1, labels=self.y))
        # tf.add_to_collection('losses', ce_loss)
        ce_loss2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=self.logits_y2, labels=self.y2))
        # tf.add_to_collection("losses", ce_loss2)

        # self.loss = tf.add_n(tf.get_collection('losses', scope=self.scope), name='loss')
        self.loss = tf.add_n([ce_loss, ce_loss2])
        tf.summary.scalar('ce_loss', ce_loss)
        tf.summary.scalar('ce_loss2', ce_loss2)
        tf.summary.scalar('loss', self.loss)
        # tf.add_to_collection('ema/scalar', self.loss)

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
    def get_feed_dict(self, batch_idxs, is_training, dataset):
        feed_dict = {}
        x = []
        q = []
        y1 = []
        y2 = []
        label_smoothing = self.config['train']['label_smoothing']

        def word2id(word):  # to convert a word to its respective id
            if self.config['pre']['lower_word']:
                word = word.lower()
            if word in dataset['shared']['known_word2idx']:
                return dataset['shared']['known_word2idx'][word] + self.WVs
            elif word in dataset['shared']['unk_word2idx']:
                return dataset['shared']['unk_word2idx'][word]
            else:
                return 1  # unknown word

        # Padding for passages, questions and answers.
        # The answers output are (1-label_smoothing)/len(x)
        def padding(seq, label_smoothing=1.0, max_size=None):  # for padding a batch
            seq_len = [len(seq[i]) for i in range(len(seq))]
            if max_size is None:
                max_size = max(seq_len)
            new_seq = [np.concatenate([np.array(seq[i]), np.zeros([max_size-len(seq[i])])], axis=0) for i in range(len(seq))]
            new_seq_y = [np.concatenate([np.ones(seq_len[i])*(1.0-label_smoothing)/seq_len[i], np.zeros([max_size-len(seq[i])])], axis=0) for i in range(len(seq))]
            return np.int_(new_seq), new_seq_y

        # TODO: Add characters
        # convert every word to its respective id
        for i in batch_idxs:
            qi = list(map(
                word2id,
                dataset['data']['q'][i]))
            rxi = dataset['data']['*x'][i]
            yi = dataset['data']['y'][i]
            xi = list(map(word2id, dataset['shared']['x'][rxi[0]][rxi[1]]))
            q.append(qi)
            x.append(xi)
            # Get all the first indices in the sequence
            y1.append([y[0] for y in yi])
            # Get all the second indices... and correct for -1
            y2.append([y[1]-1 for y in yi])

        self.answer = [y1, y2]
        # Padding
        if self.config['train']['check_available_memory']:
            x, new_seq_y = padding(x,
                                   label_smoothing=label_smoothing,
                                   max_size=self.config['pre']['max_paragraph_size'])
            q, _ = padding(q, max_size=self.config['pre']['max_question_size'])
        else:
            x, new_seq_y = padding(x, label_smoothing=label_smoothing)
            q, _ = padding(q)
        y1_new = new_seq_y
        y2_new = np.copy(new_seq_y)
        for i in range(self.Bs):
            y1_new[i][y1[i]] += label_smoothing
            y2_new[i][y2[i]] += label_smoothing

        # cq = np.zeros([self.Bs, self.Qs, self.Ws], dtype='int32')
        feed_dict[self.x] = x
        # feed_dict[self.cx] = cx
        feed_dict[self.q] = q
        # feed_dict[self.cq] = cq
        feed_dict[self.y] = y1_new
        feed_dict[self.y2] = y2_new
        feed_dict[self.is_training] = is_training
        if self.config['pre']['use_glove_for_unk']:
            feed_dict[self.new_emb_mat] = dataset['shared']['emb_mat_known_words']

        return feed_dict


def attention_layer(x, q, x_embed, q_embed, x_len, q_len, hidden_size, batch_size, h, u, scope=None):
    """
        Define attention mechanism between vectors h (question) and u (paragraph):
             att(m,n)  = vec1 . h_m + vec2 . u_n + (vec3 * h_m) . u_n
        where '.' is the dot product and '*' the elementwise product
    """
    mask_matrix = tf.sign(
                    tf.matmul(
                        tf.expand_dims(x, -1),
                        tf.expand_dims(q, -1),
                        transpose_b=True))
    with tf.variable_scope(scope):
        vec1_att = tf.get_variable("att1", dtype='float', shape=[hidden_size, 1])  # [Hs, 1]
        vec2_att = tf.get_variable("att2", dtype='float', shape=[hidden_size, 1])  # [Hs, 1]
        vec3_att = tf.get_variable("att3", dtype='float', shape=[hidden_size])  # [Hs]

    # shaping all the vectors into a matrix to compute attention values in one matrix multiplication
    shaped_h = tf.concat(
        tf.unstack(
            value=h,
            axis=0),
        axis=0)  # [Hs, Ps * Bs]

    shaped_u = tf.concat(
        tf.unstack(
            value=u,
            axis=0),
        axis=0)  # [Hs, Qs * Bs]

    # Computation of vec1 * h + vec2 * u
    att_1_Product = tf.reshape(
        tf.matmul(
            shaped_h,
            vec1_att),
        shape=[batch_size, 1, -1])  # [Bs, 1, Ps]

    att_2_Product = tf.reshape(
        tf.matmul(
            shaped_u,
            vec2_att),
        shape=[batch_size, -1, 1])  # [Bs, Qs, 1]

    att_1_Product = tf.tile(
        att_1_Product,
        [1, tf.reduce_max(q_len), 1])  # [Bs, Qs, Ps]

    att_2_Product = tf.tile(
        att_2_Product,
        [1, 1, tf.reduce_max(x_len)])  # [Bs, Qs, Ps]

    #  of (vec3 * h)  . u

    h_vectorized = tf.multiply(h, vec3_att)  # vect 3 * h
    att_3_Product = tf.matmul(u, h_vectorized, transpose_b=True)  # ((vec 3 * h) . u [Bs, Qs, Ps]
    att_final = tf.transpose(
        att_1_Product + att_2_Product + att_3_Product,
        perm=[0, 2, 1])  # [Bs, Ps, Qs]

    att_final_masked = att_final + tf.multiply(tf.cast(1 - mask_matrix, tf.float32), -3e15) # masking

    # paragraph to question attention
    p2q = tf.multiply(
        tf.nn.softmax(logits=att_final_masked),
        tf.cast(mask_matrix, 'float')
    )  # computing logits, taking into account mask

    U_a = tf.matmul(p2q, u)

    q2p = tf.nn.softmax(
        tf.reduce_max(att_final_masked, axis=-1))

    H_a = tf.tile(
        tf.matmul(
            tf.expand_dims(q2p, 1),
            h),
        [1, tf.reduce_max(x_len), 1])

    G = tf.concat([h, U_a, tf.multiply(h, U_a), tf.multiply(h, H_a)], axis=-1)
    return G
