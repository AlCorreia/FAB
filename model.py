import inspect
import math
import numpy as np
import os
import pandas as pd
import random
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
from tqdm import tqdm
import pdb

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
        self.MHs = config['model']['multi_head_size']  #Multi-Head
        self.FFHs = config['model']['FeedForward_Hidden_Size']
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
        self.q = tf.placeholder('int32', [self.Bs, None], name='q')
        # self.cq = tf.placeholder('int32', [self.Bs, None, W], name='cq')
        self.y = tf.placeholder('bool', [self.Bs, None], name='y')
        self.y2 = tf.placeholder('bool', [self.Bs, None], name='y2')
        self.is_train = tf.placeholder('bool', [], name='is_train')
        self.new_emb_mat = tf.placeholder(tf.float32, [None, self.WEs], name='new_emb_mat')
        
        #Masks
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

        #Values for computing EM and F1 for dev 
        self.EM_dev = []
        self.F1_dev = []

        if config['model']['is_Attention_Model']:
            self._build_forward_Attention()
        else:
            self._build_forward()
        self._build_loss()

        # Define optimizer and train step
        # TODO: We could add the optimizer option to the config file. ADAM for now.
        self.optimizer = tf.train.AdadeltaOptimizer(learning_rate=self.learning_rate)
        #self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

        #not sure if tf.contrib.layers.optimize_loss better than self.optimizer
        # Using contrib.layers to automatically log the gradients
        #self.train_step = tf.contrib.layers.optimize_loss(
        #    self.loss, global_step=self.global_step, learning_rate=self.learning_rate, optimizer='Adam',
        #    summaries=["gradients"], name='TIBINO')
        self.train_step = self.optimizer.minimize(self.loss, global_step=self.global_step)

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
        self.saver = tf.train.Saver(max_to_keep = 100)#to not delete previous checkpoints
        # Initialize all variables
        if not config['model']['load_checkpoint']:
                self.sess.run(tf.global_variables_initializer())
        else:
                self._load()
        # Add a writer object to log the models's progress in the "train" folder
        self.writer = tf.summary.FileWriter(self.directory + 'train',
                                            self.sess.graph)


    def train(self, batch_idxs, dataset):
        """ Runs a train step given the input X and the correct label y.

            Parameters
          d  ----------
            batch_idxs : list of the idxs of each example
            dataset : the correspondent json file

        """
        # Combine the input dictionaries for all the features models
        feed_dict = self.get_feed_dict(batch_idxs, is_training=True, dataset=dataset)

        summary, _, loss_val, global_step, max_x, max_q, Start_Index, End_Index = self.sess.run([self.summary, self.train_step, self.loss,self.global_step,self.max_size_x, self.max_size_q,self.Start_Index,self.End_Index],
                                   feed_dict=feed_dict)
        # Write the results to Tensorboard
        EM, F1 = EM_and_F1(self.answer,[Start_Index,End_Index])
        summary_EM = tf.Summary(value=[tf.Summary.Value(tag='EM', simple_value=EM)])
        summary_F1 = tf.Summary(value=[tf.Summary.Value(tag='F1', simple_value=F1)])
        self.writer.add_summary(summary, global_step)
        self.writer.add_summary(summary_F1, global_step)
        self.writer.add_summary(summary_EM, global_step)
        # Regularly save the models parameters      
        if global_step % self.config['model']['steps_to_save'] == 0:
            self.saver.save(self.sess, self.directory + 'ckpt/'+str(round(global_step/1000))+'k/model.ckpt')
            self.saver.save(self.sess, self.directory + 'model.ckpt')

    def evaluate(self, batch_idxs, dataset):
        """ Compute F1 and EM for the dev dataset

            Parameters
          d  ----------
            batch_idxs : list of the idxs of each example
            dataset : the correspondent json file

        """
        # Combine the input dictionaries for all the features models
        feed_dict = self.get_feed_dict(batch_idxs, is_training=False, dataset=dataset)

        summary, max_x, max_q, Start_Index, End_Index = self.sess.run([self.summary,self.max_size_x, self.max_size_q,self.Start_Index,self.End_Index], feed_dict=feed_dict)
        # Write the results to Tensorboard
        EM_dev, F1_dev = EM_and_F1(self.answer,[Start_Index,End_Index])
        self.EM_dev.append(EM_dev)
        self.F1_dev.append(F1_dev)



    def _load(self): #To load a checkpoint
        #TODO: Add an structure to allow the user to save/load different checkpoints.
        self.saver.restore(self.sess,self.directory + 'model.ckpt')

    def _build_forward_Attention(self):

        def embed_scaling (x):
            W_Scal = tf.get_variable('W_Scal', shape = [self.WEs, self.WEs])
            b_Scal = tf.get_variable('b_Scal', shape = [1, self.WEs])
            affine_op = tf.add(tf.matmul(tf.reshape(x,[-1,self.WEs]),W_Scal),b_Scal) #W1*x+b1
            x_reshaped = tf.reshape(affine_op, [self.Bs,-1,self.WEs])
            return x_reshaped

        def encoder (X,Q):
            #Compute the number of words in passage and question
            size_x = tf.shape(X)[-2]
            size_q = tf.shape(Q)[-2]
            #Create a row vector with range(0,n) = [0,1,2,n-1], where n is the greatest size between x and q.
            pos = tf.cast(tf.expand_dims(tf.range(tf.cond(tf.greater(size_x,size_q),lambda: size_x, lambda: size_q)), 1),tf.float32)
            #Create a vector with all the exponents
            exponents = tf.divide(tf.range(self.WEs/2),self.WEs/2-1)
            #Power the base frequency by exponents
            freq_PG = tf.expand_dims(tf.pow(1/config['model']['encoder_base_freq'],exponents),0)

            #Compute the encoder values
            encoder_sin = tf.sin(tf.matmul(pos,freq_PG))
            encoder_cos = tf.cos(tf.matmul(pos,freq_PG))

            #Concatenate both values
            encoder = tf.concat([encoder_sin,encoder_cos], axis = 1)

            #Computes the encoder values for x and q
            encoder_x = tf.slice(encoder,[0,0],[size_x,self.WEs])
            encoder_q = tf.slice(encoder,[0,0],[size_q,self.WEs])

	#Encoding x and q
            x_encoded = tf.add(X, encoder_x)
            q_encoded = tf.add(Q, encoder_q)
            return x_encoded, q_encoded

        def attention_layer(X, mask,X2 = None, scope=None):
            #Self-Attention is defined as:  softmax(X*W_sym*X')*X*WV
            with tf.variable_scope(scope):
                #In order to get a symmetric matrix W_V, only its eigenvectors and eigenvalues are variables.
                WQ_EigVec = tf.nn.l2_normalize(tf.get_variable(name = 'WQEigVec', shape = [self.WEs,self.WEs], dtype = tf.float32), dim = 0)
                WQ_EigenVal = tf.get_variable(name = 'WGEigVal', shape = [self.WEs])
                WV = tf.get_variable(name = 'WV', shape = [self.WEs,self.WEs], dtype = tf.float32)

                #W_sym = WQ_EigVec*EigenVal*EigVec
                # x*EigVec
                
                x_proj = tf.matmul(X,
                    tf.tile(tf.expand_dims(WQ_EigVec,0),[self.Bs,1,1]))
                # x*EigVec is split into multi_head_size smaller matrices
                x_proj_split = tf.transpose(tf.split(x_proj,num_or_size_splits = self.MHs, axis = 2),[1,2,0,3])
                # x*EigVec*EigVal computed
                WQ_EigenVal_Split = tf.split(WQ_EigenVal,num_or_size_splits = self.MHs, axis = 0)
                x_proj_scaled = tf.multiply(x_proj_split,WQ_EigenVal_Split)
                if X2 is None:
                    logits = tf.matmul(tf.transpose(x_proj_split,[0,2,1,3]),tf.transpose(x_proj_scaled,[0,2,3,1]))
                else:
                    X2_proj = tf.matmul(X2,
                        tf.tile(tf.expand_dims(WQ_EigVec,0),[self.Bs,1,1]))
                    X2_proj_split = tf.transpose(tf.split(X2_proj,num_or_size_splits = self.MHs, axis = 2),[1,2,0,3])
                    logits = tf.matmul(tf.transpose(X2_proj_split,[0,2,1,3]),tf.transpose(x_proj_scaled,[0,2,3,1]))

                #(x*EigVec) * (x*EigVec*EigVal)' 
                #Sofmatx with masking
                softmax = tf.nn.softmax(
                            tf.add(
                                tf.divide(tf.transpose(logits,[1,0,2,3]),tf.sqrt(tf.cast(self.WEs,tf.float32))),
                                tf.multiply(1 - mask, VERY_LOW_NUMBER)
                                ), dim = -1)
                #Final mask is applied
                softmax = tf.multiply (mask,softmax)
                #Computed the new x vector accoring to weights from softmax
                x_weighted = tf.matmul(softmax,tf.tile(tf.expand_dims(X,0),[self.MHs,1,1,1]))
                #Because of multihead attention, WV must be split into multi_head_size smaller matrices
                WV_Split = tf.split(WV, num_or_size_splits = self.MHs, axis = 1)
                #x_weighted*Wv
                x_weighted_proj = tf.matmul(x_weighted,tf.tile(tf.expand_dims(WV_Split,1),[1,self.Bs,1,1]))
                #Concatenate everything togeter
                x_weighted_proj_concat = tf.concat(tf.unstack(x_weighted_proj, axis = 0), axis = 2)
            return x_weighted_proj_concat

        def layer_normalization (x, gain = 1.0):
            mean, var = tf.nn.moments(x, axes=[-1]) #To compute variance and means
            var += 1e-30 #to avoid NaN, if variance = 0
            normalized_x = tf.transpose(
                                tf.multiply(
                                    tf.add(mean,
                                        tf.transpose(x,[2,0,1])), #Transpose for add and multiply operations
                                    tf.divide(gain,var)),
				[1,2,0])
            return normalized_x

        def FeedForward_NN(x,scope = None):
            #Starting variables
            with tf.variable_scope(scope):
                W1 = tf.get_variable('W1', shape = [self.WEs, self.FFHs])
                b1 = tf.get_variable('b1', shape = [1, self.FFHs])
                W2 = tf.get_variable('W2', shape = [self.FFHs, self.WEs])
                b2 = tf.get_variable('b2', shape = [1, self.WEs])
                
                #Computation of #W2*Relu(W1*x+b1)+b2
                affine_op1 = tf.add(tf.matmul(tf.reshape(x,[-1,self.WEs]),W1),b1) #W1*x+b1
                nonlinear_op = tf.nn.relu(affine_op1) #Relu(W1*x+b1)
                affine_op2 = tf.add(tf.matmul(nonlinear_op,W2),b2) #W2*Relu(W1*x+b1)+b2
                output = tf.reshape(affine_op2, [self.Bs,-1,self.WEs]) #Reshaping
            return output

        def one_layer (X1, X2, mask, scope):
            with tf.variable_scope(scope):
                att_layer_X1X1 = layer_normalization(tf.add(X1, attention_layer(X = X1, mask = mask['x1x1'], scope = 'x1x1')))
                att_layer_X2X2 = layer_normalization(tf.add(X2, attention_layer(X = X2, mask = mask['x2x2'], scope = 'x2x2')))
                FF_X1X1 = layer_normalization(tf.add(att_layer_X1X1, FeedForward_NN(att_layer_X1X1,'FF_11')))
                att_layer_X1X2 = layer_normalization(tf.add(att_layer_X2X2,attention_layer(X = FF_X1X1, X2 = att_layer_X2X2, mask = mask['x2x1'], scope = 'x2x1')))
                FF_X2X2 = layer_normalization(tf.add(att_layer_X1X2,FeedForward_NN(att_layer_X1X2,'FF_22')))
            return FF_X1X1, FF_X2X2

        def y_selection(X, mask, scope):
            with tf.variable_scope(scope):
                W = tf.get_variable('W', shape = [self.WEs, 1], dtype = tf.float32)
                logits = tf.reshape(tf.matmul(tf.reshape(X,[-1,self.WEs]),W), [self.Bs,-1]) #W*x
                output = tf.nn.softmax(tf.add(logits,tf.multiply(1.0 - mask, VERY_LOW_NUMBER)))
            return output, logits
    

        config = self.config
        #Mask matrices
        mask = {}
        mask['x1'] = tf.cast(tf.sign(self.q),tf.float32)
        mask['x2'] = tf.cast(tf.sign(self.x),tf.float32)
        mask['x1x1'] = tf.cast(tf.sign(tf.matmul(tf.expand_dims(self.q,-1),tf.expand_dims(self.q,1))),tf.float32)
        mask['x2x2'] = tf.cast(tf.sign(tf.matmul(tf.expand_dims(self.x,-1),tf.expand_dims(self.x,1))),tf.float32)
        mask['x2x1'] = tf.cast(tf.sign(tf.matmul(tf.expand_dims(self.x,-1),tf.expand_dims(self.q,1))),tf.float32)
        with tf.variable_scope("word_emb"), tf.device("/cpu:0"):
            # TODO: I am not sure that having a config variable for this is the best solution
            # TODO: Save the embedding matrix somewhere other than the config file
            word_emb_mat = tf.get_variable("word_emb_mat",
                                       dtype=tf.float32,
                                       initializer =  config['model']['emb_mat_unk_words']) # [self.WVs, self.WEs]
            if config['pre']['use_glove_for_unk']:
                word_emb_mat = tf.concat([word_emb_mat, self.new_emb_mat], axis = 0)
            with tf.name_scope("word"):
                Ax = tf.nn.embedding_lookup(word_emb_mat, self.x)  # [Bs, Ps, Hn]
                Aq = tf.nn.embedding_lookup(word_emb_mat, self.q)  # [Bs, Qs, Hn]

        with tf.variable_scope('Scaling') as scope:
            x_scaled = embed_scaling(Ax)
            scope.reuse_variables()
            q_scaled = embed_scaling(Aq)

        #Encoding Variables
        x_encoded, q_encoded = encoder (x_scaled,q_scaled)
        #Computing all attentions
        q_1, x_1 = one_layer(q_encoded, x_encoded, mask, 'layer_0')
        q_2, x_2 = one_layer(q_1, x_1, mask, 'layer_1')
        q_3, x_3 = one_layer(q_2, x_2, mask, 'layer_2')
        q_4, x_4 = one_layer(q_3, x_3, mask, 'layer_3')
        q_5, x_5 = one_layer(q_4, x_4, mask, 'layer_4')
        q_6, x_6 = one_layer(q_5, x_5, mask, 'layer_5')
        self.yp, self.logits_y1 = y_selection(X = x_6,scope = 'y1_sel', mask = mask['x2'])
        self.yp2, self.logits_y2 = y_selection(X = x_6,scope = 'y2_sel', mask = mask['x2'])
        self.Start_Index = tf.argmax(self.logits_y1, axis=-1)
        self.End_Index = tf.argmax(self.logits_y2, axis=-1)
        
    def _build_forward(self):
        """
            Builds the model's feedforward network.

        """
        config = self.config
        with tf.variable_scope("word_emb"), tf.device("/cpu:0"):
            # TODO: I am not sure that having a config variable for this is the best solution
            # TODO: Save the embedding matrix somewhere other than the config file
            word_emb_mat = tf.get_variable("word_emb_mat",
                                       dtype=tf.float32,
                                       initializer =  config['model']['emb_mat_unk_words']) # [self.WVs, self.WEs]
            if config['pre']['use_glove_for_unk']:
                word_emb_mat = tf.concat([word_emb_mat, self.new_emb_mat], axis = 0)
            with tf.name_scope("word"):
                Ax = tf.nn.embedding_lookup(word_emb_mat, self.x)  # [Bs, Ps, Hn]
                Aq = tf.nn.embedding_lookup(word_emb_mat, self.q)  # [Bs, Qs, Hn]
                self.tensor_dict['x'] = Ax
                self.tensor_dict['q'] = Aq
        # Build the LSTM cell with dropout
        cell = tf.contrib.rnn.BasicLSTMCell(self.Hn, state_is_tuple=True, forget_bias=config['model']['forget_bias'])
        dropout_cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob = tf.cond(self.is_training,lambda: config['model']['input_keep_prob'],lambda: 1.0))

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
            
            h = tf.concat([fw_h, bw_h], axis = 2)  # [Bs, Ps, 2Hn]
            self.tensor_dict['u'] = u
            self.tensor_dict['h'] = h

        with tf.variable_scope("main"):
			#AttentionLayer
            p0 =  attention_layer(self.x, self.q, Ax, Aq, x_len, q_len, self.Hn*2, self.Bs, h, u, scope='p0') #[Bs, Ps, 8Hn]
            # Hidden size multiplied by two because of bidirectional layer

            # [Bs, Ps, 8Hn]
            cell_after_att = tf.contrib.rnn.BasicLSTMCell(self.Hn, state_is_tuple=True, forget_bias=config['model']['forget_bias'])
            dropout_cell_after_att = tf.contrib.rnn.DropoutWrapper(cell_after_att, input_keep_prob = tf.cond(self.is_training,lambda: config['model']['input_keep_prob'],lambda: 1.0))
            (fw_g0, bw_g0), _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=dropout_cell_after_att,
                                                                cell_bw=dropout_cell_after_att,
                                                                inputs=p0,
                                                                sequence_length=x_len,
                                                                dtype=tf.float32,
                                                                scope='g0')
            g0 = tf.concat([fw_g0, bw_g0], axis = 2)

            # [Bs, Ps, 8Hn]
            cell_after_att_2 = tf.contrib.rnn.BasicLSTMCell(self.Hn, state_is_tuple=True, forget_bias=config['model']['forget_bias'])
            dropout_cell_after_att_2 = tf.contrib.rnn.DropoutWrapper(cell_after_att_2, input_keep_prob = tf.cond(self.is_training,lambda: config['model']['input_keep_prob'],lambda: 1.0))
            (fw_g1, bw_g1), _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=dropout_cell_after_att_2,
                                                                cell_bw=dropout_cell_after_att_2,
                                                                inputs=g0,
                                                                sequence_length=x_len,
                                                                dtype='float',
                                                                scope='g1')

            g1 = tf.concat([fw_g1, bw_g1, p0], axis = 2)

            w_y1 = tf.get_variable('w_y1', shape = [10*self.Hn,1], dtype = tf.float32)
            logits_y1 = tf.reshape(
                tf.matmul(
                    tf.concat(tf.unstack(value=g1, axis=0), axis=0),
                    w_y1),
            [self.Bs, -1]) + tf.multiply(tf.cast(1 - self.x_mask,tf.float32), VERY_LOW_NUMBER) #mask
            smax = tf.nn.softmax(logits_y1, 1)
            a1i = tf.matmul(tf.expand_dims(smax, 1),
				g1) #softsel

            a1i = tf.tile(a1i,
                          [1, self.Ps, 1])

            # [Bs, Sn, Ss, 2Hn]
            cell_y2 = tf.contrib.rnn.BasicLSTMCell(self.Hn, state_is_tuple=True, forget_bias=config['model']['forget_bias'])
            dropout_cell_y2 = tf.contrib.rnn.DropoutWrapper(cell_y2, input_keep_prob = tf.cond(self.is_training,lambda: config['model']['input_keep_prob'],lambda: 1.0))
            (fw_g2, bw_g2), _ = tf.nn.bidirectional_dynamic_rnn(cell_fw = dropout_cell_y2,
                                                                cell_bw=dropout_cell_y2,
                                                                inputs = tf.concat([p0, g1,a1i, tf.multiply(g1, a1i)], axis = 2),
                                                                sequence_length=x_len,
                                                                dtype='float',
                                                                scope='g2')
            g2 = tf.concat([fw_g2, bw_g2], axis = 2)
            # TODO: Rewrite the get_logits function
            w_y2 = tf.get_variable('w_y2', shape = [2*self.Hn,1], dtype = tf.float32)
            logits_y2 = tf.reshape(
                tf.matmul(
                    tf.concat(tf.unstack(value=g2,axis=0),axis=0),
                    w_y2),
            [self.Bs,-1]) + tf.multiply(tf.cast(1-self.x_mask, tf.float32), VERY_LOW_NUMBER) #mask

            yp = smax # [Bs, Ps]
            yp2 = tf.nn.softmax(logits_y2) #[Bs,Ps]

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
            logits = self.logits_y1, labels = tf.cast(self.y, 'float')))
        # tf.add_to_collection('losses', ce_loss)
        ce_loss2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits = self.logits_y2, labels = tf.cast(self.y2, 'float')))
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
        x=[]
        q=[]
        y1=[]
        y2=[]
        def word2id (word): #to convert a word to its respective id
            if self.config['pre']['lower_word']:
                word=word.lower()
            if word in dataset['shared']['known_word2idx']:
                return dataset['shared']['known_word2idx'][word] + self.WVs #vocabulary size
            elif word in dataset['shared']['unk_word2idx']:
                return dataset['shared']['unk_word2idx'][word]
            else:
                return 1 #unknown word

        def padding(seq): #for padding a batch
            max_size=max([len(seq[i]) for i in  range(len(seq))])
            new_seq=[np.concatenate([np.array(seq[i]), np.zeros([max_size-len(seq[i])])],axis=0)  for i in range(len(seq))]
            return np.int_(new_seq)

        # TODO: Add characters
        # convert every word to its respective id
        for i in batch_idxs:
            qi = list(map(
                word2id,
                dataset['data']['q'][i]))
            rxi = dataset['data']['*x'][i]
            yi = dataset['data']['y'][i]
            xi = list(map(word2id,
                dataset['shared']['x'][rxi[0]][rxi[1]]))
            q.append(qi)
            x.append(xi)
            y1.append([y[0] for y in yi]) # Get all the first indices in the sequence
            y2.append([y[1]-1 for y in yi]) # Get all the second indices... and correct for -1
        
        self.answer=[y1,y2]
        #padding
        q = padding(q)
        x = padding(x)
        y1_new=np.zeros([self.Bs, len(next(iter(x)))], dtype = np.bool)
        y2_new=np.zeros([self.Bs, len(next(iter(x)))], dtype = np.bool)
        for i in range(self.Bs):
            y1_new[i][y1[i]]=True
            y2_new[i][y2[i]]=True


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
    mask_matrix=tf.sign(tf.matmul(tf.expand_dims(x, -1), tf.expand_dims(q, -1), transpose_b=True))
    with tf.variable_scope(scope):
        vec1_att = tf.get_variable("att1",dtype='float', shape=[hidden_size, 1]) # [Hs, 1]
        vec2_att = tf.get_variable("att2",dtype='float', shape=[hidden_size, 1]) # [Hs, 1]
        vec3_att = tf.get_variable("att3",dtype='float', shape=[hidden_size]) # [Hs]

    # shaping all the vectors into a matrix to compute attention values in one matrix multiplication
    shaped_h = tf.concat(
        tf.unstack(
            value=h,
            axis=0),
        axis=0) # [Hs, Ps * Bs]

    shaped_u = tf.concat(
        tf.unstack(
            value=u,
            axis=0),
        axis=0) # [Hs, Qs * Bs]

    # Computation of vec1 * h + vec2 * u
    att_1_Product = tf.reshape(
        tf.matmul(
            shaped_h,
            vec1_att),
        shape=[batch_size, 1, -1]) # [Bs, 1, Ps]

    att_2_Product = tf.reshape(
        tf.matmul(
            shaped_u,
            vec2_att),
        shape=[batch_size, -1, 1]) # [Bs, Qs, 1]

    att_1_Product = tf.tile(
        att_1_Product,
        [1, tf.reduce_max(q_len), 1]) # [Bs, Qs, Ps]

    att_2_Product = tf.tile(
        att_2_Product,
        [1, 1, tf.reduce_max(x_len)]) # [Bs, Qs, Ps]

    #  of (vec3 * h)  . u

    h_vectorized = tf.multiply(h, vec3_att) # vect 3 * h
    att_3_Product = tf.matmul(u, h_vectorized, transpose_b=True) # ((vec 3 * h) . u [Bs, Qs, Ps]
    att_final=tf.transpose(
        att_1_Product + att_2_Product + att_3_Product,
        perm=[0, 2, 1]) # [Bs, Ps, Qs]

    att_final_masked = att_final + tf.multiply(tf.cast(1 - mask_matrix, tf.float32), -3e15) # masking

    # paragraph to question attention
    p2q = tf.multiply(
        tf.nn.softmax(logits=att_final_masked),
        tf.cast(mask_matrix,'float')
    ) # computing logits, taking into account mask

    U_a = tf.matmul(p2q, u)

    q2p = tf.nn.softmax(
        tf.reduce_max(att_final_masked, axis=-1))

    H_a = tf.tile(
	    tf.matmul(
            tf.expand_dims(q2p, 1),
            h),
	    [1, tf.reduce_max(x_len), 1]
    )

    G = tf.concat([h, U_a, tf.multiply(h, U_a), tf.multiply(h, H_a)], axis=-1)
    return G

def EM_and_F1(answer,answer_est):
    EM=[]
    F1=[]
    y1_est,y2_est = answer_est
    y1,y2 = answer 
    for i in range(len(y1_est)):
        EM_i = []
        F1_i = []
        for j in range(len(y1[i])):
            EM_i.append(1.0 if y1[i][j]==y1_est[i] and y2[i][j]==y2_est[i] else 0.0)
            TT=max([min([y2[i][j]+1,y2_est[i]+1])-max([y1[i][j],y1_est[i]]),0])
            FT=y2_est[i]+1-y1_est[i]-TT
            FF=y2[i][j]+1-y1[i][j]-TT
            a=TT/(TT+FT)
            b=TT/(TT+FF)
            F1_i.append(2/(1/a+1/b) if a!=0 and b!=0 else 0)
        EM.append(max(EM_i))
        F1.append(max(F1_i))
    return [sum(EM)/len(EM), sum(F1)/len(F1)]
