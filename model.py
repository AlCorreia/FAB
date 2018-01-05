import math
import numpy as np
import os
import pandas as pd
import random
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
from tqdm import tqdm
from random import randint
from utils import plot, send_mail, EM_and_F1, F1, maxSubArraySum
from evaluate_dev import evaluate_dev
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
        self.weights =[0,-1.32237,-4.66205,-8.53077,-11.484,-12.9115,-13.2051,-13.2673,-13.7885,-14.8277,-15.9396,-16.6519,-16.8666,-16.8842,-17.0945,-17.6377,-18.3228,-18.8376,-19.0372,-19.0528,-19.1493,-19.4805,-19.965,-20.3816,-20.5785,-20.6036,-20.6465,-20.8586,-21.221,-21.574,-21.771,-21.8099,-21.8262,-21.9625,-22.2395,-22.5445,-22.7415,-22.7957,-22.8004,-22.8855,-23.0979,-23.3628,-23.5582,-23.6277,-23.63,-23.6798,-23.841,-24.0705,-24.2621,-24.3461,-24.3521,-24.3778,-24.4975,-24.6945,-24.8801,-24.977,-24.9905,-25.0007,-25.0865,-25.2534,-25.4305,-25.5382,-25.5619,-25.5633,-25.6216,-25.7603,-25.9267,-26.0428,-26.0781,-26.076,-26.1125,-26.2249,-26.3786,-26.5006,-26.5479,-26.5467,-26.5664,-26.6546,-26.7939,-26.919,-26.9782,-26.9813,-26.9887,-27.0551,-27.1786,-27.304,-27.3743,-27.3842,-27.3836,-27.4306,-27.5376,-27.6605,-27.7406,-27.7592,-27.7544,-27.7847,-27.8747,-27.9927,-28.0807,-28.1092,-28.1036,-28.1201,-28.1932,-28.3038,-28.3978,-28.4367,-28.4332,-28.4389,-28.4955,-28.5967,-28.6944,-28.7437,-28.745,-28.7427,-28.7838,-28.8741,-28.973,-29.0321,-29.0401,-29.0329,-29.0599,-29.1379,-29.2356,-29.3035,-29.3197,-29.3105,-29.3251,-29.3899,-29.4841,-29.5593,-29.5848,-29.5763,-29.5805,-29.6319,-29.7204,-29.801,-29.8363,-29.8309,-29.827,-29.865,-29.9458,-30.0299,-30.0749,-30.0747,-30.0652,-30.0904,-30.1618,-30.2472,-30.3014,-30.3083,-30.2956,-30.309,-30.3696,-30.454,-30.5165,-30.5318,-30.5184,-30.5215,-30.5704,-30.6516,-30.721,-30.7456,-30.734,-30.7284,-30.7652,-30.8409,-30.9155,-30.9499,-30.9424,-30.9302,-30.9548,-31.0231,-31.1009,-31.1451,-31.1436,-31.127,-31.1399,-31.1991,-31.278,-31.3314,-31.3377,-31.3191,-31.3211,-31.3697,-31.4474,-31.5091,-31.5246,-31.5065,-31.4989,-31.536,-31.6101,-31.6788,-31.7042,-31.689,-31.6735,-31.6985,-31.7668,-31.8407,-31.8767,-31.8666,-31.8453,-31.858,-31.9184,-31.9953,-32.0419,-32.0391,-32.0142,-32.0151,-32.0656,-32.1433,-32.1999,-32.2063,-32.1803,-32.1701,-32.2093,-32.2853,-32.3508,-32.3679,-32.3435,-32.3236,-32.3501,-32.4218,-32.4949,-32.5237,-32.5036,-32.4757,-32.4889,-32.5536,-32.6323,-32.6734,-32.6602,-32.6267,-32.6261,-32.6814,-32.7633,-32.8168,-32.8131,-32.7764,-32.7624,-32.806,-32.8884,-32.9538,-32.9618,-32.925,-32.8984,-32.9281,-33.0079,-33.0843,-33.106,-33.0723,-33.0344,-33.0485,-33.1225,-33.2081,-33.2451,-33.2179,-33.1709,-33.168,-33.2327,-33.3251,-33.3786,-33.3615,-33.3081,-33.2875,-33.3393,-33.4356,-33.5059,-33.5026,-33.4463,-33.4078,-33.443,-33.5394,-33.6265,-33.6407,-33.5855,-33.5298,-33.5447,-33.6368,-33.7397,-33.775,-33.7259,-33.6544,-33.6457,-33.728,-33.8446,-33.9045,-33.8673,-33.7829,-33.7472,-33.8134,-33.9401,-34.0276,-34.0095,-33.9165,-33.8512,-33.8936,-34.0252,-34.1426,-34.1517,-34.0568,-33.9602,-33.97,-34.0984,-34.2465,-34.2924,-34.2052,-34.0777,-34.0449,-34.1585,-34.3352,-34.4284,-34.3629,-34.2089,-34.1227,-34.2049,-34.4033,-34.5541,-34.5297,-34.3598,-34.2112,-34.2398,-34.4443,-34.6597,-34.7016,-34.5369,-34.3226,-34.2701,-34.4529,-34.7306,-34.8674,-34.7439,-34.4739,-34.3123,-34.4291,-34.7486,-35.0049,-34.9749,-34.6835,-34.3955,-34.3876,-34.6979,-35.0786,-35.2042,-34.9608,-34.5604,-34.3675,-34.5807,-35.0465,-35.3776,-35.2864,-34.8445,-34.4376,-34.44,-34.884,-35.4162,-35.5892,-35.2484,-34.679,-34.3767,-34.6271,-35.2512,-35.7425,-35.6891,-35.1288,-34.5313,-34.4122,-34.8993,-35.6147,-35.9805,-35.6964,-34.9949,-34.457,-34.5381,-35.193,-35.9091,-36.121,-35.6673,-34.9197,-34.4828,-34.711,-35.4321,-36.0907,-36.1896,-35.6741,-34.9606,-34.6037,-34.8709,-35.5537,-36.145,-36.2283,-35.7757,-35.1446,-34.7981,-34.9723,-35.5294,-36.0801,-36.2651,-35.9899,-35.4654,-35.0502]

        self.Tensors_out = {}
        self.config = config
        # Define the directory where the results will be saved
        # TODO: Define a better convention for directories names
        self.directory = config['directories']['target_dir']
        self.dir_plots = config['directories']['plots']
        # Define global step and learning rate decay
        self.global_step = tf.Variable(0, trainable=False)
        # Define a placeholder for the dropout
        self.keep_prob_attention_pre_softmax = tf.placeholder_with_default(1.0, shape=(), name='dropout_attention_pre_softmax')
        self.keep_prob_attention_post_softmax = tf.placeholder_with_default(1.0, shape=(), name='dropout_attention_post_softmax')
        self.keep_prob_encoder = tf.placeholder_with_default(1.0, shape=(), name='dropout_encoder')
        self.keep_prob_attention = tf.placeholder_with_default(1.0, shape=(), name='dropout_attention')
        self.keep_prob_concat = tf.placeholder_with_default(1.0, shape=(), name='dropout_concat')
        self.keep_prob_Relu = tf.placeholder_with_default(1.0, shape=(), name='dropout_Relu')
        self.keep_prob_FF = tf.placeholder_with_default(1.0, shape=(), name='dropout_FF')
        self.keep_prob_selector = tf.placeholder_with_default(1.0, shape=(), name='dropout_selector')
        self.keep_prob_char_pre = tf.placeholder_with_default(1.0, shape=(), name='dropout_char_pre')
        self.keep_prob_char_post = tf.placeholder_with_default(1.0, shape=(), name='dropout_char_post')
        self.keep_prob_word_passage = tf.placeholder_with_default(1.0, shape=(), name='dropout_word_passage')
        self.keep_prob_last_x = tf.placeholder_with_default(1.0, shape=(), name='dropout_last_layer_passage')

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
        self.WEs = int(config['glove']['vec_size'])  # Assumed to be equal to the GloVe vector size. IT IS UPDATED, if char_embedding is used!!!!
        self.WEOs = config['model']['word_emb_size_after_scaling']
        self.WEAs = config['model']['attention_emb_size']  # Word-embedding attention size
        self.WEPs = config['model']['process_emb_size']  # Word-embedding attention size for attention and feed-forward sublayers
        self.Hn = config['model']['n_hidden']  # Number of hidden units in the LSTM cell

        self.x_comp_size = [self.WEAs, self.WEPs, self.WEAs, self.FFHs, self.MHs, self.WEAs] #size of x attention model/x processing size/size of q attention mode/multi-head size/X4 size in attention
        self.q_reduction = config['model']['q_variables_reduction']
        q_MHs = int(config['model']['q_multi_head_size'])
        q_WEPs = int(np.ceil(self.WEPs*self.q_reduction/q_MHs)*q_MHs)
        q_FFHs = int(self.FFHs*self.q_reduction)
        self.q_comp_size = [self.WEAs, q_WEPs, self.WEAs, q_FFHs, q_MHs, self.WEAs]#size of q attention model/q processing size/size of x attention model/ size o Feedforward number of heads/ X4 size in attention
        # Define placeholders
        # TODO: Include characters
        self.is_training = tf.placeholder(tf.bool, name='is_training')
        self.x = tf.placeholder('int32', [self.Bs, None], name='x')  # number of batches and number of words
        self.q = tf.placeholder('int32', [self.Bs, None], name='q')
        self.encoder_pos = tf.placeholder('float32', [self.Bs, None], name='encoder_pos')
        self.y = tf.placeholder('float32', [self.Bs, None], name='y')
        if self.config['model']['sigmoid_loss']:
            self.prob_out = []
        else:
            self.y2 = tf.placeholder('float32', [self.Bs, None], name='y2')
        self.mask_sentence = tf.placeholder('float32', [self.Bs, None, None], name='y2')
        if self.config['model']['sentence_skip_steps']>0:
            self.sentence_skip = True
        else:
            self.sentence_skip = False
        # Add another placeholder if a second loss is used

        self.new_emb_mat = tf.placeholder(tf.float32,
                                          [None, self.WEs],
                                          name='new_emb_mat')

        # Masks
        self.x_mask = tf.sign(self.x)
        self.x_without_unk_mask = tf.sign(tf.nn.relu(self.x-1))
        self.q_mask = tf.sign(self.q)
        self.q_without_unk_mask = tf.sign(tf.nn.relu(self.q-1))

        self.max_size_x = tf.shape(self.x)
        self.max_size_q = tf.shape(self.q)

        if self.config['model']['sigmoid_loss']:
            self.y_mask_1 = tf.cast(tf.greater(self.y,0), tf.int32)
            self.y_mask = self.x_mask - self.y_mask_1
            self.y_abs = tf.abs(self.y)

        if config['model']['char_embedding']:  # If there is char embedding
            self.COs = config['model']['char_out_size']  # Char output size
            self.CEs = config['model']['char_embedding_size']  # Char embedding size
            self.CVs = config['model']['char_vocabulary_size']
            self.xc = tf.placeholder('int32', [self.Bs, None], name='xc')  # Char level of x
            self.qc = tf.placeholder('int32', [self.Bs, None], name='qc')  # Char-level of q
            self.short_words_char = tf.placeholder('int32', [None, None], name='short_words_char')
            self.long_words_char = tf.placeholder('int32', [None, None], name='long_words_char')
            self.xc_size = tf.shape(self.xc)
            self.qc_size = tf.shape(self.qc)
            self.xc_mask = tf.sign(self.xc)
            self.qc_mask = tf.sign(self.qc)
            self.new_char_emb_mat = tf.placeholder(tf.float32,
                                                  [None, self.CEs],
                                                  name='new_emb_mat')
        else:
            self.COs = 0 #No char embedding
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
        if self.config['train']['xavier_initialization']:
            self.initializer = tf.contrib.layers.xavier_initializer(
                                                            uniform=False,
                                                            seed=None,
                                                            dtype=tf.float32)
        else:
            self.initializer = tf.contrib.layers.xavier_initializer(
                                                            uniform=True,
                                                            seed=None,
                                                            dtype=tf.float32)

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


        self.layer_red_vars = []
        self.other_vars= []
        #get layer_reduction variables
        for i in tf.trainable_variables():
            if 'one_layer_reduction' in i.name: #if it is a layer_reduction variable
                self.layer_red_vars.append(i)
            else:
                self.other_vars.append(i)

        # Define optimizer and train step
        if config['train']['type'] == "Adadelta":
            self.learning_rate = tf.train.exponential_decay(
                learning_rate=config['train']['Adadelta']['learning_rate'],
                global_step=self.global_step,
                decay_steps=config['train']['Adadelta']['decay_steps'],
                decay_rate=config['train']['Adadelta']['decay_rate'],
                staircase=True)
            self.optimizer = tf.train.AdadeltaOptimizer(
                learning_rate=self.learning_rate,
                rho=config['train']['Adadelta']['rho'])

        elif config['train']['type'] == "Adam":
		    # Decay_Rate is positive and therefore the - sign in this equation.
            self.learning_rate = config['train']['Adam']['learning_rate']*tf.multiply(
                                    tf.reduce_min([tf.pow(tf.cast(self.global_step, tf.float32), -self.config['train']['Adam']['decay_rate']), tf.multiply(tf.cast(self.global_step,tf.float32), tf.pow(tf.cast(config['train']['Adam']['WarmupSteps'], tf.float32),-config['train']['Adam']['decay_rate']-1.0))]),
                                    tf.pow(tf.cast(self.WEAs, tf.float32), -0.5))
            self.optimizer_1 = tf.train.AdamOptimizer(learning_rate=self.learning_rate if config['train']['Adam']['constant_LR'] else self.config['train']['Adam']['learning_rate'], beta1 = config['train']['Adam']['beta1'], beta2=config['train']['Adam']['beta2'], epsilon = config['train']['Adam']['epsilon'])
            reduction_factor = config['train']['Adam']['learning_rate_layer_reduction_factor']
            self.optimizer_2 = tf.train.AdamOptimizer(learning_rate=self.learning_rate/reduction_factor if config['train']['Adam']['constant_LR'] else self.config['train']['Adam']['learning_rate']/reduction_factor, beta1 = config['train']['Adam']['beta1'], beta2=config['train']['Adam']['beta2'], epsilon = config['train']['Adam']['epsilon'])

        elif config['train']['type'] == "Adagrad":
            self.optimizer = tf.train.AdagradOptimizer(
                learning_rate=config['train']['Adagrad']['learning_rate'],
                initial_accumulator_value=config['train']['Adagrad']['initial_accumulator_value'])

        elif config['train']['type'] == "RMS":
            self.optimizer = tf.train.RMSPropOptimizer(
                learning_rate=config['train']['RMS']['learning_rate'],
                decay=config['train']['RMS']['decay'],
                momentum=config['train']['RMS']['momentum']
            )

        # not sure if tf.contrib.layers.optimize_loss better than self.optimizer
        # Using contrib.layers to automatically log the gradients
        # self.train_step = tf.contrib.layers.optimize_loss(
        #    self.loss, global_step=self.global_step, learning_rate=self.learning_rate, optimizer='Adam',
        #    summaries=["gradients"], name='TIBINO')

        grads = tf.gradients(self.loss, self.other_vars + self.layer_red_vars)
        grads1 = grads[:len(self.other_vars)]
        grads2 = grads[len(self.other_vars):]
        train_op1 = self.optimizer_1.apply_gradients(zip(grads1, self.other_vars), self.global_step, name='training1')
        train_op2 = self.optimizer_2.apply_gradients(zip(grads2, self.layer_red_vars), name='training2')
        self.train_step = tf.group(train_op1, train_op2)
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
        self.saver = tf.train.Saver(max_to_keep=100)  # not to delete previous checkpoints
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
        feed_dict['dropout_attention_pre_softmax:0'] = self.config['train']['dropout_attention_pre_softmax']
        feed_dict['dropout_attention_post_softmax:0'] = self.config['train']['dropout_attention_post_softmax']
        feed_dict['dropout_encoder:0'] = self.config['train']['dropout_encoder']
        feed_dict['dropout_attention:0'] = self.config['train']['dropout_attention']
        feed_dict['dropout_concat:0'] = self.config['train']['dropout_concat']
        feed_dict['dropout_Relu:0'] = self.config['train']['dropout_Relu']
        feed_dict['dropout_FF:0'] = self.config['train']['dropout_FF']
        feed_dict['dropout_selector:0'] = self.config['train']['dropout_selector']
        feed_dict['dropout_char_pre:0'] = self.config['train']['dropout_char_pre_conv']
        feed_dict['dropout_char_post:0'] = self.config['train']['dropout_char_post_conv']
        feed_dict['dropout_word_passage:0'] = self.config['train']['dropout_word_passage']
        feed_dict['dropout_last_layer_passage:0'] = self.config['train']['dropout_last_layer_passage']
        if self.sess.run(self.global_step) % self.config['train']['steps_to_save'] == 0:
            if self.config['model']['sigmoid_loss']:
                summary, _, loss_val, global_step, max_x, max_q, index_prob_out,tensors_out = self.sess.run([self.summary, self.train_step, self.loss, self.global_step, self.max_size_x, self.max_size_q, self.prob_out, self.Tensors_out],
                                       feed_dict=feed_dict)
                Start_Index, End_Index, _ = maxSubArraySum(index_prob_out)
            else:
                summary, _, loss_val, global_step, max_x, max_q, Start_Index, End_Index = self.sess.run([self.summary, self.train_step, self.loss, self.global_step, self.max_size_x, self.max_size_q, self.Start_Index, self.End_Index],
                                       feed_dict=feed_dict)
            # Write the results to Tensorboard
            EM, F1, _, _, _ = EM_and_F1(self.answer, [Start_Index, End_Index])
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
            if self.config['model']['sigmoid_loss']:
                index_prob_out, _ = self.sess.run([self.prob_out, self.train_step], feed_dict=feed_dict)
                Start_Index, End_Index, _ = maxSubArraySum(index_prob_out)
            else:
                Start_Index, End_Index, _ = self.sess.run([self.Start_Index, self.End_Index, self.train_step], feed_dict=feed_dict)
            EM, F1, _, _, _ = EM_and_F1(self.answer, [Start_Index, End_Index])
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
        if self.config['model']['sigmoid_loss']:
            summary, max_x, max_q, global_step, Tensors_out, index_prob_out = self.sess.run([self.summary, self.max_size_x, self.max_size_q, self.global_step, self.Tensors_out, self.prob_out], feed_dict=feed_dict)
            Start_Index, End_Index, _ = maxSubArraySum(index_prob_out)
        else:
            summary, max_x, max_q, Start_Index, End_Index, global_step, Tensors_out = self.sess.run([self.summary, self.max_size_x, self.max_size_q, self.Start_Index, self.End_Index, self.global_step, self.Tensors_out], feed_dict=feed_dict)
        # Write the results to Tensorboard
        EM_dev, F1_dev, y1_correct, y2_correct, y2_greater_y1_correct = EM_and_F1(self.answer, [Start_Index, End_Index])
        self.EM_dev.append(EM_dev)
        self.F1_dev.append(F1_dev)
        self.y1_correct_dev.append(y1_correct)
        self.y2_correct_dev.append(y2_correct)
        self.y2_greater_y1_correct.append(y2_greater_y1_correct)
        self.dev_writer.add_summary(summary, global_step=global_step)
        return Start_Index, End_Index

    def _load(self):  # To load a checkpoint
        # TODO: Add structure to save/load different checkpoints.
        self.saver.restore(self.sess, self.config['directories']['dir'] + 'model.ckpt')

    def evaluate_all_dev(self, valid_idxs, data_dev):
        """ Compute F1 and EM for the dev dataset

            Parameters
            ----------
            batch_idxs : list of the idxs of each example
            dataset : the correspondent json file

        """
        # Combine the input dictionaries for all the features models
        feed_dict = self.get_feed_dict(valid_idxs, is_training=False, dataset=data_dev)
        if self.config['model']['sigmoid_loss']:
            index_prob_out, y_prob_out = self.sess.run([self.prob_out, self.yp], feed_dict=feed_dict)
            Start_Index, End_Index, prob = maxSubArraySum(index_prob_out)
        else:
            Start_Index, End_Index, prob = self.sess.run([self.Start_Index, self.End_Index, tf.reduce_max(self.yp,1)*tf.reduce_max(self.yp2,1)], feed_dict=feed_dict)
        # Write the results to Tensorboard
        return Start_Index, End_Index, prob

    def _load(self):  # To load a checkpoint
        # TODO: Add structure to save/load different checkpoints.
        self.saver.restore(self.sess, self.directory + 'model.ckpt')

    def _char2word_embedding(self, Ac, mask):
        Ac_size = tf.shape(Ac)
        mask = tf.expand_dims(tf.cast(mask,tf.float32),2)
        Ac = tf.multiply(Ac,mask) #To zero all embeddings
        Ac=tf.expand_dims(Ac,1)
        if self.config['train']['dropout_char_pre_conv']<1.0:
            Ac = tf.nn.dropout(Ac, keep_prob=self.keep_prob_char_pre)
        A_word_convolution = tf.layers.conv2d(inputs=Ac,
                                              filters=self.COs,
                                              kernel_size=[1,self.config['model']['char_convolution_size']],
                                              strides=[1, 1],
                                              padding='valid',
                                              activation=tf.tanh)
        mask = tf.slice(mask, [0,0,0],[Ac_size[0],Ac_size[1]-self.config['model']['char_convolution_size']+1, 1]) #Mask must be updated because of valid
        A_word_convolution_masked = tf.squeeze(A_word_convolution,1)+tf.cast(1-mask,tf.float32)*(VERY_LOW_NUMBER)#To ignore padding vectors in reduce_max
        char_embedded_word = tf.reduce_max(A_word_convolution_masked, axis=1)  # Reduce all info to a vector
        if self.config['train']['dropout_char_post_conv']<1.0:
            char_embedded_word = tf.nn.dropout(char_embedded_word, keep_prob=self.keep_prob_char_post)
        return char_embedded_word

    def _embed_scaling(self, X, inp_size, out_size, second=False):
        length_X = X.get_shape()[1]  # number of words in the passage
        # If the word2vec vector is scaled by a matrix
        # Scaling word2vec matrices before adding encoder
        with tf.variable_scope('Scaling', reuse=second) as scope:
            if self.config['model_options']['word2vec_scaling']=='scalar':
                weigths = tf.get_variable(
                            'gain',
                            initializer=1.0)
                if self.config['model_options']['use_bias']:
                    bias = tf.get_variable('bias',
                                           shape=[inp_size],
                                           initializer=tf.zeros_initializer())
                    X = tf.multiply(weigths, X) + bias
                else:
                    X = tf.multiply(weigths, X)
            elif self.config['model_options']['word2vec_scaling']=='vector':
                weigths = tf.get_variable(
                            'vector', shape=[inp_size],
                            initializer=self.initializer)
                if self.config['model_options']['use_bias']:
                    bias = tf.get_variable('bias',
                                           shape=[inp_size],
                                           initializer=tf.zeros_initializer())
                    X = tf.multiply(weigths, X) + bias
                else:
                    X = tf.multiply(weigths, X)
            elif self.config['model_options']['word2vec_scaling']=='matrix':
                with tf.variable_scope('conv2d', reuse=second):
                    # If the scaling matrix was previously trained
                    # In order to be orthonormal
                    if self.config['model_options']['word2vec_orthonormal_scaling']:
                        weights_init = np.random.random((1, 1, inp_size, inp_size)).astype(np.float32)  # might not work properly if WEs different from WEAs.
                        _, _, U = np.linalg.svd(weights_init, full_matrices=False)
                        weigths = tf.get_variable(
                                    'kernel',
                                    initializer=U[...,0:out_size])
                    else:
                        weigths = tf.get_variable(
                                    'kernel',
                                    shape=[1, 1, inp_size, out_size],
                                    initializer=self.initializer)
                    if self.config['model_options']['use_bias']:
                        bias = tf.get_variable('bias',
                                               shape=[out_size],
                                               initializer=tf.zeros_initializer())
                X = tf.expand_dims(X, 2)
                X.set_shape([self.Bs, length_X, 1, inp_size])
                X = tf.squeeze(tf.layers.conv2d(X,
                                                filters=out_size,
                                                kernel_size=1,
                                                strides=1,
                                                use_bias=self.config['model_options']['use_bias'],
                                                reuse=True,
                                                name="conv2d"))  # XW+B
            elif self.config['model_options']['word2vec_scaling']=='nonlinear':
                X = tf.expand_dims(X, 2)
                X.set_shape([self.Bs, length_X, 1, inp_size])
                X = tf.layers.conv2d(X,
                                     filters=self.config['model']['FeedForward_Hidden_Size'],
                                     kernel_size=1,
                                     strides=1,
                                     use_bias=self.config['model_options']['use_bias'],
                                     reuse=second,
                                     activation=tf.nn.relu,
                                     name="conv2d_first")  # XW+B
                X = tf.squeeze(tf.layers.conv2d(X,
                                                filters=out_size,
                                                kernel_size=1,
                                                strides=1,
                                                use_bias=self.config['model_options']['use_bias'],
                                                reuse=second,
                                     name="conv2d_second"))  # XW+B
            else: print("\n WORD2VEC SCALING NOT SELECTED\n")
        return X

    def _context_network(self, X, second=False):
        with tf.variable_scope('Context_network', reuse=second):
            length_X = X.get_shape()[1]
            X = tf.expand_dims(X, 2)
            X.set_shape([self.Bs, length_X, 1, self.WEs])
            # Affine operation followed by Relu.
            # It is done by a convolution and it is the same as X*W+b
            affine_op = tf.layers.conv2d(X,
                                         filters=self.WEs,
                                         kernel_size=1,
                                         strides=1,
                                         use_bias=True,
                                         activation=tf.nn.relu,
                                         kernel_initializer=self.initializer,
                                         name='context_1',
                                         reuse=second)

            # Second affine oepration.
            # It is done by a convolution and it is the same as X*W+b
            affine_op2 = tf.squeeze(tf.layers.conv2d(affine_op,
                                                 filters=self.WEs,
                                                 kernel_size=1,
                                                 strides=1,
                                                 use_bias=True,
                                                 kernel_initializer=self.initializer,
                                                 name='context_2',
                                                 reuse=second))
            context = tf.reduce_mean(affine_op2, axis=1)
            context.set_shape([self.Bs, 1, 1, self.WEs])
            mask = tf.layers.conv2d(context,
                                    filters=self.WEs,
                                    kernel_size=1,
                                    strides=1,
                                    use_bias=True,
                                    activation=tf.nn.sigmoid,
                                    kernel_initializer=self.initializer,
                                    name='context_3',
                                    reuse=second)

            output = tf.multiply(X, mask)
        return output


    def _highway_network(self, X, num_layers, input_length, second=False):
        with tf.variable_scope('Highway_network', reuse=second):
            # X = tf.expand_dims(X, 2)
            X = tf.reshape(X,[self.Bs, -1,1,input_length])
            input_layer = X
            for i in range(num_layers):
                with tf.variable_scope('layer_'+str(i), reuse=second):
                    # Compute Lin1 and Lin2
                    Lin1_Lin2 = tf.layers.conv2d(input_layer,
                                    filters=input_length*2,
                                    kernel_size=1,
                                    strides=1,
                                    use_bias=True,
                                    reuse=second,
                                    kernel_initializer=self.initializer,
                                    name="linear")
                    Lin1, Lin2 = tf.split(
                                         Lin1_Lin2,
                                         num_or_size_splits=2,
                                         axis=3)
                    #sigmoid(lin1)
                    weight = tf.sigmoid(Lin1)
                    #sigmoid*y+(1-sigmoid)*tf.nn.relu(Lin2)
                    input_layer = input_layer*weight+(1.0-weight)*tf.nn.relu(Lin2)
            output_highway = tf.squeeze(input_layer)
        return output_highway

    def _encoder(self, X, Q, input_size, sentence_skip=False, reuse=False):
        # Compute the number of words in passage and question
        size_x = tf.shape(X)[-2]
        size_q = tf.shape(Q)[-2]
        low_frequency = self.config['model']['encoder_low_freq']
        high_frequency = self.config['model']['encoder_high_freq']
        exponents = tf.multiply(tf.log(high_frequency/low_frequency), tf.divide(tf.range((input_size)/2), (input_size)/2-1))
        # Create a row vector with range(0,n) = [0,1,2,n-1], where n is the greatest size between x and q.
        freq = tf.multiply(1/low_frequency, tf.exp(-exponents))
        if sentence_skip:
            pos_x = tf.expand_dims(self.encoder_pos, 2)
            pos_x = tf.tile(pos_x, [1,1,tf.cast(input_size/2,'int32')])
            pos_freq_product = tf.multiply
            pos_q = tf.cast(tf.expand_dims(tf.range(size_q), 1), tf.float32)
            axis_concat = 2
        else:
            pos_x = tf.cast(tf.expand_dims(tf.range(size_x), 1), tf.float32)
            pos_q = tf.cast(tf.expand_dims(tf.range(size_q), 1), tf.float32)
            pos_freq_product = tf.matmul
            axis_concat = 1
        pos_x = pos_x + self.config['model']['encoder_initial_step'] #To change initial step
        pos_q = pos_q + self.config['model']['encoder_initial_step'] #To change initial step
        # Create a vector with all the exponents
        # Power the base frequency by exponents
        if self.config['model']['encoder_learn_freq']:  # Encoder frequencies are trained
            freq_PG = tf.get_variable('wave_length', dtype=tf.float32, initializer=freq)
        else:  # Encoder frequencies are not trained
            freq_PG = freq

        freq_PG_scalar = tf.summary.scalar('wave_length', tf.reduce_mean(freq_PG))

        # Compute the encoder values
        if self.config['train']['moving_encoder_regularization']:
            #Generates a random phase for sin and cosin
            random_steps = freq_PG * tf.random_uniform(shape=[1],
                                                       minval=-self.config['model']['encoder_high_freq']*2*np.pi,
                                                       maxval=+self.config['model']['encoder_high_freq']*2*np.pi)
            #If it is not training, this random phase is not generated to make it deterministic
            encoder_angles_x = pos_freq_product(pos_x, freq_PG) + random_steps*tf.to_float(self.is_training)
            encoder_angles_q = pos_freq_product(pos_q, tf.expand_dims(freq_PG, 0)) + random_steps*tf.to_float(self.is_training)
        else:
            encoder_angles_x = pos_freq_product(pos_x, tf.expand_dims(freq_PG, 0))
            encoder_angles_q = pos_freq_product(pos_q, tf.expand_dims(freq_PG, 0))

        # Compute the encoder values
        encoder_sin_x = tf.sin(encoder_angles_x)
        encoder_cos_x = tf.cos(encoder_angles_x)

        # Concatenate both values
        encoder_x = tf.concat([encoder_sin_x, encoder_cos_x], axis=axis_concat)

        if self.config['model']['encoder_no_cross']:
            # If no cross attention between encoders is desired
            freq_q_sum = tf.add(
                                tf.multiply(self.config['model']['encoder_step_skip_size'],
                                            freq),
                                (np.pi/2))
            encoder_angles_q = tf.add(
                                        encoder_angles_q,
                                        freq_q_sum)
        encoder_sin_q = tf.sin(encoder_angles_q)
        encoder_cos_q = tf.cos(encoder_angles_q)
        encoder_q = tf.concat([encoder_sin_q,encoder_cos_q], axis=1)
        if self.config['model']['encoder_scaling']:
            with tf.variable_scope('Encoder', reuse=reuse):
                w = tf.get_variable('weight_encoder', shape =[1], dtype=tf.float32, initializer=tf.ones_initializer())
                encoder_x = w*encoder_x
                encoder_q = w*encoder_q

    # Encoding x and q
        x_encoded = tf.add(X, encoder_x)
        if self.config['model']['number_of_totens']>0:
            size_each_q = tf.cast(tf.expand_dims(tf.reduce_sum(self.q_mask, axis=1),1), tf.int32)  # Compute size of each question
            range_q = tf.expand_dims(tf.range(0, size_q, 1), 0)  # generate range sequence
            mask_encoder_q = tf.cast(tf.round(tf.cast(tf.greater(size_each_q-self.config['model']['number_of_totens'], range_q), tf.int32)), tf.float32)  # Compute new mask without totens
            encoder_q = tf.multiply(encoder_q, tf.expand_dims(mask_encoder_q,2))  # Compute masked encoder_q to not encode totens
        q_encoded = tf.add(Q, encoder_q)
        if self.config['train']['dropout_encoder']<1.0:  # This is done to save memory if dropout is not used
            x_encoded = tf.nn.dropout(x_encoded, keep_prob=self.keep_prob_encoder)
            q_encoded = tf.nn.dropout(q_encoded, keep_prob=self.keep_prob_encoder)
        return x_encoded, q_encoded


    def _attention_layer(self, X1, mask, X2=None, X3=None, X4=None, scope=None, comp_size=None, reuse=False, dropout=1.0, cross=False, sentence_masking=False):
        # Q = X1*WQ, K = X2*WK, V=X1*WV, X2 = X1 if X1 is None
        keep_prob_attention = tf.pow(self.keep_prob_attention,dropout)
        keep_prob_concat = tf.pow(self.keep_prob_concat,dropout)
        keep_prob_pre_softmax = self.keep_prob_attention_pre_softmax
        keep_prob_post_softmax = self.keep_prob_attention_post_softmax

        with tf.variable_scope(scope, reuse=reuse):
            length_X1 = X1.get_shape()[1]
            X1 = tf.expand_dims(X1, 2)
            X1.set_shape([self.Bs, length_X1, 1, comp_size[0]])
            if X2 is None:
                length_X2 = length_X1
                # (SELF ATTENTION)
                # If X2 is None Compute Q = X1*WQ, K = X1*WK, V=X1*WV
                QKV = tf.squeeze(tf.layers.conv2d(X1,
                                                  filters=comp_size[1]*3,
                                                  kernel_size=1,
                                                  strides=1,
                                                  kernel_initializer=self.initializer,
                                                  use_bias=self.config['model_options']['use_bias'],
                                                  reuse=reuse,
                                                  name='QKV_Comp'))
                Q, K, V = tf.split(
                    QKV,
                    num_or_size_splits=[comp_size[1], comp_size[1], comp_size[1]],
                    axis=2)
            elif X3 is None:
                # (CROSS ATTENTION)
                # If X2 is not none, compute Q = X1*WQ, K = X2*WK, V=X1*WV
                length_X2 = X2.get_shape()[1]
                KV = tf.squeeze(tf.layers.conv2d(X1,
                                                 filters=comp_size[1]*2,
                                                 kernel_size=1,
                                                 strides=1,
                                                 kernel_initializer=self.initializer,
                                                 use_bias=self.config['model_options']['use_bias'],
                                                 reuse=reuse,
                                                 name='KV_Comp'))
                K, V = tf.split(KV,
                                num_or_size_splits=[comp_size[1], comp_size[1]],
                                axis=2)
                X2 = tf.expand_dims(X2, 2)
                X2.set_shape([self.Bs, length_X2, 1, comp_size[2]])
                Q = tf.squeeze(tf.layers.conv2d(X2,
                                                filters=comp_size[1],
                                                kernel_size=1,
                                                strides=1,
                                                kernel_initializer=self.initializer,
                                                use_bias=self.config['model_options']['use_bias'],
                                                reuse=reuse,
                                                name='Q_Comp'))
                X2 = tf.squeeze(X2)
                X2.set_shape([self.Bs, length_X2, comp_size[2]])
            else:
                # (CROSS ATTENTION)
                # If X2 and X3 are not none, compute Q = X1*WQ, K = X2*WK, V=X3*WV
                #X2 Processing
                length_X2 = X2.get_shape()[1]
                X2 = tf.expand_dims(X2, 2)
                X2.set_shape([self.Bs, length_X2, 1, comp_size[2]])
                Q = tf.squeeze(tf.layers.conv2d(X2,
                                                filters=comp_size[1],
                                                kernel_size=1,
                                                strides=1,
                                                kernel_initializer=self.initializer,
                                                use_bias=self.config['model_options']['use_bias'],
                                                reuse=reuse,
                                                name='Q_Comp'))
                X2 = tf.squeeze(X2)
                X2.set_shape([self.Bs, length_X2, comp_size[2]])
                #X1 Processing
                K = tf.squeeze(tf.layers.conv2d(X1,
                                                filters=comp_size[1],
                                                kernel_size=1,
                                                strides=1,
                                                kernel_initializer=self.initializer,
                                                use_bias=self.config['model_options']['use_bias'],
                                                reuse=reuse,
                                                name='K_Comp'))
                # X3 Processing
                length_X3 = X3.get_shape()[1]
                X3 = tf.expand_dims(X3, 2)
                X3.set_shape([self.Bs, length_X3, 1, comp_size[2]])
                V = tf.squeeze(tf.layers.conv2d(X3,
                                                filters=comp_size[1],
                                                kernel_size=1,
                                                strides=1,
                                                kernel_initializer=self.initializer,
                                                use_bias=self.config['model_options']['use_bias'],
                                                reuse=reuse,
                                                name='V_Comp'))
                X3 = tf.squeeze(X3)
                X3.set_shape([self.Bs, length_X3, comp_size[2]])
            X1 = tf.squeeze(X1)
            X1.set_shape([self.Bs, length_X1, self.WEAs])

            if X4 is not None:
                # X3 Processing
                length_X4 = X4.get_shape()[1]
                X4 = tf.expand_dims(X4, 2)
                X4.set_shape([self.Bs, length_X4, 1, comp_size[5]])
                V_X4 = tf.squeeze(tf.layers.conv2d(X4,
                                                   filters=comp_size[5],
                                                   kernel_size=1,
                                                   strides=1,
                                                   kernel_initializer=self.initializer,
                                                   use_bias=self.config['model_options']['use_bias'],
                                                   reuse=reuse,
                                                   name='V4_Comp'))
                X4 = tf.squeeze(X4)
                X4.set_shape([self.Bs, length_X4, comp_size[2]])
                V_X4 = tf.split(V_X4, num_or_size_splits=comp_size[4], axis=2)

            # Split Q, K, V for multi-head attention
            Q = tf.split(Q, num_or_size_splits=comp_size[4], axis=2)
            K = tf.split(K, num_or_size_splits=comp_size[4], axis=2)
            V = tf.split(V, num_or_size_splits=comp_size[4], axis=2)

            if self.config['model']['max_out']:
                Q = tf.expand_dims(tf.reduce_max(Q, axis=0), axis=0)
                K = tf.expand_dims(tf.reduce_max(K, axis=0), axis=0)
                V = tf.expand_dims(tf.reduce_max(V, axis=0), axis=0)
                MHs = 1
                WEPs = int(comp_size[1]/comp_size[4])
            else:
                MHs = comp_size[4]
                WEPs = comp_size[1]
            # Compute transpose of K for multiplyting Q*K^T
            Scaling = tf.sqrt(tf.cast(comp_size[1],tf.float32)/tf.cast(comp_size[4], tf.float32))
            logits = tf.matmul(Q, tf.transpose(K, [0, 1, 3, 2]))

            if (self.config['model']['conv_attention']=="cross" and cross) or self.config['model']['conv_attention']=="all":
                att_kernel_size = list(map(int,self.config['model']['conv_attention_kernel_size'].split(',')))
                if cross:
                    att_kernel_size = att_kernel_size[-2:] #Last two if cross
                else:
                    att_kernel_size = att_kernel_size[:2] #First two if not cross
                logits = logits*mask  # masking logits
                logits = tf.transpose(logits, [1, 2, 3, 0])
                logits.set_shape([self.Bs, length_X2, length_X1, MHs])


                if self.config['model']['conv_attention_kernel'] == "per_channel":
                    # [filter_height, filter_width, in_channels, channel_multiplier]
                    kernel = tf.get_variable('kernel',
                                              shape=att_kernel_size+[MHs, 1],
                                              dtype=tf.float32,
                                              initializer=self.initializer)
                    conv_logits = tf.nn.depthwise_conv2d_native(logits,
                                                          filter=kernel,
                                                          strides=[1, 1, 1, 1],
                                                          padding="SAME",
                                                          name='Conv_att_Comp')
                elif (self.config['model']['conv_attention_kernel'] == "multi") and (MHs == 4):
                    conv_logits1 = tf.layers.conv2d(logits,
                                                    filters=1,
                                                    kernel_size=[5, 5],
                                                    strides=1,
                                                    kernel_initializer=self.initializer,
                                                    use_bias=False,
                                                    padding='same',
                                                    reuse=False,
                                                    name='Conv_att_Comp1')
                    conv_logits2 = tf.layers.conv2d(logits,
                                                    filters=1,
                                                    kernel_size=[11, 11],
                                                    strides=1,
                                                    kernel_initializer=self.initializer,
                                                    use_bias=False,
                                                    padding='same',
                                                    reuse=False,
                                                    name='Conv_att_Comp2')
                    conv_logits3 = tf.layers.conv2d(logits,
                                                    filters=1,
                                                    kernel_size=[5, 11],
                                                    strides=1,
                                                    kernel_initializer=self.initializer,
                                                    use_bias=False,
                                                    padding='same',
                                                    reuse=False,
                                                    name='Conv_att_Comp3')
                    conv_logits4 = tf.layers.conv2d(logits,
                                                    filters=1,
                                                    kernel_size=[11, 5],
                                                    strides=1,
                                                    kernel_initializer=self.initializer,
                                                    use_bias=False,
                                                    padding='same',
                                                    reuse=False,
                                                    name='Conv_att_Comp4')
                    conv_logits = tf.concat([conv_logits1, conv_logits2, conv_logits3, conv_logits4], axis=3)
                # Default to normal convolution
                elif (self.config['model']['conv_attention_kernel'] == "deep"):
                    hidden_logits = tf.layers.conv2d(logits,
                                                   filters=2*MHs,
                                                   kernel_size=att_kernel_size,
                                                   strides=1,
                                                   kernel_initializer=self.initializer,
                                                   use_bias=False,
                                                   padding='same',
                                                   reuse=False,
                                                   activation = tf.tanh,
                                                   name='Conv_att_Comp_1')
                    conv_logits = tf.layers.conv2d(hidden_logits,
                                                   filters=MHs,
                                                   kernel_size=att_kernel_size,
                                                   strides=1,
                                                   kernel_initializer=self.initializer,
                                                   use_bias=False,
                                                   padding='same',
                                                   reuse=False,
                                                   name='Conv_att_Comp_2')
                else:
                    conv_logits = tf.layers.conv2d(logits,
                                                   filters=MHs,
                                                   kernel_size=att_kernel_size,
                                                   strides=1,
                                                   kernel_initializer=self.initializer,
                                                   use_bias=False,
                                                   padding='same',
                                                   reuse=reuse,
                                                   name='Conv_att_Comp')

                logits = conv_logits
                logits = tf.transpose(logits, [3, 0, 1, 2])

            if (cross and self.config['model']['inverse_softmax_cross_att']): # Softmax direction is changed
                dimension = 2
            else:
                dimension = 3

        if sentence_masking and self.config['model']['sentence_masking']:
            sentence_masking_gains = tf.get_variable('sentence_masking_attention'+scope, shape=[comp_size[4], 1, 1, 1], initializer = tf.ones_initializer())
            logits = logits + tf.sigmoid(sentence_masking_gains)*self.mask_sentence
        with tf.variable_scope(scope, reuse=reuse):
            if (cross and self.config['model']['gated_cross_softmax']): # Softmax direction is changed
                no_attention = tf.get_variable('no_attention', shape=[1], initializer = tf.ones_initializer())
                logits = logits/Scaling
                max_logits = tf.expand_dims(tf.reduce_max(logits,3),3)
                exp_logits = tf.exp(logits-max_logits)*mask
                exp_no_att = tf.exp(no_attention-max_logits)
                sum_logits = tf.expand_dims(tf.reduce_sum(exp_logits,3),3) + exp_no_att
                softmax = exp_logits/(sum_logits+1e-6)
            else:

                # Sofmax in each head of the splitted Q and K softmax(Q*K^T):
                if self.config['train']['dropout_attention_pre_softmax'] < 1.0:
                    # This is done to save memory if dropout is not used
                    softmax = tf.nn.softmax(
                        tf.add(
                            tf.divide(logits, Scaling),
                            tf.multiply(1.0 - tf.nn.dropout(mask, keep_prob=keep_prob_pre_softmax), VERY_LOW_NUMBER)),
                        dim=dimension)
                else:
                    softmax = tf.nn.softmax(
                        tf.add(
                            tf.divide(logits, Scaling),
                            tf.multiply(1.0 - mask, VERY_LOW_NUMBER)),
                        dim=dimension)
            # Final mask is applied
            if self.config['train']['dropout_attention_post_softmax'] < 1.0:
                # To normalize softmax sum to 1
                softmax = 1/keep_prob_post_softmax*tf.nn.dropout(tf.multiply(mask, softmax), keep_prob_post_softmax)
            else:
                softmax = tf.multiply(mask, softmax)

            # Multihead attention
            # WV must be split into multi_head_size smaller matrices
            x_attention = tf.matmul(softmax, V)  # softmax(Q*K^T)*V
            # Concatenate everything together
            x_attention_concat = tf.concat(
                tf.unstack(x_attention,
                           axis=0,
                           num=MHs),
                axis=2)
            x_attention_concat.set_shape([self.Bs, length_X2, int(WEPs)])
            if self.config['train']['dropout_concat']<1.0:  # This is done to save memory if dropout is not used
                x_attention_concat = tf.nn.dropout(x_attention_concat, keep_prob=keep_prob_concat)
            # Compute softmax(Q*K^T)*V*WO
            x_final = tf.squeeze(
                tf.layers.conv2d(tf.expand_dims(x_attention_concat, 2),
                                 filters=comp_size[0],
                                 kernel_size=1,
                                 strides=1,
                                 name='Att_Comp'))

            # X4 processing
            if X4 is not None:
                X4_attention = tf.matmul(softmax, V_X4)  # softmax(Q*K^T)*V
            # Concatenate everything together
                X4_attention_concat = tf.concat(
                tf.unstack(X4_attention,
                           axis=0,
                           num=MHs),
                axis=2)
                X4_attention_concat.set_shape([self.Bs, length_X4, comp_size[5]])
            # Compute softmax(Q*K^T)*V*WO
                X4_final = tf.squeeze(
                    tf.layers.conv2d(tf.expand_dims(X4_attention_concat, 2),
                                     filters=comp_size[5],
                                     kernel_size=1,
                                     strides=1,
                                     reuse=reuse,
                                     name='Att_Comp_X4'))
                X4_final = tf.nn.dropout(
                    X4_final,
                    keep_prob=self.keep_prob_attention)


            # Add Dropout
            if self.config['train']['dropout_attention']<1.0:  # This is done to save memory if dropout is not used
                x_final = tf.nn.dropout(
                    x_final,
                    keep_prob=keep_prob_attention)
        if X4 is not None:
            return [x_final, X4_final]
        else:
            return x_final

    def _proc_cross_layer(self, X, scope, comp_size):
        with tf.variable_scope(scope):
            length_X = X.get_shape()[1]
            X = tf.expand_dims(X, 2)
            X.set_shape([self.Bs, length_X, 1, comp_size[2]])
            X_scaled = tf.squeeze(tf.layers.conv2d(X,
                                                 filters=self.config['model']['number_of_cross_attentions'],
                                                 kernel_size=1,
                                                 strides=1,
                                                 kernel_initializer=self.initializer,
                                                 use_bias=self.config['model_options']['use_bias'],
                                                 name='cross_prepare'))
            X = tf.squeeze(X)
            X.set_shape([self.Bs, length_X, self.WEAs])
            X_prod=tf.matmul(tf.nn.softmax(tf.transpose(X_scaled,[0,2,1]),-1),X)
        return X_prod

    def _layer_normalization(self, x, scope=None, shape=None, reuse=False):
        if shape is None:
            shape = self.WEAs
        with tf.variable_scope(scope, reuse=reuse):
            # Compute variance and means
            mean_val = tf.reduce_mean(x, axis=[-1])
            mean_val = tf.expand_dims(mean_val,axis=-1)
            variance = 1e-8 + tf.reduce_mean(tf.square(x-mean_val),axis=[-1])  # 1e-8 to avoid NaN
            std_dev = tf.sqrt(variance)
            std_dev = tf.expand_dims(std_dev,axis=-1)
            normalized_x = tf.divide((x-mean_val),std_dev)
        # In Google Attention Model original code, there are these weights.
        # By now, they were turned off in FAB.
            if self.config['model_options']['norm_layer']:
                W_Scale = tf.get_variable('weight',
                                          shape=[shape],
                                          dtype=tf.float32,
                                          initializer=tf.ones_initializer())
                b_Scale = tf.get_variable('bias',
                                          shape=[shape],
                                          dtype=tf.float32,
                                          initializer=tf.zeros_initializer())
                normalized_x = normalized_x * W_Scale + b_Scale
        return normalized_x

    def _FeedForward_NN(self, X, scope=None, comp_size=None, dropout=1.0, reuse=False):
        keep_prob_relu_func = tf.pow(self.keep_prob_Relu,dropout)  #If dropout greater than 2.0, than a original dropout of 0.9 is reduced to 0.81
        keep_prob_FF_func = tf.pow(self.keep_prob_FF, dropout)
        # Starting variables
        with tf.variable_scope(scope, reuse=reuse):
            length_X = X.get_shape()[1]
            X = tf.expand_dims(X, 2)
            X.set_shape([self.Bs, length_X, 1, comp_size[0]])
            # Affine operation followed by Relu.
            # It is done by a convolution and it is the same as X*W+b
            affine_op = tf.layers.conv2d(X,
                                         filters=comp_size[3],
                                         kernel_size=1,
                                         strides=1,
                                         use_bias=True,
                                         activation=tf.nn.relu,
                                         kernel_initializer=self.initializer,
                                         name='affine_op_1',
                                         reuse=reuse)
            if self.config['train']['dropout_Relu'] < 1.0:  # This is done to save memory if dropout is not used
                affine_op = tf.nn.dropout(affine_op, keep_prob=keep_prob_relu_func)
            X = tf.squeeze(X)
            X.set_shape([self.Bs, length_X, comp_size[0]])
            # Second affine oepration.
            # It is done by a convolution and it is the same as X*W+b
            output = tf.squeeze(tf.layers.conv2d(affine_op,
                                                 filters=comp_size[0],
                                                 kernel_size=1,
                                                 strides=1,
                                                 use_bias=True,
                                                 kernel_initializer=self.initializer,
                                                 name='affine_op_2',
                                                 reuse=reuse))
            # Apply Dropout
            if self.config['train']['dropout_FF']<1.0: #This is done to save memory if dropout is not used
                output = tf.nn.dropout(
                    output,
                    keep_prob=keep_prob_FF_func)
        return output

    def _one_layer_reduction(self, Q, X, mask, scope, out_size, Q_Cs, X_Cs, switch=False):
        # Defining masks and scopes
        if switch:
            X1 = X
            X1_comp_size = X_Cs
            X2 = Q
            X2_comp_size = Q_Cs
            X1X1, X2X2, X2X1, X1X2, X2_v = 'xx', 'qq', 'qx', 'xq', 'q'
            X1_masking = True
            X2_masking = False
        else:
            X1 = Q
            X1_comp_size = Q_Cs
            X2 = X
            X2_comp_size = X_Cs
            X1X1, X2X2, X2X1, X1X2, X2_v = 'qq', 'xx', 'xq', 'qx', 'x'
            X1_masking = False
            X2_masking = True
        with tf.variable_scope(scope):
            if ((self.config['model']['encode_char_and_vec_separately']) and (self.config['model']['char_embedding'])):
                #An encoder for char and word are defined separetely
                X1_word, X1_char = tf.split(X1, [self.WEOs, self.COs], axis=2)
                X2_word, X2_char = tf.split(X2, [self.WEOs, self.COs], axis=2)
                with tf.variable_scope("encode_word"):
                    X2_word_enc, X1_word_enc = self._encoder(X2_word, X1_word, self.WEOs, sentence_skip = self.sentence_skip)
                with tf.variable_scope("encode_char"):
                    X2_char_enc, X1_char_enc = self._encoder(X2_char, X1_char, self.COs, sentence_skip = self.sentence_skip)
                X1_enc = tf.concat([X1_word_enc, X1_char_enc], axis=2)
                X2_enc = tf.concat([X2_word_enc, X2_char_enc], axis=2)
            else: #same encoder for both
                X2_enc, X1_enc = self._encoder(X2, X1, self.WEOs+self.COs, sentence_skip = self.sentence_skip)
            len_X1 = tf.shape(X1)[1]
            len_X2 = tf.shape(X2)[1]
            X1_enc_red = tf.zeros(shape=[self.Bs,len_X1,self.WEAs], dtype=tf.float32)
            X2_enc_red = tf.zeros(shape=[self.Bs,len_X2,self.WEAs], dtype=tf.float32)
            X2_enc_red, X1_enc_red = self._encoder(X2_enc_red, X1_enc_red, out_size, sentence_skip = self.sentence_skip, reuse=True)
            X1_enc = tf.nn.dropout(X1_enc, tf.pow(self.keep_prob_encoder, self.config['model']['reduced_layer_dropout_amplification']-1))
            X2_enc = tf.nn.dropout(X2_enc, tf.pow(self.keep_prob_encoder, self.config['model']['reduced_layer_dropout_amplification']-1))
            X2 = tf.nn.dropout(X2, tf.pow(self.keep_prob_encoder, self.config['model']['reduced_layer_dropout_amplification']))
            X1 = tf.nn.dropout(X1, tf.pow(self.keep_prob_encoder, self.config['model']['reduced_layer_dropout_amplification']))
            att_layer_X1X1_out, X1_enc_out = self._attention_layer(X1=X1_enc, X2=X1_enc, X3=X1, X4=X1_enc_red,
                                                       mask=mask[X1X1],
                                                       scope='Layer_red',
                                                       comp_size=X1_comp_size,
                                                       reuse=False,
                                                       dropout=self.config['model']['reduced_layer_dropout_amplification'],
                                                       sentence_masking = X1_masking)
            att_layer_X1X1 = self._layer_normalization(
                                tf.add(X1,
                                       att_layer_X1X1_out),
                                scope='norm_'+X1X1,
                                shape=X1_comp_size[0])

            X1_enc_out = self._layer_normalization(X1_enc_out+X1_enc_red, scope='norm_Encoder'+X1X1,shape=out_size)

            FF_X1X1 = self._layer_normalization(
                                    tf.add(att_layer_X1X1,
                                           self._FeedForward_NN(att_layer_X1X1,
                                                          'FF_' + X1X1,
                                                           comp_size=X1_comp_size,
                                                           dropout=self.config['model']['reduced_layer_dropout_amplification'])),
                                    scope='norm_FF_'+X1X1,
                                    shape=X1_comp_size[0])

            att_layer_X2X2_out, X2_enc_out = self._attention_layer(X1=X2_enc, X2=X2_enc, X3=X2, X4=X2_enc_red,
                                                       mask=mask[X2X2],
                                                       scope='Layer_red',
                                                       comp_size=X2_comp_size,
                                                       reuse=True,
                                                       dropout=self.config['model']['reduced_layer_dropout_amplification'],
                                                       sentence_masking = X2_masking)

            att_layer_X2X2 = self._layer_normalization(
                                tf.add(X2,
                                       att_layer_X2X2_out),
                                scope='norm_'+X2X2,
                                shape=X2_comp_size[0])
            X2_enc_out = self._layer_normalization(X2_enc_out+X2_enc_red, scope='norm_Encoder'+X2X2,shape=out_size)

            if self.config['model']['number_of_cross_attentions']>0:
                cross_out_X1 = self._proc_cross_layer(FF_X1X1, 'cross_prepare_att', X1_comp_size)
                mask_X2X1 = tf.expand_dims(mask[X2_v],2)
            else:
                cross_out_X1 = FF_X1X1
                mask_X2X1 = mask[X2X1]
            att_layer_X1X2 = self._layer_normalization(
                                tf.add(att_layer_X2X2,
                                       self._attention_layer(
                                                       X1=cross_out_X1,
                                                       X2=att_layer_X2X2,
                                                       mask=mask_X2X1,
                                                       scope=X2X1,
                                                       comp_size=X2_comp_size,
                                                       dropout=self.config['model']['reduced_layer_dropout_amplification'],
                                                       cross=True)),
                                scope='norm_'+X2X1,
                                shape=X2_comp_size[0])

            FF_X2X2 = self._layer_normalization(
                                    tf.add(att_layer_X1X2,
                                           self._FeedForward_NN(att_layer_X1X2,
                                                                'FF_' + X2X2,
                                                                comp_size=X2_comp_size,
                                                                dropout=self.config['model']['reduced_layer_dropout_amplification'],
                                                                reuse=False)),
                                    scope='norm_FF_' + X2X2,
                                    shape=X2_comp_size[0])

            length_X1 = X1.get_shape()[1]
            length_X2 = X2.get_shape()[1]
            FF_X1X1 = tf.expand_dims(FF_X1X1, 1)
            FF_X2X2 = tf.expand_dims(FF_X2X2, 1)
            FF_X1X1.set_shape([self.Bs, 1, length_X1, X1_comp_size[0]])
            FF_X2X2.set_shape([self.Bs, 1, length_X2, X2_comp_size[0]])
            output_1 = tf.layers.conv2d(FF_X1X1,
                                        filters=out_size,
                                        kernel_size=1,
                                        strides=1,
                                        use_bias=True,
                                        activation=None,
                                        kernel_initializer=self.initializer,
                                        reuse=False,
                                        name='affine_op_X')

            output_2 = tf.layers.conv2d(FF_X2X2,
                                        filters=out_size,
                                        kernel_size=1,
                                        strides=1,
                                        use_bias=True,
                                        activation=None,
                                        kernel_initializer=self.initializer,
                                        reuse=True,
                                        name='affine_op_X')
            output_1 = tf.nn.dropout(tf.squeeze(output_1, 1), self.keep_prob_encoder)
            output_2 = tf.nn.dropout(tf.squeeze(output_2, 1), self.keep_prob_encoder)
            output_1.set_shape([self.Bs, length_X1, out_size])
            output_1 = [X1_enc_out, output_1]
            output_2.set_shape([self.Bs, length_X2, out_size])
            output_2 = [X2_enc_out, output_2]
            if switch:
                return output_2, output_1
            else:
                return output_1, output_2

    def _one_layer_split_enc_emb(self, Q, X, mask, scope, switch=False):
        # Defining masks and scopes
        if switch:
            X1 = X
            X1_comp_size = self.x_comp_size
            X2 = Q
            X2_comp_size = self.q_comp_size
            X1X1, X2X2, X2X1, X1X2, X2_v = 'xx', 'qq', 'qx', 'xq', 'q'
        else:
            X1 = Q
            X1_comp_size = self.q_comp_size
            X2 = X
            X2_comp_size = self.x_comp_size
            X1X1, X2X2, X2X1, X1X2, X2_v = 'qq', 'xx', 'xq', 'qx', 'x'
        with tf.variable_scope(scope):
            X1_enc, X1_emb = X1
            X2_enc, X2_emb = X2
            weight_1 = tf.get_variable("weight_enc_emb_X1", shape=[1], dtype=tf.float32, initializer = tf.zeros_initializer())
            weight_2 = tf.get_variable("weight_enc_emb_X2", shape=[1], dtype=tf.float32, initializer = tf.zeros_initializer())
            weight_3 = tf.get_variable("weight_enc_emb_X3", shape=[1], dtype=tf.float32, initializer = tf.zeros_initializer())
            sig_weight_1 = tf.sigmoid(weight_1)
            sig_weight_2 = tf.sigmoid(weight_2)
            sig_weight_3 = tf.sigmoid(weight_3)
            X1_emb_enc = X1_enc*sig_weight_1 + X1_emb*(1-sig_weight_1)
            X2_emb_enc = X2_enc*sig_weight_2 + X2_emb*(1-sig_weight_2)
            att_layer_X1X1_out, X1_enc_out = self._attention_layer(X1=X1_emb_enc, X2=X1_emb_enc, X3=X1_emb, X4=X1_enc,
                                                       mask=mask[X1X1],
                                                       scope='X1X1',
                                                       comp_size=X1_comp_size,
                                                       reuse=False,
                                                       dropout=self.config['model']['reduced_layer_dropout_amplification'])
            att_layer_X1X1 = self._layer_normalization(
                                tf.add(X1_emb_enc,
                                       att_layer_X1X1_out),
                                scope='norm_'+X1X1,
                                shape=X1_comp_size[0])

            X1_enc_out = self._layer_normalization(X1_enc_out+X1_enc, scope='norm_Encoder'+X1X1,shape=X1_comp_size[0])

            FF_X1X1 = self._layer_normalization(
                                    tf.add(att_layer_X1X1,
                                           self._FeedForward_NN(att_layer_X1X1,
                                                          'FF_' + X1X1,
                                                           comp_size=X1_comp_size,
                                                           reuse=False)),
                                    scope='norm_FF_'+X1X1,
                                    shape=X1_comp_size[0])


            att_layer_X2X2_out, X2_enc_out = self._attention_layer(X1=X2_emb_enc, X2=X2_emb_enc, X3=X2_emb, X4=X2_enc,
                                                       mask=mask[X2X2],
                                                       scope='X2X2',
                                                       comp_size=X2_comp_size,
                                                       reuse=False)

            att_layer_X2X2 = self._layer_normalization(
                                tf.add(X2_emb_enc,
                                       att_layer_X2X2_out),
                                scope='norm_'+X2X2,
                                shape=X2_comp_size[0])
            X2_enc_out = self._layer_normalization(X2_enc_out+X2_enc, scope='norm_Encoder'+X2X2,shape=X1_comp_size[0])

            if self.config['model']['number_of_cross_attentions']>0:
                cross_out_X1 = self._proc_cross_layer(FF_X1X1, 'cross_prepare_att', X1_comp_size)
                mask_X2X1 = tf.expand_dims(mask[X2_v],2)
            else:
                cross_out_X1 = FF_X1X1
                mask_X2X1 = mask[X2X1]
            att_layer_X1X2 = self._layer_normalization(
                                tf.add(att_layer_X2X2,
                                       self._attention_layer(
                                                       X1=cross_out_X1,
                                                       X2=att_layer_X2X2,
                                                       X3=cross_out_X1*(sig_weight_3)+X1_enc_out*(1-sig_weight_3),
                                                       mask=mask_X2X1,
                                                       scope=X2X1,
                                                       comp_size=X2_comp_size,
                                                       cross=True)),
                                scope='norm_'+X2X1,
                                shape=X2_comp_size[0])

            FF_X2X2 = self._layer_normalization(
                                    tf.add(att_layer_X1X2,
                                           self._FeedForward_NN(att_layer_X1X2,
                                                                'FF_' + X2X2,
                                                                comp_size=X2_comp_size,
                                                                reuse=False)),
                                    scope='norm_FF_' + X2X2,
                                    shape=X2_comp_size[0])

            output_1 = [X1_enc_out, FF_X1X1]
            output_2 = [X2_enc_out, FF_X2X2]
            if switch:
                return output_2, output_1
            else:
                return output_1, output_2

    def _one_layer(self, Q, X, mask, scope, switch=False):
        # Defining masks and scopes
        if switch:
            X1 = X
            X1_comp_size = self.x_comp_size
            X2 = Q
            X2_comp_size = self.q_comp_size
            X1X1, X2X2, X2X1, X1X2, X2_v = 'xx', 'qq', 'qx', 'xq', 'q'
            X1_masking = True
            X2_masking = False
        else:
            X1 = Q
            X1_comp_size = self.q_comp_size
            X2 = X
            X2_comp_size = self.x_comp_size
            X1X1, X2X2, X2X1, X1X2, X2_v = 'qq', 'xx', 'xq', 'qx', 'x'
            X1_masking = False
            X2_masking = True
        with tf.variable_scope(scope):
            att_layer_X1X1 = self._layer_normalization(
                                tf.add(X1,
                                       self._attention_layer(X1=X1,
                                                       mask=mask[X1X1],
                                                       scope=X1X1,
                                                       comp_size=X1_comp_size,
                                                       sentence_masking=X1_masking)),
                                scope='norm_'+X1X1)

            output_1 = FF_X1X1 = self._layer_normalization(
                                    tf.add(att_layer_X1X1,
                                           self._FeedForward_NN(att_layer_X1X1,
                                                          'FF_' + X1X1,
                                                           comp_size=X1_comp_size)),
                                    scope='norm_FF_'+X1X1)

            att_layer_X2X2 = self._layer_normalization(
                                tf.add(X2,
                                       self._attention_layer(X1=X2,
                                                       mask=mask[X2X2],
                                                       scope=X1X1,
                                                       comp_size=X1_comp_size,
                                                       sentence_masking=X2_masking,
                                                       reuse=True)),
                                scope='norm_' + X2X2)
            if self.config['model']['number_of_cross_attentions']>0:
                cross_out_X1 = self._proc_cross_layer(FF_X1X1, 'cross_prepare_att', X1_comp_size)
                mask_X2X1 = tf.expand_dims(mask[X2_v],2)
            else:
                cross_out_X1 = FF_X1X1
                mask_X2X1 = mask[X2X1]
            att_layer_X1X2 = self._layer_normalization(
                                tf.add(att_layer_X2X2,
                                       self._attention_layer(
                                                       X1=cross_out_X1,
                                                       X2=att_layer_X2X2,
                                                       mask=mask_X2X1,
                                                       scope=X2X1,
                                                       comp_size=X2_comp_size,
                                                       cross=True)),
                                scope='norm_'+X1X2)

            output_2 = FF_X2X2 = self._layer_normalization(
                                    tf.add(att_layer_X1X2,
                                           self._FeedForward_NN(att_layer_X1X2,
                                                                'FF_' + X2X2,
                                                                comp_size=X1_comp_size,
                                                                reuse=False)),
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
                                                       scope='QQ',
                                                       comp_size=self.q_comp_size)),
                                scope='norm_QQ')
            # FF neural network Q_Layer
            FF_QQ = self._layer_normalization(
                        tf.add(att_layer_QQ,
                               self._FeedForward_NN(att_layer_QQ,
                                                    'FF_QQ',
                                                    comp_size=self.q_comp_size)),
                        scope='norm_FF_QQ')
            # Self-Atttention Layer X
            att_layer_XX = self._layer_normalization(
                                tf.add(X,
                                       self._attention_layer(
                                                       X1=X,
                                                       mask=mask['xx'],
                                                       scope='XX',
                                                       comp_size=self.x_comp_size)),
                                scope='norm_XX')
            # FF neural network X_Layer
            FF_XX = self._layer_normalization(
                                tf.add(att_layer_XX,
                                       self._FeedForward_NN(att_layer_XX,
                                                            'FF_XX',
                                                            comp_size=self.x_comp_size)),
                                scope='norm_FF_XX')
            # Cross attention of X and Q:
            att_layer_XQ = self._layer_normalization(
                                tf.add(att_layer_QQ,
                                       self._attention_layer(
                                                       X1=FF_XX,
                                                       X2=att_layer_QQ,
                                                       mask=mask['qx'],
                                                       scope='QX',
                                                       comp_size=self.x_comp_size)),
                                scope='norm_QX')
            att_layer_QX = self._layer_normalization(
                                tf.add(att_layer_XX,
                                       self._attention_layer(
                                                       X1=FF_QQ,
                                                       X2=att_layer_XX,
                                                       mask=mask['xq'],
                                                       scope='XQ',
                                                       comp_size=self.q_comp_size)),
                                scope='norm_XQ')
            # Output of X and Q:
            output_Q = self._layer_normalization(
                            tf.add(att_layer_XQ,
                                   self._FeedForward_NN(att_layer_XQ,
                                                        'FF_Q_out',
                                                        comp_size=self.x_comp_size)),
                            scope='norm_FF_Q_out')
            output_X = self._layer_normalization(
                            tf.add(att_layer_QX,
                                   self._FeedForward_NN(att_layer_QX,
                                                        'FF_X_out',
                                                        comp_size=self.q_comp_size)),
                            scope='norm_FF_X_out')
            return output_Q, output_X

    def _one_layer_symmetric_small(self, Q, X, mask, scope, switch=False):  # Although switch input is not used here, it was added for compatibility with one layer function.
        with tf.variable_scope(scope):
            # Self-Atttention Layer Q
            att_layer_QQ = self._layer_normalization(
                                tf.add(Q,
                                       self._attention_layer(
                                                       X1=Q,
                                                       mask=mask['qq'],
                                                       scope='QQ',
                                                       comp_size=self.q_comp_size)),
                                scope='norm_QQ')
            # FF neural network Q_Layer
            FF_QQ = self._layer_normalization(
                        tf.add(att_layer_QQ,
                               self._FeedForward_NN(att_layer_QQ,
                                                    'FF_QQ',
                                                    comp_size=self.q_comp_size)),
                        scope='norm_FF_QQ')
            # Self-Atttention Layer X
            att_layer_XX = self._layer_normalization(
                                tf.add(X,
                                       self._attention_layer(
                                                       X1=X,
                                                       mask=mask['xx'],
                                                       scope='XX',
                                                       comp_size=self.x_comp_size)),
                                scope='norm_XX')
            # FF neural network X_Layer
            FF_XX = self._layer_normalization(
                                tf.add(att_layer_XX,
                                       self._FeedForward_NN(att_layer_XX,
                                                            'FF_XX',
                                                            comp_size=self.x_comp_size)),
                                scope='norm_FF_XX')
            # Cross attention of X and Q:
            output_Q = self._layer_normalization(
                                tf.add(FF_QQ,
                                       self._attention_layer(
                                                       X1=FF_XX,
                                                       X2=att_layer_QQ,
                                                       mask=mask['qx'],
                                                       scope='QX',
                                                       comp_size=self.x_comp_size)),
                                scope='norm_QX')
            output_X = self._layer_normalization(
                                tf.add(FF_XX,
                                       self._attention_layer(
                                                       X1=FF_QQ,
                                                       X2=att_layer_XX,
                                                       mask=mask['xq'],
                                                       scope='XQ',
                                                       comp_size=self.q_comp_size)),
                                scope='norm_XQ')
            return output_Q, output_X

    def _one_layer_parallel(self, Q, X, mask, scope, switch=False):
        # Defining masks and scopes
        Q_left, X_left = self._one_layer(Q, X, mask, scope + '_left', switch)
        Q_right, X_right = self._one_layer(Q, X, mask, scope + '_right', switch)

        Q = tf.maximum(Q_left, Q_right)
        X = tf.maximum(X_left, X_right)

        return Q, X

    def _self_att_layer(self, Q, X, mask, scope, switch=False):
        with tf.variable_scope(scope):
            att_layer_QQ = self._layer_normalization(
                                tf.add(Q,
                                       self._attention_layer(
                                                       X1=Q,
                                                       mask=mask['qq'],
                                                       scope='QQ',
                                                       comp_size=self.q_comp_size)),
                                scope='norm_QQ')
            FF_QQ = self._layer_normalization(
                        tf.add(att_layer_QQ,
                               self._FeedForward_NN(att_layer_QQ,
                                                    'FF_QQ',
                                                    comp_size=self.q_comp_size)),
                        scope='norm_FF_QQ')
            # Self-Atttention Layer X
            att_layer_XX = self._layer_normalization(
                                tf.add(X,
                                       self._attention_layer(
                                                       X1=X,
                                                       mask=mask['xx'],
                                                       scope='XX',
                                                       comp_size=self.x_comp_size)),
                                scope='norm_XX')

            FF_XX = self._layer_normalization(
                                tf.add(att_layer_XX,
                                       self._FeedForward_NN(att_layer_XX,
                                                            'FF_XX',
                                                            comp_size=self.x_comp_size)),
                                scope='norm_FF_XX')

        return FF_QQ, FF_XX

    def _memory_layer(self, q):
        M_size = self.config['model']['memory_size']
        with tf.variable_scope("memory"):
            # Create the memory cells
            M = tf.get_variable(
                "M",
                dtype=tf.float32,
                shape=(1, M_size, self.WEAs),
                initializer=self.initializer)
            # Tile up the memory cells
            # we need a copy for each example in the batch
            M_batch = tf.tile(M, [self.Bs, 1, 1])
            mask = tf.cast(
                      tf.sign(
                          tf.matmul(tf.expand_dims(self.q, -1),
                                    tf.ones([self.Bs, 1, M_size], tf.int32))), tf.float32)
            qM = self._attention_layer(X1=M_batch, X2=q,
                                       mask=mask,
                                       scope='qM',
                                       comp_size=self.q_comp_size)
            qM = self._layer_normalization(qM+q, scope='norm_memory')
            qM = self._layer_normalization(qM + self._FeedForward_NN(qM,
                                                                     'FF_qM',
                                                                     comp_size=self.q_comp_size),
                                           scope='norm_FF_memory')

        return qM


    def _linear_sel(self, X, mask, scope, size_input):
        """ Select one vector among n vectors by max(w*X) """
        with tf.variable_scope(scope):
            length_X = X.get_shape()[1]
            X = tf.expand_dims(X, 2)
            X.set_shape([self.Bs, length_X, 1, size_input])
            #X = tf.reshape(X, [self.Bs, -1, 1, self.WEAs])
            logits = tf.layers.conv2d(X,
                                      filters=1,
                                      kernel_size=(1, 1),
                                      strides=1,
                                      padding='same',
                                      use_bias=True,
                                      kernel_initializer=self.initializer,
                                      name='conv_sel')
            logits = tf.reshape(logits, [self.Bs, -1])
            X = tf.squeeze(X)
            X.set_shape([self.Bs, length_X, size_input])
            output = tf.nn.softmax(
                        tf.add(logits,
                               tf.multiply(1.0 - mask['x'], VERY_LOW_NUMBER)))
        return output, logits

    def _linear_sel_y2(self, X, y1_sel, mask, scope, size_input):
        """ Select one vector among n vectors by max(w*X) """
        with tf.variable_scope(scope):
            length_X = X.get_shape()[1]
            X = tf.expand_dims(X, 2)
            X.set_shape([self.Bs, length_X, 1, size_input])

            y1_selected = tf.cast(tf.expand_dims(tf.argmax(y1_sel, axis=1),1), tf.int32)
            range_x = tf.expand_dims(tf.range(0, self.max_size_x[-1], 1), 0)
            mask_new = tf.cast(tf.round(tf.cast(tf.less(y1_selected-1,range_x), tf.float32) + mask['x']-1.0), tf.float32)
            self.y_corrected = self.y
            self.y2_corrected = tf.multiply(self.y2, mask_new)

            logits = tf.layers.conv2d(X,
                                      filters=1,
                                      kernel_size=(1, 1),
                                      strides=1,
                                      padding='same',
                                      use_bias=True,
                                      kernel_initializer=self.initializer,
                                      name='conv_sel')
            logits = tf.reshape(logits, [self.Bs, -1])
            X = tf.squeeze(X)
            X.set_shape([self.Bs, length_X, size_input])
            output = tf.nn.softmax(
                        tf.add(logits,
                               tf.multiply(1.0 - mask_new, VERY_LOW_NUMBER)))
        return output, logits

    def _conv_1D(self, X, scope, emb_in, emb_out, kernel_size, reuse, activation=None, padding='same', use_bias=True):
        """ Select one vector among n vectors by max(w*X) """
        length_X = X.get_shape()[1]
        X = tf.expand_dims(X, 1)
        X.set_shape([self.Bs, 1, length_X, emb_in])
        conv_out = tf.squeeze(tf.layers.conv2d(X,
                                  filters=emb_out,
                                  kernel_size=kernel_size,
                                  strides=1,
                                  padding=padding,
                                  use_bias=use_bias,
                                  activation=activation,
                                  kernel_initializer=self.initializer,
                                  reuse=reuse,
                                  name=scope))
        X = tf.squeeze(X)
        X.set_shape([self.Bs, length_X, emb_in])
        return conv_out


    def _conv_sel(self, X, mask, scope, size_input):
        """ Select one vector among n vectors by max(w*X) """
        with tf.variable_scope(scope):
            length_X = X.get_shape()[1]
            X = tf.expand_dims(X, 2)
            X.set_shape([self.Bs, length_X, 1, size_input])
            logits_hidden = tf.layers.conv2d(X,
                                      filters=32,
                                      kernel_size=(self.config['model']['conv_selector_kernel_size'], 1),
                                      strides=1,
                                      padding='same',
                                      use_bias=True,
                                      activation=tf.nn.relu,
                                      kernel_initializer=self.initializer,
                                      name='conv_sel')
            logits_drop = tf.nn.dropout(logits_hidden, keep_prob=self.keep_prob_selector)
            logits_out = tf.squeeze(tf.layers.conv2d(logits_drop,
                                      filters=1,
                                      kernel_size=(self.config['model']['conv_selector_kernel_size'], 1),
                                      strides=1,
                                      padding='same',
                                      use_bias=True,
                                      kernel_initializer=self.initializer,
                                      name='conv_sel_2'))
            X = tf.squeeze(X)
            X.set_shape([self.Bs, length_X, size_input])
            output = tf.nn.softmax(
                        tf.add(logits_out,
                               tf.multiply(1.0 - mask['x'], VERY_LOW_NUMBER)))

        return output, logits_out

    def _conv_sel_2(self, X, y1_sel, mask, scope, size_input):
        """ Select one vector among n vectors by max(w*X) """
        length_X = X.get_shape()[1]
        with tf.variable_scope(scope):
            X = tf.reshape(X, [self.Bs, -1, 1, size_input])

            y1_selected = tf.cast(tf.expand_dims(tf.argmax(y1_sel, axis=1),1), tf.int32)
            range_x = tf.expand_dims(tf.range(0, self.max_size_x[-1], 1), 0)
            mask_new = tf.cast(tf.round(tf.cast(tf.less(y1_selected-1,range_x), tf.float32) + mask['x']-1.0), tf.float32)
            self.y_corrected = self.y
            self.y2_corrected = tf.multiply(self.y2, mask_new)

            logits = tf.layers.conv2d(X,
                                      filters=1,
                                      kernel_size=(7, 1),
                                      strides=1,
                                      padding='same',
                                      use_bias=True,
                                      kernel_initializer=self.initializer,
                                      name='conv_sel_2')
            logits = tf.reshape(logits, [self.Bs, -1])
            output = tf.nn.softmax(
                        tf.add(logits,
                               tf.multiply(1.0 - mask_new, VERY_LOW_NUMBER)))
        return output, logits

    def _single_conv(self, X, mask, scope, size_input):
        """ Select one vector among n vectors by max(w*X) """
        length_X = X.get_shape()[1]
        with tf.variable_scope(scope):
            X = tf.reshape(X, [self.Bs, -1, 1, size_input])
            logits = tf.layers.conv2d(X,
                                      filters=16,
                                      kernel_size=(1, 1),
                                      strides=1,
                                      padding='same',
                                      use_bias=True,
                                      kernel_initializer=self.initializer,
                                      name='conv_sel_2')

            logits = tf.squeeze(logits, 2)
            logits1, logits2 = tf.split(logits, 2, 2)
            logits1, logits2 = tf.reduce_max(logits1, -1), tf.reduce_max(logits2, -1)
            output1, output2 = self._process_logits(logits1, logits2, mask)
        return output1, logits1, output2, logits2

    def _double_conv(self, X, mask, scope, size_input):
        """ Select one vector among n vectors by max(w*X) """
        length_X = X.get_shape()[1]
        with tf.variable_scope(scope):
            X = tf.reshape(X, [self.Bs, -1, 1, size_input])
            logits = tf.layers.conv2d(X,
                                      filters=32,
                                      kernel_size=(self.config['model']['conv_selector_kernel_size'], 1),
                                      strides=1,
                                      padding='same',
                                      use_bias=True,
                                      activation=tf.nn.relu,
                                      kernel_initializer=self.initializer,
                                      name='conv_sel')
            logits = tf.nn.dropout(logits, keep_prob=self.keep_prob_selector)
            if self.config['model']['sigmoid_loss']:
                logits = tf.layers.conv2d(logits,
                                          filters=1,
                                          kernel_size=(self.config['model']['conv_selector_kernel_size'], 1),
                                          strides=1,
                                          padding='same',
                                          use_bias=True,
                                          kernel_initializer=self.initializer,
                                          name='conv_sel_2')
                logits = tf.squeeze(logits,[2,3])
                output = tf.sigmoid(logits)*tf.cast(self.x_mask, tf.float32)
                return output, logits, [], []

            else:
                logits = tf.layers.conv2d(logits,
                                          filters=2,
                                          kernel_size=(self.config['model']['conv_selector_kernel_size'], 1),
                                          strides=1,
                                          padding='same',
                                          use_bias=True,
                                          kernel_initializer=self.initializer,
                                          name='conv_sel_2')

                logits = tf.reshape(logits, [self.Bs, -1, 2])
                logits1, logits2 = tf.split(logits, 2, 2)
                logits1, logits2 = tf.reshape(logits1, [self.Bs, -1]), tf.reshape(logits2, [self.Bs, -1])
                output1, output2 = self._process_logits(logits1, logits2, mask)

                return output1, logits1, output2, logits2


    def _all_layers_sel(self, Q, X, mask, scope, size_input):
        with tf.variable_scope(scope):
            yp1_vec = []
            yp2_vec = []
            logits_y1_vec = []
            logits_y2_vec = []
            for i in range(len(X)):
                yp, logits_y1, yp2, logits_y2 = self._double_conv(X=X[i],
                                                                  mask=mask,
                                                                  scope='sel_layer_'+str(i),
                                                                  size_input=size_input)
                yp1_vec.append(tf.expand_dims(yp, 1))
                yp2_vec.append(tf.expand_dims(yp2, 1))
                logits_y1_vec.append(tf.expand_dims(logits_y1, 1))
                logits_y2_vec.append(tf.expand_dims(logits_y2, 1))
            yp1_out = tf.reduce_mean(tf.concat(yp1_vec, 1), 1)
            yp2_out = tf.reduce_mean(tf.concat(yp2_vec, 1), 1)
            logits_y1_out = tf.reduce_mean(tf.concat(logits_y1_vec, 1), 1)
            logits_y2_out = tf.reduce_mean(tf.concat(logits_y2_vec, 1), 1)

        return yp1_out, logits_y1_out, yp2_out, logits_y2_out

    def _sym_double_conv(self, X, mask, scope, size_input):
        """ Select one vector among n vectors by max(w*X) """
        length_X = X.get_shape()[1]
        with tf.variable_scope(scope):
            X = tf.reshape(X, [self.Bs, -1, 1, size_input])
            logits = tf.layers.conv2d(X,
                                      filters=32,
                                      kernel_size=(9, 1),
                                      strides=1,
                                      padding='same',
                                      use_bias=True,
                                      activation=tf.nn.relu,
                                      kernel_initializer=self.initializer,
                                      name='conv_sel')
            logits = tf.nn.dropout(logits, keep_prob=self.keep_prob_selector)
            logits = tf.layers.conv2d(logits,
                                      filters=4,
                                      kernel_size=(9, 1),
                                      strides=1,
                                      padding='same',
                                      use_bias=True,
                                      kernel_initializer=self.initializer,
                                      name='conv_sel_2')

            logits = tf.reshape(logits, [self.Bs, -1, 4])
            l1_left, l2_left, l1_right, l2_right = tf.split(logits, num_or_size_splits=4, axis=2)
            l1_left, l2_left = tf.reshape(l1_left, [self.Bs, -1]), tf.reshape(l2_left, [self.Bs, -1])
            l1_right, l2_right = tf.reshape(l1_right, [self.Bs, -1]), tf.reshape(l2_right, [self.Bs, -1])

            o1_left = tf.nn.softmax(
                        tf.add(l1_left,
                               tf.multiply(1.0 - mask['x'], VERY_LOW_NUMBER)))
            y1_left = tf.cast(tf.expand_dims(tf.argmax(o1_left, axis=1), 1), tf.int32)

            range_x = tf.expand_dims(tf.range(0, self.max_size_x[-1], 1), 0)

            mask_new_left = tf.cast(tf.round(tf.cast(tf.less(y1_left-1, range_x), tf.float32) + mask['x']-1.0), tf.float32)
            o2_left = tf.nn.softmax(
                        tf.add(l2_left,
                               tf.multiply(1.0 - mask_new_left, VERY_LOW_NUMBER)))

            o2_right = tf.nn.softmax(
                        tf.add(l2_right,
                               tf.multiply(1.0 - mask['x'], VERY_LOW_NUMBER)))
            y2_right = tf.cast(tf.expand_dims(tf.argmax(o2_right, axis=1), 1), tf.int32)

            mask_new_right = tf.cast(tf.round(tf.cast(tf.greater(y2_right+1, range_x), tf.float32) + mask['x']-1.0), tf.float32)
            o1_right = tf.nn.softmax(
                        tf.add(l2_right,
                               tf.multiply(1.0 - mask_new_right, VERY_LOW_NUMBER)))

            output1 = (o1_left + o1_right)/2
            output2 = (o2_left + o2_right)/2

            logits1 = (l1_left + l1_right)/2
            logits2 = (l2_left + l2_right)/2

        return output1, logits1, output2, logits2

    def _process_logits(self, logits1, logits2, mask):
        range_x = tf.expand_dims(tf.range(0, self.max_size_x[-1], 1), 0)
        def f1():
            output1 = tf.nn.softmax(
                        tf.add(logits1,
                               tf.multiply(1.0 - mask['x'], VERY_LOW_NUMBER)))
            y1_selected = tf.cast(tf.expand_dims(tf.argmax(output1, axis=1),1), tf.int32)
            mask1 = tf.cast(mask['x'], tf.float32)
            mask2 = tf.cast(tf.round(tf.cast(tf.less(y1_selected-1, range_x), tf.float32) + mask['x']-1.0), tf.float32)
            return mask1, mask2

        def f2():
            output2 = tf.nn.softmax(
                        tf.add(logits2,
                               tf.multiply(1.0 - mask['x'], VERY_LOW_NUMBER)))
            y2_selected = tf.cast(tf.expand_dims(tf.argmax(output2, axis=1),1), tf.int32)
            mask1 = tf.cast(tf.round(tf.cast(tf.greater(y2_selected+1, range_x), tf.float32) + mask['x']-1.0), tf.float32)
            mask2 = tf.cast(mask['x'], tf.float32)
            return mask1, mask2
        if self.config['model']['alternating_y1_y2']:
            mask1, mask2 = tf.cond(tf.equal(tf.floormod(self.global_step, 2), 0), f1, f2)
        elif self.config['model']['single_loss'] or self.config['model']['no_masking_loss_computation']:
            mask1 = mask2 = tf.cast(mask['x'], tf.float32)
        else:
            mask1, mask2 = f1()
        output1 = tf.nn.softmax(
                    tf.add(logits1,
                           tf.multiply(1.0 - mask1, VERY_LOW_NUMBER)))
        output2 = tf.nn.softmax(
                    tf.add(logits2,
                           tf.multiply(1.0 - mask2, VERY_LOW_NUMBER)))
        self.y_corrected = tf.multiply(self.y, mask1)
        self.y2_corrected = tf.multiply(self.y2, mask2)

        return output1, output2

    def get_prob_matrix(self, prob_1, prob_2, flat=True, max_answer_size=-1):
        # Multiply the probabilities of y1 and y2
        prob_1 = tf.expand_dims(prob_1, 1)
        prob_2 = tf.expand_dims(prob_2, 1)
        prob_1T = tf.transpose(prob_1, [0, 2, 1])
        P = tf.matmul(prob_1T, prob_2)
        # Filer out examples where y2 < y1
        upper_diag = tf.matrix_band_part(tf.ones_like(P),
                                         num_lower=0,
                                         num_upper=max_answer_size)
        P = tf.multiply(P, upper_diag)
        if flat:
            return tf.reshape(P, [self.Bs, -1])
        else:
            return P

    def get_y1_y2(self, prob_1, prob_2):
        # Get the probability matrix
        P = self.get_prob_matrix(prob_1, prob_2, flat=False, max_answer_size=self.config['model']['max_answer_size']-1)
        flat_P = tf.reshape(P, [self.Bs, -1])
        # Take the value and index of the maximum
        values, indices = tf.nn.top_k(flat_P, k=1)
        # Calculate the final indices
        ind_x = tf.cast(tf.floor(indices/tf.shape(P)[2]), tf.int32)
        ind_y = tf.floormod(indices, tf.shape(P)[2])

        return tf.concat(tf.unstack(ind_x), 0), tf.concat(tf.unstack(ind_y), 0)

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

    def _cross_sel(self, Q, X, mask, scope, size_input):
        """ Attention over attention for computing y1"""
        with tf.variable_scope(scope):
            length_Q = Q.get_shape()[1]
            Q = tf.expand_dims(Q, 2)
            Q.set_shape([self.Bs, length_Q, 1, size_input])
            #Scale question matrix with a matrix W*Q
            Q_Scaled = tf.squeeze(tf.layers.conv2d(Q,
                                              filters=size_input,
                                              kernel_size=1,
                                              strides=1,
                                              name='W'))
            Q = tf.squeeze(Q)
            Q.set_shape([self.Bs, length_Q, size_input])
            logits = tf.matmul(Q_Scaled, tf.transpose(X, [0, 2, 1]))
            logits_mean = tf.reduce_mean(logits, 2, keep_dims=True)
            logits_std = tf.sqrt(1e-8+tf.reduce_mean(tf.square(logits-logits_mean),2, keep_dims=True))
            logits_norm = (logits-logits_mean)/logits_std
            logits_norm = tf.squeeze(tf.reduce_sum(logits_norm,1))

            """ Select one vector among n vectors by max(w*X) """
            length_logits = logits_norm.get_shape()[1]
            logits_norm = tf.expand_dims(tf.expand_dims(logits_norm, 2), 3)
            logits_norm.set_shape([self.Bs, length_logits, 1, 1])
            logits_norm = tf.layers.conv2d(logits_norm,
                                      filters=32,
                                      kernel_size=(9, 1),
                                      strides=1,
                                      padding='same',
                                      use_bias=True,
                                      activation=tf.nn.relu,
                                      kernel_initializer=self.initializer,
                                      name='conv_sel')
            logits_norm = tf.nn.dropout(logits_norm, keep_prob=self.keep_prob_selector)
            logits_norm = tf.layers.conv2d(logits_norm,
                                      filters=2,
                                      kernel_size=(9, 1),
                                      strides=1,
                                      padding='same',
                                      use_bias=True,
                                      kernel_initializer=self.initializer,
                                      name='conv_sel_2')

            logits_norm = tf.reshape(logits_norm, [self.Bs, -1, 2])
            logits1, logits2 = tf.split(logits_norm, 2, 2)
            logits1, logits2 = tf.reshape(logits1, [self.Bs, -1]), tf.reshape(logits2, [self.Bs, -1])
            output1, output2 = self._process_logits(logits1, logits2, mask)

        return output1, logits1, output2, logits2

    def _AoA_sel(self, Q, X, mask, scope):
        """ Attention over attention for computing y1"""
        with tf.variable_scope(scope):
            length_Q = Q.get_shape()[1]
            Q = tf.expand_dims(Q, 2)
            Q.set_shape([self.Bs, length_Q, 1, self.WEAs])
            #Scale question matrix with a matrix W*Q
            Q_Scaled = tf.squeeze(tf.layers.conv2d(Q,
                                              filters=self.WEAs,
                                              kernel_size=1,
                                              strides=1,
                                              name='W'))
            Q = tf.squeeze(Q)
            Q.set_shape([self.Bs, length_Q, self.WEAs])
            logits = tf.matmul(X, tf.transpose(Q_Scaled, [0, 2, 1]))

            # Sofmax for attention of Q in X: softmax(X*W*Q)
            softmax_X = tf.nn.softmax(
                tf.add(
                    tf.divide(logits, tf.sqrt(tf.cast(self.WEAs, tf.float32))),
                    tf.multiply(1.0 - mask['xq'], VERY_LOW_NUMBER)),
                dim=1)


            # Sofmax for attention of X in Q: softmax((X*W*Q)^T)
            softmax_Q = tf.nn.softmax(
                tf.add(
                    tf.divide(logits, tf.sqrt(tf.cast(self.WEAs, tf.float32))),
                    tf.multiply(1.0 - mask['xq'], VERY_LOW_NUMBER)),
                dim=2)
            softmax_Q = tf.expand_dims(tf.reduce_mean(softmax_Q, axis=1), 2) #the add of the 2nd dim is analogous to transpose.
            #Compute Attention over Attention
            logits_y = tf.squeeze(
                            tf.matmul(
                                softmax_X,
                                softmax_Q), axis=-1)
            #Final mask is applied
            softmax_y = tf.nn.softmax(
                tf.add(
                    logits_y,
                    tf.multiply(1.0 - mask['x'], VERY_LOW_NUMBER)),
                dim=-1)
        return softmax_y, logits_y

    def _y_selection(self, Q, X, mask, scope, method="linear", y1_sel=None, size_input=None):
        if method == "linear":
            output, logits = self._linear_sel(X, mask, scope, size_input)
        elif method == "split_layer":
            output, logits = self._split_layer_sel(Q, X, mask, scope)
        elif method == "AoA":
            output, logits = self._AoA_sel(Q, X, mask, scope)
        elif method == "cross":
            output, logits = self._cross_sel(Q, X, mask, scope)
        elif method == "linear_y2":
            output, logits = self._linear_sel_y2(X, y1_sel, mask, scope, size_input)
        elif method == "conv":
            output, logits = self._conv_sel(X, mask, scope, size_input)
        elif method == "conv2":
            output, logits = self._conv_sel_2(X, y1_sel, mask, scope, size_input)
        return output, logits

    def _get_layer_func(self, ID):
        if ID == "S":
            return self._self_att_layer
        elif ID=="O":
            return self._one_layer

    def _build_forward_Attention(self):
        config = self.config
        # Mask matrices
        mask = {}
        mask['q'] = tf.cast(tf.sign(self.q), tf.float32)
        mask['x'] = tf.cast(tf.sign(self.x), tf.float32)
        mask['qq'] = tf.cast(
                        tf.sign(
                            tf.matmul(tf.sign(tf.expand_dims(self.q, -1)),
                                      tf.sign(tf.expand_dims(self.q, 1)))), tf.float32)
        mask['xx'] = tf.cast(
                        tf.sign(
                            tf.matmul(tf.sign(tf.expand_dims(self.x, -1)),
                                      tf.sign(tf.expand_dims(self.x, 1)))), tf.float32)
        mask['xq'] = tf.cast(
                        tf.sign(
                            tf.matmul(tf.sign(tf.expand_dims(self.x, -1)),
                                      tf.sign(tf.expand_dims(self.q, 1)))), tf.float32)
        mask['qx'] = tf.cast(
                        tf.sign(
                            tf.matmul(tf.sign(tf.expand_dims(self.q, -1)),
                                      tf.sign(tf.expand_dims(self.x, 1)))), tf.float32)
        with tf.variable_scope("word_emb"):
            # TODO: Having a config variable for this is not the best solution
            # TODO: Save the embedding matrix somewhere else
            word_emb_mat = tf.get_variable(
                "word_emb_mat",
                dtype=tf.float32,
                shape=[self.WVs, self.WEs],
                initializer=self.initializer)  # [WVs,WEAs]
            if config['pre']['use_glove_for_unk']:
                word_emb_mat = tf.concat([word_emb_mat, self.new_emb_mat],
                                         axis=0)
            with tf.name_scope("word"):
                Ax = tf.nn.embedding_lookup(word_emb_mat, self.x)  # [Bs,Ps,Hn]
                Aq = tf.nn.embedding_lookup(word_emb_mat, self.q)  # [Bs,Qs,Hn]

            if self.config['model']['UNK=zero']:
                Ax = tf.multiply(Ax,tf.cast(tf.expand_dims(self.x_without_unk_mask, 2), tf.float32))
                Aq = tf.multiply(Aq,tf.cast(tf.expand_dims(self.q_without_unk_mask, 2), tf.float32))
            else:
                Ax = tf.multiply(Ax,tf.cast(tf.expand_dims(self.x_mask, 2), tf.float32))
                Aq = tf.multiply(Aq,tf.cast(tf.expand_dims(self.q_mask, 2), tf.float32))
            if self.config['model']['word2vec_scaling']:
                x_scaled = self._embed_scaling(Ax, inp_size=self.WEs, out_size=self.WEOs)
                q_scaled = self._embed_scaling(Aq, inp_size=self.WEs, out_size=self.WEOs, second=True)
            else:
                x_scaled = Ax
                q_scaled = Aq

            if self.config['model']['context']:
                Ax = self._context_network(Ax)
                Aq = self._context_network(Aq, second=True)

        if self.config['model']['char_embedding']:
            with tf.variable_scope("char_emb"):
                if self.config['model']['pre_trained_char']:
                    char_emb_mat = tf.get_variable(
                        "char_emb_mat",
                        dtype=tf.float32,
                        initializer=self.config['model']['emb_mat_unk_chars'])  # [CVs,CEs]:
                    char_emb_mat = tf.concat([char_emb_mat, self.new_char_emb_mat],
                                         axis=0)
                else:  # There are not pre-trained characters
                    char_emb_mat = tf.get_variable(
                        "char_emb_mat",
                        initializer=tf.random_normal(shape=[self.CVs, self.CEs], mean=0.0, stddev=1.0, dtype=tf.float32))  # [CVs,CEs]
                # Embedding of characters
                Ac_short = tf.nn.embedding_lookup(char_emb_mat, self.short_words_char)
                Ac_long = tf.nn.embedding_lookup(char_emb_mat, self.long_words_char)
                # Computation of equivalent word embedding
                Ac_short = self._char2word_embedding(Ac=Ac_short, mask=tf.sign(self.short_words_char))  # Compute a vector for each word in x
                Ac_long = self._char2word_embedding(Ac=Ac_long, mask=tf.sign(self.long_words_char))  # Compute a vector for each word in q
                Ac = tf.concat([Ac_short, Ac_long], axis=0)

                # Embedding of equivalent word embedding
                Acx_word = tf.nn.embedding_lookup(Ac, self.xc)
                Acq_word = tf.nn.embedding_lookup(Ac, self.qc)
                # Masking
                Acx_word = tf.multiply(Acx_word, tf.cast(tf.expand_dims(self.xc_mask,2),tf.float32))
                Acq_word = tf.multiply(Acq_word, tf.cast(tf.expand_dims(self.qc_mask,2),tf.float32))
                if self.config['model']['highway_type']=='char': #Highway and then concatenate
                    Acx_word = self._highway_network(Acx_word, self.config['model']['highway_num_layers'], input_length=self.COs)
                    Acq_word = self._highway_network(Acq_word, num_layers=self.config['model']['highway_num_layers'], input_length=self.COs, second=True)
                    x_scaled = tf.concat([x_scaled, Acx_word], axis=2)
                    q_scaled = tf.concat([q_scaled, Acq_word], axis=2)
                    if self.config['model']['char&word2vec_scaling']:
                        x_scaled = self._embed_scaling(x_scaled, inp_size=self.WEs+self.COs, out_size=self.WEs+self.COs)
                        q_scaled = self._embed_scaling(q_scaled, inp_size=self.WEs+self.COs, out_size=self.WEs+self.COs, second=True)
                elif self.config['model']['highway_type']=='word': #Concatenate and then highway
                    x_concat = tf.concat([x_scaled, Acx_word], axis=2)
                    q_concat = tf.concat([q_scaled, Acq_word], axis=2)
                    if self.config['model']['char&word2vec_scaling']:
                        x_concat = self._embed_scaling(x_concat, inp_size=self.WEs+self.COs, out_size=self.WEs+self.COs)
                        q_concat = self._embed_scaling(q_concat, inp_size=self.WEs+self.COs, out_size=self.WEs+self.COs, second=True)
                    x_scaled = self._highway_network(x_concat, self.config['model']['highway_num_layers'], input_length=self.COs+self.WEOs)
                    q_scaled = self._highway_network(q_concat, num_layers=self.config['model']['highway_num_layers'], input_length=self.COs+self.WEOs, second=True)
                elif self.config['model']['highway_type']=='none': #Only concatenate
                # Concatenate word2vec and char2word2vec together
                    x_scaled = tf.concat([x_scaled, Acx_word], axis=2)
                    q_scaled = tf.concat([q_scaled, Acq_word], axis=2)
                    if self.config['model']['char&word2vec_scaling']:
                        x_scaled = self._embed_scaling(x_scaled, inp_size=self.WEs+self.COs, out_size=self.WEs+self.COs)
                        q_scaled = self._embed_scaling(q_scaled, inp_size=self.WEs+self.COs, out_size=self.WEs+self.COs, second=True)
                else:
                    raise Exception("Highway_type not chosen in config.json. Set it to word, char or none")

        # Dropout to zero a word2vec of a word
        if self.config['train']['dropout_word_passage']<1.0:
            x_scaled = tf.multiply(x_scaled,tf.nn.dropout(tf.cast(
                                                   tf.expand_dims(self.x_mask, 2),
                                                   tf.float32),keep_prob=self.keep_prob_word_passage))

        if config['model']['memory']:
            q_scaled = self._memory_layer(q_scaled)
        # Encoding Variables
        if config['model']['time_encoding']:
            with tf.variable_scope("Encoding"):
                if self.config['model']['one_layer_reduction']:
                    WEAs_reduct = self.WEOs+self.COs
                    FFHs_reduct = WEAs_reduct*2
                    Q_Cs = [WEAs_reduct, WEAs_reduct, WEAs_reduct, FFHs_reduct, self.MHs, self.WEAs]  # size of q attention model/q processing size/size of x attention model/number of heads/output_size of X4
                    X_Cs = Q_Cs
                    q_scaled, x_scaled = self._one_layer_reduction(Q=q_scaled, X=x_scaled, mask=mask, scope='one_layer_reduction', switch=False, out_size=self.WEAs, Q_Cs=Q_Cs, X_Cs=X_Cs) #It is already encoded
                elif ((self.config['model']['encode_char_and_vec_separately']) and (self.config['model']['char_embedding'])):
                    # An encoder for char and word are defined separetely
                    x_word, x_char = tf.split(x_scaled, [self.WEOs, self.COs], axis=2)
                    q_word, q_char = tf.split(q_scaled, [self.WEOs, self.COs], axis=2)
                    with tf.variable_scope("encode_word"):
                        x_word_enc, q_word_enc = self._encoder(x_word, q_word, self.WEOs, entence_skip = self.sentence_skip)
                    with tf.variable_scope("encode_char"):
                        x_char_enc, q_char_enc = self._encoder(x_char, q_char, self.COs, sentence_skip = self.sentence_skip)
                    x_scaled = tf.concat([x_word_enc, x_char_enc], axis=2)
                    q_scaled = tf.concat([q_word_enc, Acq_wordq_char_enc], axis=2)
                else:  # An encoder for char and word are defined together
                    if self.config['model']['matrix_reduction']:
                        x_scaled = self._conv_1D(x_scaled,
                                             scope='matrix_reduction',
                                             emb_in=self.WEOs+self.COs,
                                             emb_out=self.WEAs,
                                             kernel_size=[1,1],
                                             reuse=False,
                                             activation=None,
                                             padding='same',
                                             use_bias=True)
                        q_scaled = self._conv_1D(q_scaled,
                                             scope='matrix_reduction',
                                             emb_in=self.WEOs+self.COs,
                                             emb_out=self.WEAs,
                                             kernel_size=[1,1],
                                             reuse=True,
                                             activation=None,
                                             padding='same',
                                             use_bias=True)
                    len_q = tf.shape(self.q)[1]
                    len_x = tf.shape(self.x)[1]
                    q_enc = tf.zeros(shape=[self.Bs,len_q,self.WEAs], dtype=tf.float32)
                    x_enc = tf.zeros(shape=[self.Bs,len_x,self.WEAs], dtype=tf.float32)
                    x_enc, q_enc = self._encoder(x_enc, q_enc, self.WEAs, sentence_skip = self.sentence_skip)
                    x_scaled = tf.nn.dropout(x_scaled, self.keep_prob_encoder)
                    q_scaled = tf.nn.dropout(q_scaled, self.keep_prob_encoder)
                    x_scaled = [x_enc, x_scaled]
                    q_scaled = [q_enc, q_scaled]

        # TO BE DELETED
        # if self.WEAs != (self.WEOs+self.COs):
        #    x_scaled = self._reduce_dimension(x_scaled)
        #    q_scaled = self._reduce_dimension(q_scaled, second=True)

        # Defining functions according to config.json
        # They are used later in the final model
        # Number of layers until computation of y1
        num_layers_pre = config['model']['n_pre_layer']
        # Layers after computation of y1 to compute y2
        num_layers_post = config['model']['n_post_layer']
        switch = (lambda i: (i%2 == 1)) if config['model_options']['switching_model'] else (lambda i: False)
        if config['model_options']['layer_type'] == 'symmetric':
            layer_func = self._one_layer_symmetric
            q_scaled, x_scaled = tf.add_n(q_scaled),  tf.add_n(x_scaled)
        elif config['model_options']['layer_type']=='symmetric_small':
            layer_func = self._one_layer_symmetric_small
            q_scaled, x_scaled = tf.add_n(q_scaled),  tf.add_n(x_scaled)
        elif config['model_options']['layer_type']=='original':
            layer_func = self._one_layer
            q_scaled, x_scaled = tf.add_n(q_scaled),  tf.add_n(x_scaled)
        elif config['model_options']['layer_type']=='parallel':
            layer_func = self._one_layer_parallel
            q_scaled, x_scaled = tf.add_n(q_scaled),  tf.add_n(x_scaled)
        elif config['model_options']['layer_type']=='personalized':
            q_scaled, x_scaled = tf.add_n(q_scaled),  tf.add_n(x_scaled)
        elif config['model_options']['layer_type']=='split_emb_enc':
            layer_func = self._one_layer_split_enc_emb
        # Computing following layers after encoder
        q_i = q_scaled
        x_i = x_scaled
        if config['model_options']['layer_type']=='split_emb_enc':
            size_input = 2*self.WEAs
            q = [tf.concat(q_i,2)]
            x = [tf.concat(x_i,2)]
        else:
            size_input = self.WEAs
            q = [q_i]
            x = [x_i]
        if self.config['model_options']['layer_type']=='personalized':
            for i in range(len(self.config['model_options']['layer_sequence'])):
                layer_func = self._get_layer_func(self.config['model_options']['layer_sequence'][i])
                q_i, x_i = layer_func(q_i, x_i, mask, 'layer_'+str(i), switch=switch(i))
                q.append(q_i)
                x.append(x_i)
        else:
            for i in range(num_layers_pre+num_layers_post):
                q_i, x_i = layer_func(q_i, x_i, mask, 'layer_'+str(i), switch=switch(i))
                if config['model_options']['layer_type']=='split_emb_enc':
                    q.append(tf.concat(q_i,2))
                    x.append(tf.concat(x_i,2))
                else:
                    q.append(q_i)
                    x.append(x_i)
        if self.config['train']['dropout_last_layer_passage']<1.0:
            x[-1] = tf.nn.dropout(x[-1], keep_prob=self.keep_prob_last_x)
        if config['model']['y1_sel'] == "single_conv":
            self.yp, self.logits_y1, self.yp2, self.logits_y2 = self._single_conv(
                X=x[-1],
                mask=mask,
                scope='y1_y2_sel',
                size_input = size_input)
        elif config['model']['y1_sel'] == "double_conv":
            self.yp, self.logits_y1, self.yp2, self.logits_y2 = self._double_conv(
                X=x[-1],
                mask=mask,
                scope='y1_y2_sel',
                size_input = size_input)
        elif config['model']['y1_sel'] == "sym_double_conv":
            self.yp, self.logits_y1, self.yp2, self.logits_y2 = self._sym_double_conv(
                X=x[-1],
                mask=mask,
                scope='y1_y2_sel',
                size_input = size_input)
        elif config['model']['y1_sel'] == "cross":
            self.yp, self.logits_y1, self.yp2, self.logits_y2 = self._cross_sel(Q=q[-1],
                X=x[-1],
                mask=mask,
                scope='y1_y2_sel',
                size_input=size_input)
        elif config['model']['y1_sel'] == "all_layers":
            self.yp, self.logits_y1, self.yp2, self.logits_y2 = self._all_layers_sel(Q=q,
                X=x,
                mask=mask,
                scope='y1_y2_sel',
                size_input = size_input)
        else:
            # Computing outputs
            self.yp, self.logits_y1 = self._y_selection(
                                                  Q=q[-1-num_layers_post],
                                                  X=x[-1-num_layers_post],
                                                  mask=mask,
                                                  scope='y1_sel',
                                                  method=config['model']['y1_sel'],
                                                  size_input=size_input)
            self.yp2, self.logits_y2 = self._y_selection(
                                                   Q=q[-1],
                                                   X=x[-1],
                                                   mask=mask,
                                                   scope='y2_sel',
                                                   method=config['model']['y2_sel'],
                                                   y1_sel=self.yp,
                                                   size_input=size_input)

        if self.config['model']['sigmoid_loss']:
            epsilon = 1e-4
            self.prob_out = tf.log((self.yp+epsilon)/(1-self.yp+epsilon))
            self.Tensors_out['yp'] = self.yp
            self.Tensors_out['prob'] = self.prob_out
        else:
            if self.config['model']['max_answer_size'] >0:
                self.Start_Index, self.End_Index = self.get_y1_y2(self.yp, self.yp2)

            else:
                self.Start_Index = tf.argmax(self.yp, axis=-1)
                self.End_Index = tf.argmax(self.yp2, axis=-1)

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
            self.Start_Index = tf.argmax(self.yp, axis=-1)
            self.End_Index = tf.argmax(self.yp2, axis=-1)

    def _build_loss(self):
        """
            Defines the model's loss function.
        """
        # Calculate the loss for y1 and y2
        if self.config['model']['sigmoid_loss']:
            y_cp = tf.clip_by_value(tf.abs(self.yp - tf.cast(self.y_mask, tf.float32)),1e-10,1.0)
            ce_loss3 = -tf.reduce_mean(tf.reduce_sum(self.y_abs*tf.log(y_cp),1),0)
            # ce_loss3 = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.y3, logits=self.yp3), axis=1)
            self.loss = ce_loss3
            tf.summary.scalar('ce_loss3', tf.reduce_mean(ce_loss3))
        elif self.config['model']['single_loss']:
            pred = self.get_prob_matrix(self.yp, self.yp2)
            truth = self.get_prob_matrix(self.y, self.y2)
            self.loss = tf.reduce_mean(-tf.reduce_sum(truth*tf.log(tf.clip_by_value(pred,1e-10,1.0)), axis=1))

        elif (self.config['model']['y2_sel']=='linear_y2') or (self.config['model']['y2_sel']=='conv2') or (self.config['model']['y1_sel']=='single_conv') or (self.config['model']['y1_sel']=='double_conv') or (self.config['model']['y2_sel']=='direct2') or (self.config['model']['y2_sel']=='cross'):
            ce_loss = -tf.reduce_sum(self.y_corrected*tf.log(tf.clip_by_value(self.yp,1e-10,1.0)), axis=1)
            self.ce_loss2 = -tf.reduce_sum(self.y2_corrected*tf.log(tf.clip_by_value(self.yp2,1e-10,1.0)), axis=1)
        else:
            ce_loss = -tf.reduce_sum(self.y*tf.log(tf.clip_by_value(self.yp,1e-10,1.0)), axis=1)
            self.ce_loss2 = -tf.reduce_sum(self.y2*tf.log(tf.clip_by_value(self.yp2, 1e-10, 1.0)), axis=1)

        # Add a second loss
        if not self.config['model']['single_loss'] and not self.config['model']['sigmoid_loss']:
            self.loss = tf.reduce_mean(tf.add_n([ce_loss, self.ce_loss2]))
            tf.summary.scalar('ce_loss', tf.reduce_mean(ce_loss))
            tf.summary.scalar('ce_loss2', tf.reduce_mean(self.ce_loss2))
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
        encoder_pos = []
        q = []
        qc = []
        xc = []
        y1 = []
        y2 = []
        label_smoothing = self.config['train']['label_smoothing']


        def wordchar2id(text, dictionary, word_size):
            text_out =[]
            dict_len = len(dictionary)
            for word in text:
                if word in dictionary:
                    text_out.append(dictionary[word][0])
                else:
                    dict_len = dict_len + 1  # It starts at one
                    dictionary.update({word:[dict_len,len(word)]})
                    word_size[len(word)]=word_size[len(word)]+1
                    text_out.append(dict_len)
            return text_out, dictionary, word_size

        def wordsearch(word, known_or_unknown):
            if word in dataset['shared'][known_or_unknown]:
                return dataset['shared'][known_or_unknown][word]
            elif word.capitalize() in dataset['shared'][known_or_unknown]:
                return dataset['shared'][known_or_unknown][word.capitalize()]
            elif word.lower() in dataset['shared'][known_or_unknown]:
                return dataset['shared'][known_or_unknown][word.lower()]
            elif word.upper() in dataset['shared'][known_or_unknown]:
                return dataset['shared'][known_or_unknown][word.upper()]
            else:
                return 0

        def word2id(word):  # to convert a word to its respective id
            ID = wordsearch(word, 'known_word2idx')
            if ID != 0:  # if it was found
                return ID + self.WVs
            ID = wordsearch(word, 'unk_word2idx')
            if ID != 0:  # if it was found
                return ID
            # if it was not found in any
            return 1  # unknown word

        def char2id(char):  # to convert a char to its respective id
            def charsearch(char, known_or_unknown):
                if char in dataset['shared'][known_or_unknown]:
                    return dataset['shared'][known_or_unknown][char]
                else:
                    return 0

            ID = charsearch(char, 'known_char2idx')
            if ID != 0:  # if it was found
                return ID + len(dataset['shared']['emb_mat_unk_chars'])
            ID = charsearch(char, 'unk_char2idx')
            if ID != 0 or char=="-NULL-":  # if it was found
                return ID
            # if it was not found in any
            return 1  # unknown char

        def gauss_kern(size, mean, smoothing):
            """ Returns a normalized 1D gauss kernel array """
            mean = mean[0]
            x = np.mgrid[-mean:size-mean]
            summ = np.sum(np.exp(-(x**2)))
            g = np.exp(-x**2)
            g[mean] = 0
            g = g / (g.sum()/(1-smoothing))
            g[mean] = smoothing
            return g

        # Padding for passages, questions and answers.
        # The answers output are (1-label_smoothing)/len(x)
        def padding(seq, max_size=None, toten_ID=None):  # for padding a batch
            seq_len = [len(seq[i]) for i in range(len(seq))]
            if max_size is None:
                max_size = max(seq_len)
            if toten_ID is None:
                new_seq = [np.concatenate([np.array(seq[i]), np.zeros([max_size-len(seq[i])])], axis=0) for i in range(len(seq))]
            else:
                max_size = max_size+len(toten_ID)
                new_seq = [np.concatenate([np.array(seq[i]), np.array(toten_ID), np.zeros([max_size-len(seq[i])])], axis=0) for i in range(len(seq))]
            return np.int_(new_seq)

        def padding_answer(seq, y1, y2, label_smoothing, max_size=None):
            seq_len = [len(seq[i]) for i in range(len(seq))]
            if max_size is None:
                max_size = max(seq_len)
            new_seq = [np.concatenate([np.array(seq[i]), np.zeros([max_size-len(seq[i])])], axis=0) for i in range(len(seq))]
            if self.config['train']['gaussian_smoothing']:
                y1_new = [np.concatenate([gauss_kern(size=seq_len[i], mean=y1[i], smoothing=label_smoothing), np.zeros([max_size-len(seq[i])])], axis=0) for i in range(len(seq))]
                y2_new = [np.concatenate([gauss_kern(size=seq_len[i], mean=y2[i], smoothing=label_smoothing), np.zeros([max_size-len(seq[i])])], axis=0) for i in range(len(seq))]
            else:
                new_seq_y = [np.concatenate([np.ones(seq_len[i])*(1.0-label_smoothing)/seq_len[i], np.zeros([max_size-len(seq[i])])], axis=0) for i in range(len(seq))]
                y1_new = new_seq_y
                y2_new = np.copy(new_seq_y)
                for i in range(self.Bs):
                    y1_new[i][y1[i]] += label_smoothing
                    y2_new[i][y2[i]] += label_smoothing
            return np.int_(new_seq), y1_new, y2_new

        def padding_y3(seq, y1, y2, max_size=None):
            seq_len = [len(seq[i]) for i in range(len(seq))]
            if max_size is None:
                max_size = max(seq_len)
            ans_size = [(y2[i][0]-y1[i][0]+1) for i in range(len(seq))]
            no_ans_size = [(seq_len[i]-ans_size[i]) for i in range(len(seq))]
            out =  []
            for i in range(len(seq)):
                if y1[i][0]>0:
                    ans_size[i] = ans_size[i] + 1
                    no_ans_size[i] = no_ans_size[i] - 1
                if y2[i][0]<seq_len[i]-1:
                    ans_size[i] = ans_size[i] + 1
                    no_ans_size[i] = no_ans_size[i] - 1
                no_ans_size[i] = no_ans_size[i] if no_ans_size[i]>0 else 0.5
                ans_size[i] = ans_size[i] if ans_size[i]>0 else 0.5
                y3_i = np.concatenate([-5/3*np.ones(y1[i][0])/no_ans_size[i], 1/3*np.ones(y2[i][0]-y1[i][0]+1)/ans_size[i], -5/3*np.ones(seq_len[i]-y2[i][0]-1)/no_ans_size[i], np.zeros(max_size-seq_len[i])], axis=0)
                if y1[i][0]>0:
                    y3_i[y1[i][0]-1] = -1/3/ans_size[i]
                if y2[i][0]<seq_len[i]-1:
                    y3_i[y2[i][0]+1] = -1/3/ans_size[i]
                out.append(y3_i)
            #y3 = [np.concatenate([-answer_passage_ratio*np.ones(y1[i][0]), np.ones(y2[i][0]-y1[i][0]+1), -answer_passage_ratio*np.ones(seq_len[i]-y2[i][0]-1), np.zeros(max_size-seq_len[i])], axis=0) for i in range(len(seq))]
            return out

        def padding_chars(seq, max_size_sentence, max_size=None):  # for padding a batch
            seq_len = [len(seq[i][j]) for i in range(len(seq)) for j in range(len(seq[i]))]
            if max_size is None:
                max_size = max(seq_len)
            # First add padding in each character and later in each sentence.
            for i in range(len(seq)):
                for j in range(len(seq[i])):
                    seq[i][j] = np.concatenate([np.array(seq[i][j]), np.zeros([max_size-len(seq[i][j])])], axis=0)
                seq[i] = np.concatenate([np.array(seq[i]), np.zeros([max_size_sentence-len(seq[i]), max_size])], axis=0)
            return np.int_(seq)

        def compute_enc_pos(len_sent, sent_step):  # for padding a batch
            enc_pos = []
            for i in range(len(len_sent)):
                delta = len(enc_pos)+i*sent_step
                enc_pos = enc_pos + list(range(delta,delta+len_sent[i]))
            return enc_pos

        def compute_mask_pos(len_sent, max_sent):  # for padding a batch
            enc_pos = []
            len_after = max_sent
            len_before = 0
            final_mask = np.zeros([0,len_after])
            for i in range(len(len_sent)):
                max_delta = self.weights[len_sent[i]]
                len_after = len_after - len_sent[i]
                mask = np.concatenate([max_delta*np.ones([len_before]),np.zeros([len_sent[i]]),max_delta*np.ones([len_after])])
                mask = np.tile(mask, (len_sent[i],1))
                final_mask = np.concatenate([final_mask,mask],0)
                len_before = len_before + len_sent[i]
            final_mask = np.concatenate([final_mask, np.zeros([len_after,max_sent])],0)
            return final_mask

        # Convert every word to its respective id
        if self.config['model']['char_embedding']:
            words_dict = {}
            word_size_counter = [0]*(self.config['pre']['max_word_size']+10)
        for i in batch_idxs:
            word_count = {}
            word_list_x = {}
            word_list_q = {}
            rxi = dataset['data']['*x'][i]
            yi = dataset['data']['y'][i]
            question = dataset['data']['q'][i]
            par = dataset['shared']['x'][rxi[0]][rxi[1]]
            len_sent = dataset['shared']['len_sent'][rxi[0]][rxi[1]]
            enc_pos_i = compute_enc_pos(len_sent, self.config['model']['sentence_skip_steps'])
            xi = list(map(
                word2id,
                par))
            qi = list(map(
                word2id,
                question))
            for j in range(len(par)): #Find all UNK words in X
                if  xi[j]==1:
                    if par[j] in word_count:
                        word_count[par[j]]+=1
                        word_list_x[par[j]].append(j)
                    else:
                        word_count[par[j]]=1
                        word_list_x[par[j]]=[j]
            for j in range(len(question)): #Find all UNK words in Q
                if  qi[j]==1:
                    if question[j] in word_list_q: #If not in word_list
                        word_count[question[j]]+=1
                        word_list_q[question[j]].append(j)
                    elif question[j] in word_count: #if in word_count (because of X), but not in word_q
                        word_count[question[j]]+=1
                        word_list_q[question[j]]=[j]
                    else: #if neither in word_list nor in word_count_q
                        word_count[question[j]]=1
                        word_list_q[question[j]]=[j]
            ordered_words = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
            used_unk = set()
            for j in range(len(ordered_words)):
                if j<(self.config['pre']['number_of_unk']-1):
                    while(j>=len(used_unk)):
                        unk = randint(2, self.config['pre']['number_of_unk'])
                        used_unk.add(unk)
                    if ordered_words[j][0] in word_list_x: #If the UNK word is in x
                        for k in word_list_x[ordered_words[j][0]]: #To guarantee that random UNK is the same for Q and X
                            xi[k] = unk
                    if ordered_words[j][0] in word_list_q: #If the UNK word is in q
                        for k in word_list_q[ordered_words[j][0]]:
                            xi[k] = unk


            if self.config['model']['char_embedding']:
                qic, words_dict, word_size_counter = wordchar2id(dataset['data']['q'][i], words_dict, word_size_counter)
                xic, words_dict, word_size_counter = wordchar2id(dataset['shared']['x'][rxi[0]][rxi[1]], words_dict, word_size_counter)
                #######################################
                #qic = []  # compute char2id for question
                #for j in dataset['data']['q'][i]:
                #    qic.append(list(map(char2id,j)))
                #xic = []  # Compute char2id for passage
                #for j in dataset['shared']['x'][rxi[0]][rxi[1]]:
                #    xic.append(list(map(char2id,j)))
                qc.append(qic)
                xc.append(xic)
            q.append(qi)
            x.append(xi)
            encoder_pos.append(enc_pos_i)
            # Get all the first indices in the sequence
            y1.append([y[0] for y in yi])
            # Get all the second indices... and correct for -1
            y2.append([y[1]-1 for y in yi])
        self.answer = [y1, y2]
        # Padding
        toten_ID_begin = 1 + self.config['pre']['number_of_unk']
        toten_ID_end = toten_ID_begin + self.config['model']['number_of_totens']
        if self.config['train']['check_available_memory']:
            x, y1_new, y2_new = padding_answer(x, y1, y2,
                                   label_smoothing=label_smoothing,
                                   max_size=self.config['pre']['max_paragraph_size'])
            encoder_pos = padding(encoder_pos, max_size=self.config['pre']['max_question_size'])
            q = padding(q, max_size=self.config['pre']['max_question_size'], toten_ID = list(range(toten_ID_begin, toten_ID_end)))
        else:
            if self.config['model']['sigmoid_loss']:
                y3 = padding_y3(x, y1, y2)
            x, y1_new, y2_new = padding_answer(x, y1, y2, label_smoothing=label_smoothing)
            encoder_pos = padding(encoder_pos)
            q = padding(q, toten_ID = list(range(toten_ID_begin, toten_ID_end)))
            masks_sentence = []
        for i in batch_idxs:
            rxi = dataset['data']['*x'][i]
            len_sent = dataset['shared']['len_sent'][rxi[0]][rxi[1]]
            masks_sentence.append(compute_mask_pos(len_sent, len(x[0])))
        if self.config['model']['char_embedding']:  # Padding chars
            ordered_words = sorted(words_dict.items(), key=lambda x: x[1][1])
            longest_word_size = ordered_words[-1][1][1]
            number_of_words = len(ordered_words)
            mapping=[None]*(1+len(ordered_words))
            computational_cost = 1e10
            computational_cost_new = 1e9
            long_word_start = False
            long_words_list = []
            short_words_list = []
            index = -1
            size_short = 0
            convolution_size = self.config['model']['char_convolution_size']
            while (computational_cost>computational_cost_new): # Find optimal threshold for short and long words
                index = index + 1
                size_short = size_short+word_size_counter[index]
                computational_cost = computational_cost_new
                computational_cost_new = max(1,1+(index+1)-convolution_size)*(size_short)+(longest_word_size-convolution_size+1)*(number_of_words-size_short)
            index=max(index,convolution_size)
            short_words_list.append([0])  # index 0 is null word
            for i in range(number_of_words):
                mapping[ordered_words[i][1][0]]=i+1
                if len(ordered_words[i][0])>index:
                    long_words_list.append(list(map(char2id, ordered_words[i][0])))
                else:
                    short_words_list.append(list(map(char2id, ordered_words[i][0])))
            for i in range(len(batch_idxs)):
                xc[i] = list(map(lambda y: mapping[y], xc[i]))
                qc[i] = list(map(lambda y: mapping[y], qc[i]))
            xc = padding(xc)
            qc = padding(qc)
            short_words_list = padding(short_words_list)
            long_words_list = padding(long_words_list)
            feed_dict[self.xc] = xc
            feed_dict[self.qc] = qc
            feed_dict[self.short_words_char] = short_words_list
            feed_dict[self.long_words_char] = long_words_list
        if self.config['model']['sigmoid_loss']:
            feed_dict[self.y] = y3
        else:
            feed_dict[self.y] = y1_new
            feed_dict[self.y2] = y2_new


        # cq = np.zeros([self.Bs, self.Qs, self.Ws], dtype='int32')
        feed_dict[self.x] = x
        feed_dict[self.q] = q
        feed_dict[self.encoder_pos] = encoder_pos
        feed_dict[self.mask_sentence] = masks_sentence
        feed_dict[self.is_training] = is_training
        if self.config['pre']['use_glove_for_unk']:
            feed_dict[self.new_emb_mat] = dataset['shared']['emb_mat_known_words']
        if self.config['model']['pre_trained_char']:
            feed_dict[self.new_char_emb_mat] = dataset['shared']['emb_mat_known_chars']
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
