import csv
import json
import math
import numpy as np
import os
import pandas as pd
import pickle
import random
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
from tqdm import tqdm
from utils import plot, send_mail, EM_and_F1
import pdb
from collections import deque
from tensorflow.contrib.tensorboard.plugins import projector

from my_tf import optimize_loss
from read_data import get_batch_idxs, read_data, update_config


class Char_Embedder(object):
    def __init__(self, config):
        # Read the data
        self.data = read_data(config, 'train', ref=False, data_filter=True)
        config = update_config(config, self.data)
        self.config = config
        self.char_vocabulary_size = len(self.data['shared']['emb_mat_known_chars']) + len(self.data['shared']['emb_mat_unk_chars'])
        self.word_vocabulary_size = len(self.config['model']['emb_mat_unk_words']) + len(self.data['shared']['emb_mat_known_words'])
        self.word_embedding_size = 100
        self.char_embedding_size = 8
        # Run pre-processing
        self.pre_process(2, 2)

        num_sampled = 64 # number of negative examples nce loss
        self.Bs = 100  # batch size
        # Index of each word
        self.input = tf.placeholder('int32', [self.Bs, None], name='x')
        # The target is the composed of 1 context word
        self.target = tf.placeholder('int32', [self.Bs, 1], name='y')
        # Define a embedding matrix with GloVe vectors
        self.glove_emb_mat = tf.placeholder(tf.float32,
                                          [None, self.word_embedding_size],
                                          name='new_emb_mat')
        # Define a session
        self.sess = tf.Session()
        # Define a global step
        self.global_step = tf.Variable(0, trainable=False)
        # Define a initializer
        self.initializer = tf.contrib.layers.xavier_initializer(
                                                        uniform=False,
                                                        seed=None,
                                                        dtype=tf.float32)
        # Define the word embedding matrix
        self.target_emb_mat = tf.get_variable(
            "word_emb_mat",
            dtype=tf.float32,
            initializer=self.config['model']['emb_mat_unk_words'])
        self.target_emb_mat = tf.concat([self.target_emb_mat,
                                        self.glove_emb_mat], axis=0)
        # Define the char embedding matrix
        self.char_emb_mat = tf.Variable(
            tf.random_uniform([self.char_vocabulary_size, self.char_embedding_size], -1.0, 1.0),
            name='char_emb_mat')

        # The target is given by a word
        target_embedding = tf.nn.embedding_lookup(self.target_emb_mat,
                                                  self.target)
        # The input is given by a set of chars
        input_embedding = tf.nn.embedding_lookup(self.char_emb_mat,
                                                 self.input)
        # Computation of equivalent word embedding
        Ac = self.char2word_embedding(Ac=input_embedding)

        nce_weights = tf.Variable(
            tf.truncated_normal([self.word_vocabulary_size, self.word_embedding_size],
                                stddev=1.0 / math.sqrt(self.word_embedding_size)))
        nce_biases = tf.Variable(tf.zeros([self.word_vocabulary_size]))

        self.loss = tf.reduce_mean(
            tf.nn.nce_loss(weights=nce_weights,
                           biases=nce_biases,
                           labels=self.target,
                           inputs=tf.reshape(Ac, [self.Bs, self.word_embedding_size]),
                           num_sampled=num_sampled,
                           num_classes=self.word_vocabulary_size))

        self.train_step = optimize_loss(self.loss,
                                        global_step=self.global_step,
                                        optimizer='Adagrad',
                                        summaries=["gradients"],
                                        learning_rate=0.01)

        loss_summary = tf.summary.scalar('loss', tf.reduce_mean(self.loss))
        self.summary = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter('char_embedding',
                                            self.sess.graph)
        # Add projector to show embedding on Tensorboard
        proj_config = projector.ProjectorConfig()
        embedding = proj_config.embeddings.add()
        embedding.tensor_name = self.char_emb_mat.name
        # Create metadata with char ids
        with open('char_embedding/metadata.tsv', 'w') as f:  # Just use 'w' mode in 3.x
            w = csv.writer(f, delimiter='\t')
            w.writerow(['Name', 'ID'])
            for key, value in self.data['shared']['unk_char2idx'].items():
                w.writerow([self.char2id(value), value])
            for key, value in self.data['shared']['known_char2idx'].items():
                w.writerow([self.char2id(value), value])
        # Link this tensor to its metadata file (e.g. labels).
        embedding.metadata_path = os.path.join('metadata.tsv')
        # Saves a configuration file that TensorBoard will read during startup.
        projector.visualize_embeddings(self.writer, proj_config)

        # Initialize all variables
        self.sess.run(tf.global_variables_initializer())
        # Add ops to save and restore the pre-trained chars
        self.saver = tf.train.Saver({"Pre_trained_chars": self.char_emb_mat},
                                    max_to_keep=100)

    def train(self):
        feed_dict = self.get_feed_dict(100)
        _, summary = self.sess.run([self.train_step, self.summary], feed_dict=feed_dict)
        self.writer.add_summary(summary, self.sess.run(self.global_step))

    def save(self):
        self.saver.save(self.sess, 'char_embedding/model.ckpt')

    def char2id(self, char):  # to convert a char to its respective id
        def charsearch(char, known_or_unknown):
            if char in self.data['shared'][known_or_unknown]:
                return self.data['shared'][known_or_unknown][char]
            elif char.capitalize() in self.data['shared'][known_or_unknown]:
                return self.data['shared'][known_or_unknown][char.capitalize()]
            elif char.lower() in self.data['shared'][known_or_unknown]:
                return self.data['shared'][known_or_unknown][char.lower()]
            elif char.upper() in self.data['shared'][known_or_unknown]:
                return self.data['shared'][known_or_unknown][char.upper()]
            else:
                return 0

        ID = charsearch(char, 'known_char2idx')
        if ID != 0:  # if it was found
            return ID + len(self.data['shared']['emb_mat_unk_chars'])
        ID = charsearch(char, 'unk_char2idx')
        if ID != 0:  # if it was found
            return ID
        # if it was not found in any
        return 1  # unknown char

    def word2id(self, word):  # to convert a word to its respective id
        def wordsearch(word, known_or_unknown):
            if word in self.data['shared'][known_or_unknown]:
                return self.data['shared'][known_or_unknown][word]
            elif word.capitalize() in self.data['shared'][known_or_unknown]:
                return self.data['shared'][known_or_unknown][word.capitalize()]
            elif word.lower() in self.data['shared'][known_or_unknown]:
                return self.data['shared'][known_or_unknown][word.lower()]
            elif word.upper() in self.data['shared'][known_or_unknown]:
                return self.data['shared'][known_or_unknown][word.upper()]
            else:
                return 0

        ID = wordsearch(word, 'known_word2idx')
        if ID != 0:  # if it was found
            return ID + len(self.data['shared']['emb_mat_unk_words'])
        ID = wordsearch(word, 'unk_word2idx')
        if ID != 0:  # if it was found
            return ID
        # if it was not found in any
        return 1  # unknown word

    def pad_chars(self, seq, max_size=None):  # for padding a batch
        seq_len = [len(seq[i]) for i in range(len(seq))]
        if max_size is None:
            max_size = max(seq_len)
        # Add padding to each word
        for i in range(len(seq)):
            seq[i] = np.concatenate([np.array(seq[i]), np.zeros([max_size-len(seq[i])])], axis=0)
        return np.int_(seq)

    def pre_process(self, skip_window, num_skips):
        span = 2 * skip_window + 1  # [ skip_window target skip_window ]
        buffer = deque(maxlen=span)

        self.word = []
        self.labels = []

        for i in tqdm(range(len(self.data['shared']['x']))):
            for j in range(len(self.data['shared']['x'][i])):
                clean_data = list(filter(None, self.data['shared']['x'][i][j]))
                for k in clean_data:
                    self.word.append(list(map(self.char2id, k)))
                    self.labels.append(self.word2id(k))
                # data_index = 0
                # while data_index + span <= len(self.data['shared']['x'][i][j]):
                #     clean_data = list(filter(None, self.data['shared']['x'][i][j]))
                #     buffer.extend(clean_data[data_index:data_index + span])
                #     target = skip_window  # target label at the center of the buffer
                #     targets_to_avoid = [skip_window]
                #     self.word.append(list(map(self.char2id, buffer[skip_window])))
                #     self.labels.append(self.word2id(buffer[skip_window]))
                #     for k in range(num_skips):
                #         while target in targets_to_avoid:
                #             target = random.randint(0, span - 1)
                #         targets_to_avoid.append(target)
                #         self.word.append(list(map(self.char2id, buffer[skip_window])))
                #         self.labels.append(self.word2id(buffer[target]))
                #     data_index += span

        for i in tqdm(range(len(self.data['data']['q']))):
            for j in range(len(self.data['data']['q'][i])):
                clean_data = list(filter(None, self.data['data']['q'][i][j]))
                for k in clean_data:
                    self.word.append(list(map(self.char2id, k)))
                    self.labels.append(self.word2id(k))
                # data_index = 0
                # while data_index + span <= len(self.data['data']['q'][i][j]):
                #     clean_data = list(filter(None, self.data['data']['q'][i][j]))
                #     buffer.extend(clean_data[data_index:data_index + span])
                #     target = skip_window  # target label at the center of the buffer
                #     targets_to_avoid = [skip_window]
                #     self.word.append(list(map(self.char2id, buffer[skip_window])))
                #     self.labels.append(self.word2id(buffer[skip_window]))
                #     for k in range(num_skips):
                #         while target in targets_to_avoid:
                #             target = random.randint(0, span - 1)
                #         targets_to_avoid.append(target)
                #         self.word.append(list(map(self.char2id, buffer[skip_window])))
                #         self.labels.append(self.word2id(buffer[target]))
                #     data_index += span

        self.word = self.pad_chars(self.word)
        self.labels = np.array(self.labels).astype(int)


    def char2word_embedding(self, Ac):
        Ac = tf.reshape(Ac, [self.Bs, -1, self.char_embedding_size, 1])
        A_word_convolution = tf.layers.conv2d(inputs=Ac,
                                              filters=self.word_embedding_size,
                                              kernel_size=[self.config['model']['char_convolution_size'], 1],
                                              strides=1,
                                              padding="same",
                                              use_bias=True,
                                              activation=tf.tanh)
        char_embedded_word = tf.reduce_max(tf.reshape(A_word_convolution, [self.Bs, 1, -1, self.word_embedding_size]), axis=2)  # Reduce all info to a vector
        return tf.reshape(char_embedded_word, [self.Bs, self.word_embedding_size])

    def get_feed_dict(self, batch_size):
        indices = np.random.randint(len(self.word), size=(batch_size))
        feed_dict = {}
        feed_dict['x:0'] = self.word[indices]
        feed_dict['y:0'] = np.expand_dims(self.labels[indices], axis=1)
        feed_dict[self.glove_emb_mat] = self.data['shared']['emb_mat_known_words']
        return feed_dict

if __name__ == '__main__':
    with open('config.json') as json_data_file:
        config = json.load(json_data_file)
    Embedder = Char_Embedder(config)
    for i in tqdm(range(100000)):
        Embedder.train()
        if i % 1000 == 0:
            Embedder.save()
