import argparse
#import configparser
from read_data import prepro_each, read_data, get_batch_idxs, update_config
import sys
from time import sleep
import json
import os
import pdb
import math
from tqdm import tqdm
from utils import plot, send_mail
import numpy as np
import tensorflow as tf

from model import Model
from pdf_creator import create_pdf


def main(config):
    if config['pre']['run']:
        prepro_each(config=config, data_type='train', out_name='train') # to preprocess the train data
        prepro_each(config=config, data_type='dev', out_name='dev') # to preprocess the dev data
    if config['model']['run']:
        data = read_data(config, 'train', data_filter=True)
        data_dev = read_data(config, 'dev', data_filter=True, data_train=data)
        # update config with max_word_size, max_passage_size, embedded_vector
        config = update_config(config, data)
        # Create an instance of the model
        model = Model(config)
        # Train the model
    if config['train']['train']:
        EM_dev_plot = []
        F1_dev_plot = []
        EM_train_plot = []
        F1_train_plot = []
        steps = []
        global_step_init = 0 if not config['model']['load_checkpoint'] else model.sess.run([model.global_step][0]+1)  # to get global step
        for i in tqdm(range(global_step_init, config['train']['steps'])):
            batch_idxs = get_batch_idxs(config, data)
            model.train(batch_idxs, data)
            # every n steps check F1 and EM, based on dev dataset
            if i % config['train']['steps_to_save'] == 0:
                EM_dev, F1_dev, y1_correct_dev, y2_correct_dev, y2_greater_y1_correct = evaluate(config, model, data_dev)
                # Compute EM and F1 train averaging the last 500 steps
                EM_train, F1_train = [np.mean(model.EM_train[-500:]), np.mean(model.F1_train[-500:])]
                model.EM_train = []
                model.F1_train = []
                steps.append(i)
                EM_dev_plot.append(EM_dev)
                EM_train_plot.append(EM_train)
                F1_dev_plot.append(F1_dev)
                F1_train_plot.append(F1_train)
                # To plot the EM and F1 curve
                plot(X=steps,
                     EM=[EM_train_plot, EM_dev_plot],
                     F1=[F1_train_plot, F1_dev_plot],
                     save_dir='./plots/plot.png')
                summary_EM = tf.Summary(
                    value=[tf.Summary.Value(tag='EM',
                                            simple_value=EM_dev)])
                summary_F1 = tf.Summary(
                    value=[tf.Summary.Value(tag='F1',
                                            simple_value=F1_dev)])
                summary_y1_correct = tf.Summary(
                    value=[tf.Summary.Value(tag='y1_cor',
                                            simple_value=y1_correct_dev)])
                summary_y2_correct = tf.Summary(
                    value=[tf.Summary.Value(tag='y2_cor',
                                            simple_value=y2_correct_dev)])
                summary_y2_correct = tf.Summary(
                    value=[tf.Summary.Value(tag='y2>=y1_cor',
                                            simple_value=y2_greater_y1_correct)])
                model.dev_writer.add_summary(summary_F1, i)
                model.dev_writer.add_summary(summary_EM, i)
                model.dev_writer.add_summary(summary_y1_correct, i)
                model.dev_writer.add_summary(summary_y2_correct, i)
                last_info = '\nF1:'+str(F1_dev)+' EM:'+str(EM_dev)+' y1:'+str(y1_correct_dev)+' y2:'+str(y2_correct_dev)+' y2>=y1:'+str(y2_greater_y1_correct)+'\n'
                # TODO Make this print more readable than now
                print(last_info)
            if i % config['train']['steps_to_email'] == 0 and i > 0:
                valid_idxs = data_dev['valid_idxs']
                ids_to_pdf = [valid_idxs[0:config['train']['batch_size']], valid_idxs[60*config['train']['batch_size']: 61*config['train']['batch_size']], valid_idxs[-config['train']['batch_size']-1: -1]]
                Start_Index, End_Index, prob = [], [], []
                for i in ids_to_pdf:
                    s, e, p = model.evaluate_all_dev(i, data_dev)
                    Start_Index.append(s)
                    End_Index.append(e)
                    prob.append(p)
                Start_Index = [inner for outer in Start_Index for inner in outer]
                End_Index = [inner for outer in End_Index for inner in outer]
                prob = [inner for outer in prob for inner in outer]
                ids_to_pdf = sum(ids_to_pdf, [])
                create_pdf(config, ids_to_pdf, Start_Index, End_Index, prob, data_dev=data_dev)
                send_mail(attach_dir=['./config.json','./plots/plot.png', './plots/answers.pdf'], subject=config['model']['name'], body=last_info)
    # To check the exact match and F1 of the model for dev
    if config['model']['evaluate_dev']:
        EM_dev, F1_dev = evaluate(config, model, data_dev)
        print('\nF1:'+str(F1_dev)+' EM:'+str(EM_dev)+'\n')


def evaluate(config, model, data_dev):
    """ To check the exact match and F1 of the model """
    model.EM_dev = []
    model.F1_dev = []
    model.y1_correct_dev = []
    model.y2_correct_dev = []
    model.y2_greater_y1_correct=[]
    valid_idxs = data_dev['valid_idxs']
    for i in tqdm(range(math.floor(
            len(data_dev['valid_idxs'])/config['train']['batch_size'])),
            file=sys.stdout): # this file = sys.stdout is to only to allow the print function
        init = (i) * config['train']['batch_size']
        end = (i+1)*config['train']['batch_size']
        batch_idxs = valid_idxs[init:end]
        model.evaluate(batch_idxs, data_dev)
    return [sum(model.EM_dev)/len(model.EM_dev), sum(model.F1_dev)/len(model.F1_dev), sum(model.y1_correct_dev)/len(model.y1_correct_dev), sum(model.y2_correct_dev)/len(model.y2_correct_dev), sum(model.y2_greater_y1_correct)/len(model.y2_greater_y1_correct)]


if __name__ == '__main__':
    with open('config.json') as json_data_file:
        config = json.load(json_data_file)
    main(config)
