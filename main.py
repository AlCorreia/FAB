import argparse
#import configparser
from read_data import prepro_each, read_data, get_batch_idxs, update_config
import sys
import json
import os
import pdb
from tqdm import tqdm
#import tensorflow as tf

from model import Model


def main(config):
    if config['pre']['run']:
        prepro_each(config=config, data_type='train',out_name='train') #to preprocess the train data
        prepro_each(config=config, data_type='dev',out_name='dev') #to preprocess  the dev data
    if config['model']['run']:
        data = read_data(config,'train',ref=False)
        config = update_config(config, data) #update config with max_word_size, max_passage_size, embedded_vector
        # Create an instance of the model
        model = Model(config)
        # Train the model
        for i in tqdm(range(10000)):
            batch_idxs = get_batch_idxs(config, data)
            model.train(batch_idxs, data)


if __name__ == '__main__':

    with open('config.json') as json_data_file:
        config = json.load(json_data_file)
    main(config)
