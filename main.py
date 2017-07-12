import argparse
#import configparser
from read_data import prepro_each, read_data, get_batch_idxs
import sys
import json
import os
import pdb
#import tensorflow as tf

def main(config):
    if config['pre']['run']:
        prepro_each(config=config, data_type='train',out_name='train') #to preprocess the train data
        prepro_each(config=config, data_type='dev',out_name='dev') #to preprocess  the dev data
    if config['model']['run']:
        data_train = read_data(config,'dev',0)
        batch_idxs=get_batch_idxs(config,data_train)
        pdb.set_trace()
if __name__ == '__main__':

    with open('config.json') as json_data_file:
        config = json.load(json_data_file)
    main(config)
