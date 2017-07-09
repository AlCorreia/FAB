import argparse
import configparser
from read_data import prepro_each
import sys
#import tensorflow as tf


def main(config):
    prepro_each(config=config, data_type='train')


if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read('config.ini')

    main(config)
