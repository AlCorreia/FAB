import argparse
from read_data import prepro_each
import sys
import tensorflow as tf


def main(_):
    prepro_each(args=FLAGS, data_type='train')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--learning_rate', '-l',
        type=float,
        default=0.1,
        help='Initial learning rate.'
    )

    parser.add_argument(
        '--steps', '-s',
        type=int,
        default=1000,
        help='The number of iterations to run'
    )

    parser.add_argument(
        '--source_dir', '-d',
        type=str,
        default='./',
        help='''Directory used to store the model and load the data.'''
    )

    # I only downloaded the smaller version from https://nlp.stanford.edu/projects/glove/
    # I assumed the files are stored in a folder in the same directory as the
    # python files, hence the default argument
    parser.add_argument(
        '--glove_dir',
        type=str,
        default='./glove.6B',
        help='''Directory of the pre-trained glove vectors.'''
    )

    parser.add_argument(
        '--glove_corpus',
        type=str,
        default='6B',
        help='''GloVe Corpus used to embed the words.'''
    )

    parser.add_argument(
        '--glove_vec_size',
        type=str,
        default='300',
        help='''Number of dimensions of each word vector.'''
    )

    parser.add_argument(
        '--target_dir',
        type=str,
        default='./',
        help='''The directory where to save the pre=processed data.'''
    )

    parser.add_argument(
        '--arch', '-a',
        type=str,
        default='512,256,128',
        help=''' The neural net architecture. The number of neurons must be passed. '''
    )

    parser.add_argument(
        '--nonlinearity', '-f',
        type=str,
        default='tf.nn.relu',
        help=''' The neural net activation function. '''
    )

    FLAGS, unparsed = parser.parse_known_args()
    FLAGS.arch = [int(item) for item in FLAGS.arch.split(',')]
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
