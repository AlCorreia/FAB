#This code is being implemented for the Attention Model
#Later it is gonna be integrated in the model.py code
import tensorflow as tf
import numpy as np
import pdb
#from __future__ import print_function

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data


Base_Frequency = 10000
number_of_words=3
embedding_size=10
size_of_vocabulary=100
batch_size=3
word_zeros=np.zeros([1,embedding_size])
embedding_dict=np.array(np.concatenate([word_zeros,np.random.rand(size_of_vocabulary,embedding_size)],axis=0), dtype=np.float32)

paragraphs=[np.random.randint(1,100, size=np.random.randint(3,8)) for i in range(batch_size)]
questions = [np.random.randint(1,100, size=i) for i in range(2,7,2)]

def padding(seq):
    max_size=max([len(seq[i]) for i in  range(len(seq))])
    new_seq=[np.concatenate([np.array(seq[i]), np.zeros([max_size-len(seq[i])])],axis=0)  for i in range(len(seq))]
    return new_seq

paragraphs=padding(paragraphs)
questions=padding(questions)

def length(sequence):
    used = tf.sign(tf.reduce_max(tf.abs(sequence), 2))
    length = tf.reduce_sum(used, 1)
    length = tf.cast(length, tf.int32)
    return length

x = tf.placeholder("int32", [batch_size, None]) #embedding size =1
q = tf.placeholder("int32", [batch_size, None]) #embedding size =1
x_input=tf.nn.embedding_lookup(embedding_dict,x)
q_input=tf.nn.embedding_lookup(embedding_dict,q)
x_mask = tf.cast(tf.sign(x),tf.float32)
q_mask = tf.cast(tf.sign(q),tf.float32)

#Mask matrices
mask_matrix_xx=tf.sign(tf.matmul(tf.expand_dims(x,-1),tf.expand_dims(x,1)))
mask_matrix_qq=tf.sign(tf.matmul(tf.expand_dims(q,-1),tf.expand_dims(q,1)))
mask_matrix_xq=tf.sign(tf.matmul(tf.expand_dims(x,-1),tf.expand_dims(q,1)))

#Implementing encoder
size_x = tf.shape(x)[-1]
size_q = tf.shape(q)[-1]
pos = tf.cast(tf.expand_dims(tf.range(size_x), 1),tf.float32)
freq = tf.divide(tf.range(embedding_size/2),embedding_size/2-1)
freq_PG = tf.expand_dims(tf.pow(1/Base_Frequency,freq),0)

encoder_sin = tf.sin(tf.matmul(pos,freq_PG))
encoder_cos = tf.cos(tf.matmul(pos,freq_PG))

encoder_x = tf.concat([encoder_sin,encoder_cos], axis = 1)

encoder_q = tf.slice(encoder_x,[0,0],[size_q,embedding_size])

#Encoding x_input
x_encoded = tf.add(x_input, encoder_x)
q_encoded = tf.add(q_input, encoder_q)


# Launch the graph in a session.
# Evaluate the tensor `c`.
with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    tf.global_variables_initializer().run()
    print(sess.run([mask_matrix_xx],feed_dict={x : paragraphs, q: questions}))
    pdb.set_trace()
    a=1
		
