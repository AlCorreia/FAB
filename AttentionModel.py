#This code is being implemented for the Attention Model
#Later it is gonna be integrated in the model.py code
import tensorflow as tf
import numpy as np
import pdb
#from __future__ import print_function

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data



VERY_LOW_NUMBER = -1e30

Base_Frequency = 10000
number_of_words=3
embedding_size=512 #MUST BE EVEN FOR ENCODER
size_of_vocabulary=100
batch_size=3
multihead_size = 8
word_zeros=np.zeros([1,embedding_size])
embedding_dict=np.array(np.concatenate([word_zeros,np.random.rand(size_of_vocabulary,embedding_size)],axis=0), dtype=np.float32)

def encoder (x,q):
	#Compute the number of words in passage and question
	size_x = tf.shape(x)[-2]
	size_q = tf.shape(q)[-2]
	#Create a row vector with range(0,n) = [0,1,2,n-1], where n is the greatest size between x and q.
	pos = tf.cast(tf.expand_dims(tf.range(tf.cond(tf.greater(size_x,size_q),lambda: size_x, lambda: size_q)), 1),tf.float32)
	#Create a vector with all the exponents
	exponents = tf.divide(tf.range(embedding_size/2),embedding_size/2-1)
	#Power the base frequency by exponents
	freq_PG = tf.expand_dims(tf.pow(1/Base_Frequency,exponents),0)

	#Compute the encoder values
	encoder_sin = tf.sin(tf.matmul(pos,freq_PG))
	encoder_cos = tf.cos(tf.matmul(pos,freq_PG))

	#Concatenate both values
	encoder = tf.concat([encoder_sin,encoder_cos], axis = 1)

	#Computes the encoder values for x and q
	encoder_x = tf.slice(encoder,[0,0],[size_x,embedding_size])
	encoder_q = tf.slice(encoder,[0,0],[size_q,embedding_size])

	#Encoding x and q
	x_encoded = tf.add(x, encoder_x)
	q_encoded = tf.add(q, encoder_q)
	return x_encoded, q_encoded


def attention_layer(X, mask,X2 = None, scope=None):
	#Self-Attention is defined as:  softmax(X*W_sym*X')*X*WV
	with tf.variable_scope(scope):
		#In order to get a symmetric matrix W_V, only its eigenvectors and eigenvalues are variables.
		WQ_EigVec = tf.nn.l2_normalize(tf.get_variable(name = 'WQEigVec', shape = [embedding_size,embedding_size], dtype = tf.float32), dim = 0)
		WQ_EigenVal = tf.get_variable(name = 'WGEigVal', shape = [embedding_size])
		WV = tf.get_variable(name = 'WV', shape = [embedding_size,embedding_size], dtype = tf.float32)
		
		#W_sym = WQ_EigVec*EigenVal*EigVec
		# x*EigVec
		
		x_proj = tf.matmul(X,
			tf.tile(tf.expand_dims(WQ_EigVec,0),[batch_size,1,1]))
		# x*EigVec is split into multi_head_size smaller matrices
		x_proj_split = tf.transpose(tf.split(x_proj,num_or_size_splits = multihead_size, axis = 2),[1,2,0,3])
		# x*EigVec*EigVal computed
		WQ_EigenVal_Split = tf.split(WQ_EigenVal,num_or_size_splits = multihead_size, axis = 0)
		x_proj_scaled = tf.multiply(x_proj_split,WQ_EigenVal_Split)
		if X2 is None:
			logits = tf.matmul(tf.transpose(x_proj_split,[0,2,1,3]),tf.transpose(x_proj_scaled,[0,2,3,1]))
		else:
			X2_proj = tf.matmul(X2,
				tf.tile(tf.expand_dims(WQ_EigVec,0),[batch_size,1,1]))
			X2_proj_split = tf.transpose(tf.split(X2_proj,num_or_size_splits = multihead_size, axis = 2),[1,2,0,3])
			logits = tf.matmul(tf.transpose(X2_proj_split,[0,2,1,3]),tf.transpose(x_proj_scaled,[0,2,3,1]))

		#(x*EigVec) * (x*EigVec*EigVal)' 
		#Sofmatx with masking
		softmax = tf.nn.softmax(
				tf.add(
					tf.divide(tf.transpose(logits,[1,0,2,3]),tf.sqrt(tf.cast(embedding_size,tf.float32))),
					tf.multiply(1 - mask, VERY_LOW_NUMBER)
					), dim = -1)
		#Final mask is applied
		softmax = tf.multiply (mask,softmax)
		#Computed the new x vector accoring to weights from softmax
		x_weighted = tf.matmul(softmax,tf.tile(tf.expand_dims(X,0),[multihead_size,1,1,1]))
		#Because of multihead attention, WV must be split into multi_head_size smaller matrices
		WV_Split = tf.split(WV, num_or_size_splits = multihead_size, axis = 1)
		#x_weighted*Wv
		x_weighted_proj = tf.matmul(x_weighted,tf.tile(tf.expand_dims(WV_Split,1),[1,batch_size,1,1]))
		#Concatenate everything togeter
		x_weighted_proj_concat = tf.concat(tf.unstack(x_weighted_proj, axis = 0), axis = 2)
	return x_weighted_proj_concat

def layer_normalization (x, gain = 1.0):
	mean, var = tf.nn.moments(x, axes=[-1]) #To computed variance and means
	var += 1e-30 #to avoid NaN, if variance = 0
	normalized_x = tf.transpose(
				tf.subtract(
					tf.multiply(tf.divide(gain,var),
						tf.transpose(x,[2,0,1])), #Transpose for add and multiply operations
					mean),
				[1,2,0])
	return normalized_x

paragraphs=[np.random.randint(1,100, size=np.random.randint(9,12)) for i in range(batch_size)]
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
mask_matrix_xx=tf.cast(tf.sign(tf.matmul(tf.expand_dims(x,-1),tf.expand_dims(x,1))),tf.float32)
mask_matrix_qq=tf.cast(tf.sign(tf.matmul(tf.expand_dims(q,-1),tf.expand_dims(q,1))),tf.float32)
mask_matrix_xq=tf.cast(tf.sign(tf.matmul(tf.expand_dims(q,-1),tf.expand_dims(x,1))),tf.float32)

#Implementing encoder
x_encoded, q_encoded = encoder (x_input,q_input)
att_layer_qq = attention_layer(X = q_encoded, mask = mask_matrix_qq, scope = 'qq')
att_layer_xx = attention_layer(X = x_encoded, mask = mask_matrix_xx, scope = 'xx')
att_layer_xq = attention_layer(X = x_encoded, X2 = q_encoded, mask = mask_matrix_xq, scope = 'xq')
#self_att_layer_xx = cross_attention_layer(x_encoded,mask_matrix_xq)
normalized_qq = layer_normalization(self_att_layer_qq)
#Self-Attention

# Launch the graph in a session.
# Evaluate the tensor `c`.
with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    tf.global_variables_initializer().run()
    pdb.set_trace()
    np.shape(sess.run([cross_att_layer_xq],feed_dict={x : paragraphs, q: questions}))
    pdb.set_trace()
    a=1
