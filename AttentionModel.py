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
embedding_size=300 #MUST BE EVEN FOR ENCODER
size_of_vocabulary=100
batch_size=400
multihead_size = 6
FF_hidden_size = 1024
word_zeros=np.zeros([1,embedding_size])
embedding_dict=np.array(np.concatenate([word_zeros,np.random.rand(size_of_vocabulary,embedding_size)],axis=0), dtype=np.float32)

def embed_scaling (x):
	W_Scal = tf.get_variable('W_Scal', shape = [embedding_size, embedding_size])
	b_Scal = tf.get_variable('b_Scal', shape = [1, embedding_size])
	affine_op = tf.add(tf.matmul(tf.reshape(x,[-1,embedding_size]),W_Scal),b_Scal) #W1*x+b1
	x_reshaped = tf.reshape(affine_op, [batch_size,-1,embedding_size])
	return x_reshaped

def encoder (X,Q):
	#Compute the number of words in passage and question
	size_x = tf.shape(X)[-2]
	size_q = tf.shape(Q)[-2]
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
	x_encoded = tf.add(X, encoder_x)
	q_encoded = tf.add(Q, encoder_q)
	return x_encoded, q_encoded


def attention_layer(X, mask,X2 = None, scope=None):
	#Self-Attention is defined as:  softmax(X*W_sym*X')*X*WV
	with tf.variable_scope(scope):
		#In order to get a symmetric matrix W_V, only its eigenvectors and eigenvalues are variables.
		WQ = tf.get_variable(name = 'WQ', shape = [embedding_size,embedding_size], dtype = tf.float32)
		WK = tf.get_variable(name = 'WK', shape = [embedding_size,embedding_size], dtype = tf.float32)
		WV = tf.get_variable(name = 'WV', shape = [embedding_size,embedding_size], dtype = tf.float32)
		WO = tf.get_variable(name = 'WO', shape = [embedding_size,embedding_size], dtype = tf.float32)
		
		#W_sym = WQ_EigVec*EigenVal*EigVec
		# x*EigVec

		x1_proj = tf.reshape(tf.matmul(tf.reshape(X,[-1,embedding_size]),WQ),[batch_size,-1,embedding_size])
		# x*EigVec is split into multi_head_size smaller matrices
		x1_proj = tf.split(x1_proj,num_or_size_splits = multihead_size, axis = 2)
		# x*EigVec*EigVal computed
		if X2 is None:
			X2 = X

		x2_proj = tf.reshape(tf.matmul(tf.reshape(X2,[-1,embedding_size]),WK),[batch_size,-1,embedding_size])
		x2_proj = tf.split(x2_proj,num_or_size_splits = multihead_size, axis = 2)
		logits = tf.matmul(x2_proj,tf.transpose(x1_proj,[0,1,3,2]))

		#(x*EigVec) * (x*EigVec*EigVal)' 
		#Sofmax with masking
		softmax = tf.nn.softmax(
				tf.add(
				tf.divide(logits,
					tf.sqrt(tf.cast(embedding_size,tf.float32))),
				tf.multiply(1 - mask, VERY_LOW_NUMBER)
					), dim = -1)
		#Final mask is applied
		softmax = tf.multiply (mask,softmax)
		#Computed the new x vector accoring to weights from softmax
		
		x3_proj = tf.reshape(tf.matmul(tf.reshape(X,[-1,embedding_size]),WK),[batch_size,-1,embedding_size])
		x3_proj = tf.split(x3_proj,num_or_size_splits = multihead_size, axis = 2)
		#Because of multihead attention, WV must be split into multi_head_size smaller matrices
		x_attention = tf.matmul(softmax,x3_proj)
		x_final = tf.concat(tf.unstack(x_attention, axis = 0), axis = 2)
		x_final = tf.reshape(tf.matmul(tf.reshape(x_final,[-1,embedding_size]),WO),[batch_size,-1,embedding_size])
		#Concatenate everything togeter
	return x_final

def layer_normalization (x, gain = 1.0):
	mean, var = tf.nn.moments(x, axes=[-1]) #To computed variance and means
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
		W1 = tf.get_variable('W1', shape = [embedding_size, FF_hidden_size])
		b1 = tf.get_variable('b1', shape = [1, FF_hidden_size])
		W2 = tf.get_variable('W2', shape = [FF_hidden_size, embedding_size])
		b2 = tf.get_variable('b2', shape = [1, embedding_size])
		
		#Computation of #W2*Relu(W1*x+b1)+b2
		affine_op1 = tf.add(tf.matmul(tf.reshape(x,[-1,embedding_size]),W1),b1) #W1*x+b1
		nonlinear_op = tf.nn.relu(affine_op1) #Relu(W1*x+b1)
		affine_op2 = tf.add(tf.matmul(nonlinear_op,W2),b2) #W2*Relu(W1*x+b1)+b2
		output = tf.reshape(affine_op2, [batch_size,-1,embedding_size]) #Reshaping
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
		W = tf.get_variable('W', shape = [embedding_size, 1], dtype = tf.float32)
		logits = tf.reshape(tf.matmul(tf.reshape(X,[-1,embedding_size]),W), [batch_size,-1]) #W*x
		output = tf.nn.softmax(tf.add(logits,tf.multiply(1.0 - mask, VERY_LOW_NUMBER)))
	return output, logits
		
#$paragraphs=[np.random.randint(1,100, size=np.random.randint(200,400)) for i in range(batch_size)]
paragraphs=[np.random.randint(1,100, size=400) for i in range(batch_size)]
questions = [np.random.randint(1,100, size=60) for i in range(batch_size)]

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
x_input = tf.nn.embedding_lookup(embedding_dict,x)
q_input = tf.nn.embedding_lookup(embedding_dict,q)

#Mask matrices
mask = {}
mask['x1'] = tf.cast(tf.sign(q),tf.float32)
mask['x2'] = tf.cast(tf.sign(x),tf.float32)
mask['x1x1'] = tf.cast(tf.sign(tf.matmul(tf.expand_dims(q,-1),tf.expand_dims(q,1))),tf.float32)
mask['x2x2'] = tf.cast(tf.sign(tf.matmul(tf.expand_dims(x,-1),tf.expand_dims(x,1))),tf.float32)
mask['x2x1'] = tf.cast(tf.sign(tf.matmul(tf.expand_dims(x,-1),tf.expand_dims(q,1))),tf.float32)

#Scaling matrices
with tf.variable_scope('Scaling') as scope:
	x_scaled = embed_scaling(x_input)
	scope.reuse_variables()
	q_scaled = embed_scaling(q_input)

#Encoding Variables
x_encoded, q_encoded = encoder (x_scaled,q_scaled)
#Computing all attentions
q_1, x_1 = one_layer(q_encoded, x_encoded, mask, 'layer_0')
q_2, x_2 = one_layer(q_1, x_1, mask, 'layer_1')
q_3, x_3 = one_layer(q_2, x_2, mask, 'layer_2')
q_4, x_4 = one_layer(q_3, x_3, mask, 'layer_3')
q_5, x_5 = one_layer(q_4, x_4, mask, 'layer_4')
q_6, x_6 = one_layer(q_5, x_5, mask, 'layer_5')
yp, logits_y1 = y_selection(X = x_6,scope = 'y1_sel', mask = mask['x2'])
yp2, logits_y2 = y_selection(X = x_6,scope = 'y2_sel', mask = mask['x2'])
Start_Index = tf.argmax(logits_y1, axis=-1)
End_Index = tf.argmax(logits_y2, axis=-1)

# Launch the graph in a session.
# Evaluate the tensor `c`.
#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1)


config = tf.ConfigProto()
config.gpu_options.allocator_type = 'BFC'
with tf.Session(config=config) as sess:
    tf.global_variables_initializer().run()
    print(sess.run([Start_Index,End_Index],feed_dict={x : paragraphs, q: questions}))
    a=1
