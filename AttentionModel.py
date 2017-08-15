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
embedding_size=256 #MUST BE EVEN FOR ENCODER
size_of_vocabulary=100
batch_size=3
multihead_size = 4
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

def one_layer (X, Q, mask, scope):
	with tf.variable_scope(scope):
		att_layer_qq = layer_normalization(tf.add(Q, attention_layer(X = Q, mask = mask['qq'], scope = 'qq')))
		att_layer_xx = layer_normalization(tf.add(X, attention_layer(X = X, mask = mask['xx'], scope = 'xx')))
		FF_xx = layer_normalization(tf.add(att_layer_xx, FeedForward_NN(att_layer_xx,'FF_xx')))
		att_layer_xq = layer_normalization(tf.add(att_layer_qq,attention_layer(X = FF_xx, X2 = att_layer_qq, mask = mask['xq'], scope = 'xq')))
		FF_qq = layer_normalization(tf.add(att_layer_xq,FeedForward_NN(att_layer_xq,'FF_qq')))
	return FF_xx, FF_qq

def y_selection(X, mask, scope):
	with tf.variable_scope(scope):
		W = tf.get_variable('W', shape = [embedding_size, 1], dtype = tf.float32)
		X_scaled = tf.matmul(tf.reshape(X,[-1,embedding_size]),W) #W*x
		output = tf.nn.softmax(tf.add(tf.reshape(X_scaled, [batch_size,-1]),tf.multiply(1.0 - mask, VERY_LOW_NUMBER)))
	return output
		
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
x_input = tf.nn.embedding_lookup(embedding_dict,x)
q_input = tf.nn.embedding_lookup(embedding_dict,q)

#Mask matrices
mask = {}
mask['x'] = tf.cast(tf.sign(x),tf.float32)
mask['q'] = tf.cast(tf.sign(q),tf.float32)
mask['xx'] = tf.cast(tf.sign(tf.matmul(tf.expand_dims(x,-1),tf.expand_dims(x,1))),tf.float32)
mask['qq'] = tf.cast(tf.sign(tf.matmul(tf.expand_dims(q,-1),tf.expand_dims(q,1))),tf.float32)
mask['xq'] = tf.cast(tf.sign(tf.matmul(tf.expand_dims(q,-1),tf.expand_dims(x,1))),tf.float32)

#Scaling matrices
with tf.variable_scope('Scaling') as scope:
	x_scaled = embed_scaling(x_input)
	scope.reuse_variables()
	q_scaled = embed_scaling(q_input)

#Encoding Variables
x_encoded, q_encoded = encoder (x_scaled,q_scaled)
#Computing all attentions
x_1, q_1 = one_layer(x_encoded, q_encoded, mask, 'layer_0')
x_2, q_2 = one_layer(x_1, q_1, mask, 'layer_1')
x_3, q_3 = one_layer(x_2, q_2, mask, 'layer_2')
y1 = y_selection(X = x_3,scope = 'y1_sel', mask = mask['x'])
y2 = y_selection(X = x_3,scope = 'y2_sel', mask = mask['x'])
#att_layer_qq = attention_layer(X = q_encoded, mask = mask_matrix_qq, scope = 'qq')
#att_layer_xx = attention_layer(X = x_encoded, mask = mask_matrix_xx, scope = 'xx')
#att_layer_xq = attention_layer(X = att_layer_xx, X2 = att_layer_qq, mask = mask_matrix_xq, scope = 'xq')
#$normalized_qq = layer_normalization(att_layer_qq)
#a = FeedForward_NN(normalized_qq,'FF_nn')

# Launch the graph in a session.
# Evaluate the tensor `c`.
with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    tf.global_variables_initializer().run()
    np.shape(sess.run([y1],feed_dict={x : paragraphs, q: questions}))
    pdb.set_trace()
    a=1
