import numpy as np
import sys
import tensorflow as tf
import matplotlib.pyplot as plt

inputs_ = tf.placeholder(tf.float32,[None,16, 16, 16,3])
#targets_ = tf.placeholder(tf.float32,[None,28,28,1])

def lrelu(x,alpha=0.1):
    return tf.maximum(alpha*x,x)

conv1 = tf.layers.conv3d(inputs=inputs_, filters=16, kernel_size=[3,3,3], padding='same', activation=tf.nn.relu)
# conv => 16*16*16
conv2 = tf.layers.conv3d(inputs=conv1, filters=32, kernel_size=[3,3,3], padding='same', activation=tf.nn.relu)
# pool => 8*8*8
pool3 = tf.layers.max_pooling3d(inputs=conv2, pool_size=[2, 2, 2], strides=2)
        
# conv => 8*8*8
conv4 = tf.layers.conv3d(inputs=pool3, filters=64, kernel_size=[3,3,3], padding='same', activation=tf.nn.relu)
# conv => 8*8*8
conv5 = tf.layers.conv3d(inputs=conv4, filters=128, kernel_size=[3,3,3], padding='same', activation=tf.nn.relu)
# pool => 4*4*4
pool6 = tf.layers.max_pooling3d(inputs=conv5, pool_size=[2, 2, 2], strides=2)
        
cnn3d_bn = tf.layers.batch_normalization(inputs=pool6, training=True)
        
flattening = tf.reshape(cnn3d_bn, [-1, 4*4*4*128])
dense = tf.layers.dense(inputs=flattening, units=1024, activation=tf.nn.relu)
# (1-0.7) is the probability that the node will be kept
dropout = tf.layers.dropout(inputs=dense, rate=0.7, training=True)

y_conv = tf.layers.dense(inputs=dropout, units=10)
'''
loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,labels=targets_)

learning_rate=tf.placeholder(tf.float32)
cost = tf.reduce_mean(loss)  #cost
opt = tf.train.AdamOptimizer(learning_rate).minimize(cost) #optimizer
'''
sess = tf.Session()
tf.train.write_graph(sess.graph.as_graph_def(add_shapes=True), '/tmp', '/home/rockstar/conv3d.pbtxt', True)

#print(tf.summary())

#tf.reset_default_graph()