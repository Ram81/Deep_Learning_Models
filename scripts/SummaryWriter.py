#Source code with the blog post at http://monik.in/a-noobs-guide-to-implementing-rnn-lstm-using-tensorflow/
import numpy as np
import random
from random import shuffle
import tensorflow as tf

print "test and training data loaded"


data = tf.placeholder(tf.float32, [None, 20,1]) #Number of examples, number of input, dimension of each input

num_hidden = 24
#cell = tf.contrib.rnn.BasicLSTMCell(num_units=24, activation=None)
#cell = tf.contrib.rnn.BasicRNNCell(num_units=24, activation=None)
cell = tf.contrib.rnn.GRUCell(num_units=24, activation=None)
val, _ = tf.nn.dynamic_rnn(cell, inputs=data, dtype=tf.float32)
val = tf.layers.dense(val, 10)

sess = tf.Session()

tf.train.write_graph(tf.get_default_graph().as_graph_def(), '/tmp', '/home/rockstar/Desktop/lstm_1.pbtxt', True)
train_writer = tf.summary.FileWriter('/home/rockstar/Tensorflow_Models',
                                      sess.graph)

print('end')