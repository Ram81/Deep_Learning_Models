import numpy as np
import tensorflow as tf
from google.protobuf import text_format
from tensorflow.core.framework import graph_pb2
from keras import backend as K

config = open('test_out.pbtxt', 'r').read()

tf.reset_default_graph()
graph_def = graph_pb2.GraphDef()

try:
    text_format.Merge(config, graph_def)
except Exception as e:
    print e

tf.import_graph_def(graph_def, name='')
graph = tf.get_default_graph()

sess = tf.Session(graph = graph)
K.set_session(sess)

print(K.get_session().graph.as_graph_def())