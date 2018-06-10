import tensorflow as tf
from tensorflow.core.framework import graph_pb2
from google.protobuf import text_format

config = open('rnn.pbtxt').read()
graph_def = graph_pb2.GraphDef()
text_format.Merge(config, graph_def)

tf.import_graph_def(graph_def, name='')
tf.summary.FileWriter('logs/', graph=tf.get_default_graph())
