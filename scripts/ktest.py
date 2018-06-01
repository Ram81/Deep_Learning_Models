from keras.models import model_from_json
import tensorflow as tf
import numpy as np
from keras import backend as K
import argparse
import os

parser = argparse.ArgumentParser(description='set input arguments')
parser.add_argument('-input_file', action="store",
                    dest='input_file', type=str, default='model.json')
parser.add_argument('-output_file', action="store",
                    dest='output_file', type=str, default='model.pbtxt')
args = parser.parse_args()
input_file = args.input_file
output_file = args.output_file

K.set_learning_phase(0)
K.set_image_dim_ordering('th')


with open('model.json', 'r') as f:
    json_str = f.read()

json_str = json_str.strip("'<>() ").replace('\'', '\"')
model = model_from_json(json_str)

sess = K.get_session()
#sess.run(tf.global_variables_initializer())
print(sess.graph.get_tensor_by_name("conv2d_transpose_1/conv2d_transpose:0"))
print sess.run(sess.graph.get_tensor_by_name("conv2d_transpose_1/conv2d_transpose:0"), feed_dict={sess.graph.get_tensor_by_name('input_1:0'): np.zeros((1,28,28,1))}).shape
print (sess.graph.get_tensor_by_name("conv2d_transpose_2/conv2d_transpose:0")).shape

tf.train.write_graph(sess.graph.as_graph_def(add_shapes=True), '/home/rockstar/',
                     output_file + '.pbtxt', as_text=True)
