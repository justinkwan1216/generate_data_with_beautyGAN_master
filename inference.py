# -*- coding: utf-8 -*-

import tensorflow as tf
import keras
import numpy as np
import os
import glob
from imageio import imread, imsave
import cv2
import argparse
from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile

parser = argparse.ArgumentParser()
parser.add_argument('--no_makeup', type=str, default=os.path.join('imgs', 'no_makeup', 'xfsy_0068.png'), help='path to the no_makeup image')
args = parser.parse_args()


def preprocess(img):
    return (img / 255. - 0.5) * 2

def deprocess(img):
    return (img + 1) / 2

batch_size = 1
img_size = 256
no_makeup = cv2.resize(imread(args.no_makeup), (img_size, img_size))
X_img = np.expand_dims(preprocess(no_makeup), 0)
makeups = glob.glob(os.path.join('imgs', 'makeup', '*.*'))
result = np.ones((2 * img_size, (len(makeups) + 1) * img_size, 3))
result[img_size: 2 *  img_size, :img_size] = no_makeup / 255.

#sess = tf.Session()
#with gfile.FastGFile("models/0/variables"+'model.pb', 'rb') as f:
#    graph_def = tf.GraphDef()
#    graph_def.ParseFromString(f.read())
#    sess.graph.as_default()
#    tf.import_graph_def(graph_def, name='') # 导入计算图

# 需要有一个初始化的过程    
#sess.run(tf.global_variables_initializer())

model_filepath = "models/0/variables"+'model.pb'
print('Loading model...')
graph = tf.Graph()
sess = tf.compat.v1.InteractiveSession()(graph = graph)

with tf.gfile.GFile(model_filepath, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())j

print('Check out the input placeholders:')
nodes = [n.name + ' => ' +  n.op for n in graph_def.node if n.op in ('Placeholder')]
for node in nodes:
    print(node)

# Define input tensor
#input = tf.placeholder(np.float32, shape = [None, 32, 32, 3], name='input')
#dropout_rate = tf.placeholder(tf.float32, shape = [], name = 'dropout_rate')

#tf.import_graph_def(graph_def, {'input': input, 'dropout_rate': dropout_rate})

print('Model loading complete!')

#graph = tf.get_default_graph()
#X = graph.get_tensor_by_name('X:0')
#Y = graph.get_tensor_by_name('Y:0')
#Xs = graph.get_tensor_by_name('generator/xs:0')

for i in range(len(makeups)):
    makeup = cv2.resize(imread(makeups[i]), (img_size, img_size))
    Y_img = np.expand_dims(preprocess(makeup), 0)
    #Xs_ = sess.run(Xs, feed_dict={X: X_img, Y: Y_img})
    #Xs_ = Xs
    Xs = f(x=Y_img)
    Xs_ = deprocess(Xs_)
    result[:img_size, (i + 1) * img_size: (i + 2) * img_size] = makeup / 255.
    result[img_size: 2 * img_size, (i + 1) * img_size: (i + 2) * img_size] = Xs_[0]
    
imsave('result.jpg', result)
