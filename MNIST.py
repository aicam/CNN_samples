import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

# defining number of test and train sets
batch_size = 128
test_size = 256

img_size = 28
# 10 digits
num_classes = 10

X = tf.placeholder("float", [None, img_size, img_size, 1])
Y = tf.placeholder("float", [None, num_classes])

mnist = input_data.read_data_sets("MNIST-data", one_hot=True)
trX, trY, teX, teY = mnist.train.images, \
                     mnist.train.labels, \
                     mnist.test.images,  \
                     mnist.test.labels
trX = trX.reshape(-1, img_size, img_size, 1)
teX = teX.reshape(-1, img_size, img_size, 1)

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))
w = init_weights([3, 3, 1, 32])
w2 = init_weights([3, 3, 32, 64])
w3 = init_weights([3, 3, 64, 128])
w_o = init_weights([625, num_classes])
p_keep_conv = tf.placeholder("float")
p_keep_hidden = tf.placeholder("float")
def model(X, w, w2, w3, w4, w_o, p_keep_conv, p_keep_hidden):
    conv1 = tf.nn.conv2d(X, w, strides=[1, 1, 1, 1], padding='SAME')
    conv1_a = tf.nn.relu(conv1)
    conv1 = tf.nn.max_pool(conv1_a, ksize=[1, 2, 2, 1]\
                           ,strides=[1, 2, 2, 1],\
                           padding='SAME')
    conv1 = tf.nn.dropout(conv1, p_keep_conv)
    conv2 = tf.nn.conv2d(conv1, w2,\
                         strides=[1, 1, 1, 1],\
                         padding='SAME')
    conv2_a = tf.nn.relu(conv2)
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1],\
                        strides=[1, 2, 2, 1],\
                        padding='SAME')
    conv2 = tf.nn.dropout(conv2, p_keep_conv)

    conv3=tf.nn.conv2d(conv2, w3,\
                       strides=[1, 1, 1, 1]\
                       ,padding='SAME')

    conv3 = tf.nn.relu(conv3)
    FC_layer = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], \
                              strides=[1, 2, 2, 1], \
                              padding='SAME')

    FC_layer = tf.reshape(FC_layer,[-1, w4.get_shape().as_list()[0]])
    FC_layer = tf.nn.dropout(FC_layer, p_keep_conv)
    output_layer = tf.nn.relu(tf.matmul(FC_layer, w4))
    output_layer = tf.nn.dropout(output_layer, p_keep_hidden)
