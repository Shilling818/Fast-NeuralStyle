# -*- coding: utf-8 -*-
import math
import numpy as np
import tensorflow.contrib.layers as layers
import tensorflow as tf
import scipy.io


def ResidualBlock(x, dim):
    # w = math.sqrt(2)
    h1 = tf.nn.leaky_relu(layers.conv2d(x, dim, kernel_size=3, stride=1, rate=1, normalizer_fn=layers.layer_norm, weights_initializer=layers.xavier_initializer(uniform=False)))  # , scope='g_' + str(sp) + '_conv1', reuse=tf.AUTO_REUSE
    h2 = tf.nn.leaky_relu(layers.conv2d(h1, dim, kernel_size=3, stride=1, rate=1, normalizer_fn=layers.layer_norm, weights_initializer=layers.xavier_initializer(uniform=False)))
    if x.shape != h2.shape:
        n, hh, ww, c = x.shape
        pad_c = h2.shape[3] - c
        p = tf.zeros([n, hh, ww, pad_c], dtype=np.float32)
        x = tf.concat([p, x], 3)
        if x.shape[1] != h2.shape[1]:
            x = layers.avg_pool2d(x, 1, 2)
    return h2 + x


def FastStyleNet(n_in):
    # w = math.sqrt(2)
    c1 = tf.nn.leaky_relu(layers.conv2d(n_in, 32, kernel_size=9, stride=1, rate=1, padding='SAME', normalizer_fn=layers.layer_norm, weights_initializer=layers.xavier_initializer(uniform=False)))
    c2 = tf.nn.leaky_relu(layers.conv2d(c1, 64, kernel_size=4, stride=2, rate=1, padding='SAME', normalizer_fn=layers.layer_norm, weights_initializer=layers.xavier_initializer(uniform=False)))
    c3 = tf.nn.leaky_relu(layers.conv2d(c2, 128, kernel_size=4, stride=2, rate=1, padding='SAME', normalizer_fn=layers.layer_norm, weights_initializer=layers.xavier_initializer(uniform=False)))
    r1 = ResidualBlock(c3, 128)
    r2 = ResidualBlock(r1, 128)
    r3 = ResidualBlock(r2, 128)
    r4 = ResidualBlock(r3, 128)
    r5 = ResidualBlock(r4, 128)
    d1 = tf.nn.leaky_relu(layers.conv2d_transpose(r5, 64, kernel_size=4, stride=2, padding='SAME', normalizer_fn=layers.layer_norm, weights_initializer=layers.xavier_initializer(uniform=False)))
    d2 = tf.nn.leaky_relu(layers.conv2d_transpose(d1, 32, kernel_size=4, stride=2, padding='SAME', normalizer_fn=layers.layer_norm, weights_initializer=layers.xavier_initializer(uniform=False)))
    d3 = tf.tanh(layers.conv2d_transpose(d2, 3, kernel_size=9, stride=1, padding='SAME', normalizer_fn=layers.layer_norm, weights_initializer=layers.xavier_initializer(uniform=False)))
    return (d3 + 1) * 127.5


class VGG(object):
    def load_weights(self):
        matfn = './data/vgg16_tf.mat'
        dic = scipy.io.loadmat(matfn)
        return dic

    def weights(self, shape, name):
        weights_dic = self.load_weights()
        if name in weights_dic:
            w = tf.Variable(weights_dic[name], name=name, trainable=False)
        else:
            num = shape[0] * shape[1] * shape[2] * shape[3]
            w = tf.Variable(tf.random_normal(shape=shape, stddev=np.sqrt(1.0/num)), name=name, trainable=True)
        return w

    def bias(self, shape, name):
        weights_dic = self.load_weights()
        if name in weights_dic:
            tmp_b = weights_dic[name]
            tmp_b.shape = shape
            b = tf.Variable(tmp_b, name=name, trainable=False)
        else:
            b = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=shape), name=name, trainable=True)
        return b

    def conv2d(self, input, w, bia, strides=[1, 1, 1, 1], paddings="SAME"):
        conv = tf.nn.conv2d(input, w, strides, padding=paddings)
        return tf.nn.bias_add(conv, bia)

    def __init__(self):
        # super(VGG, self).__init__(
        self.wconv1_1 = self.weights([3, 3, 3, 64], name='wconv1_1')
        self.bconv1_1 = self.bias([64], name='bconv1_1')
        self.wconv1_2 = self.weights([3, 3, 64, 64], name='wconv1_2')
        self.bconv1_2 = self.bias([64], name='bconv1_2')

        self.wconv2_1 = self.weights([3, 3, 64, 128], name='wconv2_1')
        self.bconv2_1 = self.bias([128], name='bconv2_1')
        self.wconv2_2 = self.weights([3, 3, 128, 128], name='wconv2_2')
        self.bconv2_2 = self.bias([128], name='bconv2_2')

        self.wconv3_1 = self.weights([3, 3, 128, 256], name='wconv3_1')
        self.bconv3_1 = self.bias([256], name='bconv3_1')
        self.wconv3_2 = self.weights([3, 3, 256, 256], name='wconv3_2')
        self.bconv3_2 = self.bias([256], name='bconv3_2')
        self.wconv3_3 = self.weights([3, 3, 256, 256], name='wconv3_3')
        self.bconv3_3 = self.bias([256], name='bconv3_3')

        self.wconv4_1 = self.weights([3, 3, 256, 512], name='wconv4_1')
        self.bconv4_1 = self.bias([512], name='bconv4_1')
        self.wconv4_2 = self.weights([3, 3, 512, 512], name='wconv4_2')
        self.bconv4_2 = self.bias([512], name='bconv4_2')
        self.wconv4_3 = self.weights([3, 3, 512, 512], name='wconv4_3')
        self.bconv4_3 = self.bias([512], name='bconv4_3')
        # )
        self.train = False
        self.mean = np.asarray(120, dtype=np.float32)

    def preprocess(self, image):
        # [channel, height, width]
        # return np.rollaxis(image - self.mean, 2)
        return image - self.mean

    def __call__(self, x):
        y1 = tf.nn.relu(self.conv2d(tf.nn.relu(self.conv2d(x, self.wconv1_1, self.bconv1_1)), self.wconv1_2, self.bconv1_2))
        # stride = [1,2,2,1] 是因为中间两个值为height和width
        h = tf.nn.max_pool(y1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        y2 = tf.nn.relu(self.conv2d(tf.nn.relu(self.conv2d(h, self.wconv2_1, self.bconv2_1)), self.wconv2_2, self.bconv2_2))
        h = tf.nn.max_pool(y2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        y3 = tf.nn.relu(self.conv2d(tf.nn.relu(self.conv2d(tf.nn.relu(self.conv2d(h, self.wconv3_1, self.bconv3_1)), self.wconv3_2, self.bconv3_2)), self.wconv3_3, self.bconv3_3))
        h = tf.nn.max_pool(y3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        y4 = tf.nn.relu(self.conv2d(tf.nn.relu(self.conv2d(tf.nn.relu(self.conv2d(h, self.wconv4_1, self.bconv4_1)), self.wconv4_2, self.bconv4_2)), self.wconv4_3, self.bconv4_3))
        # y1,y2,y3,y4 height and width are not the same size
        # 要不要归一化？
        return [y1, y2, y3, y4]
