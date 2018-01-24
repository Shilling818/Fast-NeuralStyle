from __future__ import print_function, division
import numpy as np
import os
# import argparse
from scipy import signal
# Python Image Library
from PIL import Image
from net import *
import scipy.misc
import tensorflow as tf


def load_image(path):
    image = Image.open(path).convert('RGB')
    # return np.transpose(np.asarray(image, dtype=np.float32), (2, 0, 1))
    return np.asarray(image, dtype=np.float32)


def gram_matrix(y):
    b, h, w, ch = y.shape
    features = tf.reshape(y, (b, w*h, ch))
    tmp = tf.to_float(tf.convert_to_tensor(w*h*ch))
    gram = tf.matmul(features, features, transpose_a=True) / tmp
    return gram


def total_variation(x):
    wh = tf.constant([[[[1], [-1]], [[0], [0]], [[0], [0]]], [[[0], [0]], [[1], [-1]], [[0], [0]]], [[[0], [0]], [[0], [0]], [[1], [-1]]]], dtype=tf.float32)
    ww = tf.constant([[[[1, -1]], [[0, 0]], [[0, 0]]], [[[0, 0]], [[1, -1]], [[0, 0]]], [[[0, 0]], [[0, 0]], [[1, -1]]]], dtype=tf.float32)
    
    x_ = tf.transpose(x, perm=[1, 2, 0, 3])
    y = tf.reduce_sum(tf.nn.conv2d(x_, wh, strides=[1, 1, 1, 1], padding = "VALID") ** 2) + tf.reduce_sum(tf.nn.conv2d(x_, ww, strides=[1, 1, 1, 1], padding = "VALID") ** 2)
    return y


def normalization(x):
    y = []
    for feat in x:
        b, h, w, ch = feat.shape
        feat_mean = tf.reduce_mean(feat)
        tmp = tf.to_float(tf.convert_to_tensor(b * w * h * ch - 1.0))
        feat_var = tf.sqrt(tf.reduce_sum((feat - feat_mean) ** 2) / tmp)
        # feat_var = tf.constant(1.0) if tf.less(feat_var, tf.constant(1.0)) else feat_var
        feat = (feat - feat_mean) / feat_var
        y.append(feat)
    return y


batchsize = 15
image_size = 700
n_epoch = 1000
lambda_tv = 1
lambda_f = 1.0
lambda_s = 1e5
lambda_p = 1e-3
vgg = VGG()
lr = 1e-4
opt_imagepaths = ['./data/opt/1.png', './data/opt/2.png', './data/opt/3.png', './data/opt/4.png', './data/opt/5.png',
              './data/opt/6.png', './data/opt/7.png', './data/opt/8.png', './data/opt/9.png', './data/opt/10.png',
              './data/opt/11.png', './data/opt/12.png', './data/opt/13.png', './data/opt/14.png', './data/opt/15.png']
pol_imagepaths = ['./data/pol/1.png', './data/pol/2.png', './data/pol/3.png', './data/pol/4.png', './data/pol/5.png',
              './data/pol/6.png', './data/pol/7.png', './data/pol/8.png', './data/pol/9.png', './data/pol/10.png',
              './data/pol/11.png', './data/pol/12.png', './data/pol/13.png', './data/pol/14.png', './data/pol/15.png']             
style_image = './data/opt/2.png'
feat_num = [64, 128, 256, 512]
style = np.asarray(Image.open(style_image).convert('RGB'), dtype=np.float32) 
style_b = np.zeros((1, ) + style.shape, dtype=np.float32)
style_b[0] = style
feature_s = normalization(vgg(style_b))
gram_s = [gram_matrix(y) for y in feature_s]

img = tf.placeholder(tf.float32, [1, image_size, image_size, 3])
real_img = tf.placeholder(tf.float32, [1, image_size, image_size, 3])
feature = normalization(vgg(img))
# feature = vgg(img - 120)
y = FastStyleNet(img) 
L_pixel = lambda_p * tf.reduce_sum((y - real_img)**2) / (image_size * image_size * 3)
feature_hat = normalization(vgg(y))
# feature_hat = vgg(y - 120)
L_feat = lambda_f * tf.reduce_sum((feature[2] - feature_hat[2])**2) / (image_size * image_size * 16) # compute for only the output of layer conv3_3
L_style = np.zeros([1], dtype=np.float32)
for f_hat, g_s, f_num in zip(feature_hat, gram_s, feat_num):
    L_style += lambda_s * tf.reduce_sum((gram_matrix(f_hat) - g_s)**2) / f_num / f_num
# L_tv = lambda_tv * total_variation(y)
L = L_feat + L_style + L_pixel
optim = tf.train.AdamOptimizer(lr).minimize(L)


saver = tf.train.Saver(max_to_keep=1000)
init = tf.global_variables_initializer()


with tf.Session() as sess:
    sess.run(init)
    g_loss = np.zeros(400, dtype=float)
    # gram_s = sess.run(gram_s)
    for epoch in range(n_epoch):
        # train
        if os.path.isdir("./result/%04d" % epoch):
            continue
        os.makedirs("./result/%04d" % epoch)
        print('epoch', epoch)
        # se = np.random.permutation(360) + 1
        label = np.zeros((1, image_size, image_size, 3), dtype=np.float32)
        real = np.zeros((1, image_size, image_size, 3), dtype=np.float32)
        for j in range(1, batchsize):
            real[0] = load_image(opt_imagepaths[j])
            label[0] = load_image(pol_imagepaths[j])
            # x[0] = np.float32(scipy.misc.imread("./data/car_test/%08d.png" % (j + 1)))

            _, loss, loss_feat, loss_style, loss_pixel = sess.run([optim, L, L_feat, L_style, L_pixel], feed_dict={img: label, real_img: real})
            g_loss[j] = loss
            print('(epoch {}) batch {}... training loss is {}...feature loss is {}...style loss is {}...pixel loss is {}'.format(epoch, j, np.mean(g_loss[np.where(g_loss)]), loss_feat, loss_style, loss_pixel))
        
        target = open("./result/%04d/score.txt" % epoch, 'w')
        target.write("%f" % np.mean(g_loss[np.where(g_loss)]))
        target.close()
            
        # test
        x = np.zeros((1, image_size, image_size, 3), dtype=np.float32)
        x[0] = load_image(pol_imagepaths[0])
        # x[0] = np.float32(scipy.misc.imread("./data/car_test/00000001.png"))
        optic = sess.run(y, feed_dict={img: x})
        # scipy.misc.toimage(optic[0, :, :, :], cmin=0, cmax=255).save("./result/%04d/output.png" % epoch)
        scipy.misc.toimage(optic[0, :, :, :]).save("./result/%04d/output.png" % epoch)
    saver.save(sess, "./result/%04d/model.ckpt" % epoch)



