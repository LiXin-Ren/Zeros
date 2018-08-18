# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
from inputs import get_per_attributes
# import cv2

batch_size = 16

def Model(inputs):
    '''
    一个cnn模型
    '''
    # inputs: [BATCH_SIZE, 64, 64, 3]
    # in: 64x64x3 out: 32x32x64
    conv1 = tf.layers.conv2d(inputs, 64, (3, 3), padding='same', activation=tf.nn.relu)
    #pool1 = tf.layers.max_pooling2d(conv1, (2, 2), (2, 2), padding='same')
    pool1 = tf.layers.max_pooling2d(conv1, (2, 2), (2, 2), padding='valid')

    # in: 32x32x64 out: 16x16x32
    conv2 = tf.layers.conv2d(pool1, 32, (3, 3), padding='same', activation=tf.nn.relu)
    #pool2 = tf.layers.max_pooling2d(conv2, (2, 2), (2, 2), padding='same')
    pool2 = tf.layers.max_pooling2d(conv2, (2, 2), (2, 2), padding='valid')

    # in: 16x16x32 out: 8x8x32
    conv3 = tf.layers.conv2d(pool2, 32, (3, 3), padding='same', activation=tf.nn.relu)
   # pool3 = tf.layers.max_pooling2d(conv3, (2, 2), (2, 2), padding='same')
    pool3 = tf.layers.max_pooling2d(conv3, (2, 2), (2, 2), padding='valid')

    # in: 8x8x32 out: 4x4x16
    conv4 = tf.layers.conv2d(pool3, 16, (3, 3), padding='same', activation=tf.nn.relu)
    #pool4 = tf.layers.max_pooling2d(conv4, (2, 2), (2, 2), padding='same')
    pool4 = tf.layers.max_pooling2d(conv4, (2, 2), (2, 2), padding='valid')

    # in: 4x4x16 out: 256
    flatten = tf.layers.flatten(pool4)
    fcn1 = tf.layers.dense(
        inputs=flatten,
        units=128,
        activation=tf.nn.relu,
        use_bias=True,
        kernel_initializer=tf.truncated_normal_initializer(),
        bias_initializer=tf.zeros_initializer(),
    )
    fcn2 = tf.layers.dense(
        inputs=fcn1,
        units=30,
        activation=tf.nn.sigmoid,
        use_bias=True,
        kernel_initializer=tf.truncated_normal_initializer(),
        bias_initializer=tf.zeros_initializer(),
    )
    return fcn2



def loss(labels, pred):
    """Calculates the loss from the real attribute and the predicted attribute.
    """
    return tf.losses.mean_squared_error(labels, pred)


def optimize(loss, learning_rate, global_step):
    """Sets up the training Ops.

    Args:
        loss: Loss tensor, from loss().
        lr: The learning rate to use for gradient descent.

    Returns:
        train_op: The Op for training.
    """

    with tf.name_scope('optimization'):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        # with a global_step to track the global step.
        train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op


def accuracy(classLabel, predAttri):
    attributionDict = get_per_attributes()
    res = {}
    acc = 0
    for i in range(len(classLabel)):
        for category, attri in attributionDict.iteritems():
            dis = sum([(predAttri[j] - classLabel[i][j]) ** 2 for j in range(30)])
            res[category] = dis     #每种类别的距离
        predClass = max(dict, key=dict.get)
        if predClass == classLabel[i]:
            acc += 1
    return acc/len(classLabel)









