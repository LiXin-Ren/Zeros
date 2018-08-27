# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 09:46:29 2018

@author: feng
"""

import tensorflow as tf

def encoder(input_x):
    # in: 64x63x3 out: 32x32x64
    conv1 = tf.layers.conv2d(input_x, 64, (3, 3), padding = 'same', activation = tf.nn.relu)
    pool1 = tf.layers.max_pooling2d(conv1, (2, 2), (2, 2), padding = 'same')

    # in: 32x32x64 out: 16x16x32    
    conv2 = tf.layers.conv2d(pool1, 32, (3, 3), padding = 'same', activation = tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(conv2, (2, 2), (2, 2), padding = 'same')   
    
    # in: 16x16x32 out: 8x8x32    
    conv3 = tf.layers.conv2d(pool2, 32, (3, 3), padding = 'same', activation = tf.nn.relu)
    pool3 = tf.layers.max_pooling2d(conv3, (2, 2), (2, 2), padding = 'same') 
    
    # in: 8x8x32 out: 4x4x16    
    conv4 = tf.layers.conv2d(pool3, 16, (3, 3), padding = 'same', activation = tf.nn.relu)
    pool4 = tf.layers.max_pooling2d(conv4, (2, 2), (2, 2), padding = 'same') 
    
    return pool4

def decoder(input_x, batch_size):
    # in: 4x4x16 out: 8x8x32
    unsample1 = tf.image.resize_nearest_neighbor(input_x, (8, 8))
    conv1 = tf.layers.conv2d(unsample1, 32, (3, 3), padding = 'same', activation = tf.nn.relu)
    
    # in: 8x8x32 out: 16x16x32
    unsample2 = tf.image.resize_nearest_neighbor(conv1, (16, 16))
    conv2 = tf.layers.conv2d(unsample2, 32, (3, 3), padding = 'same', activation = tf.nn.relu)
    
    # in: 16x16x32 out: 32x32x64
    unsample3 = tf.image.resize_nearest_neighbor(conv2, (32, 32))
    conv3 = tf.layers.conv2d(unsample3, 64, (3, 3), padding = 'same', activation = tf.nn.relu)
    
    # in: 32x32x64 out: 64x64x64
    unsample4 = tf.image.resize_nearest_neighbor(conv3, (64, 64))
    conv4 = tf.layers.conv2d(unsample4, 64, (3, 3), padding = 'same', activation = tf.nn.relu)
    
    logits = tf.layers.conv2d(conv4, 3, (3, 3), padding = 'same', activation = tf.nn.relu)
    outputs = tf.nn.sigmoid(logits, name = 'outputs')
    
    return outputs