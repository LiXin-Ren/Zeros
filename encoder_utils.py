# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 00:43:16 2018

@author: feng
"""

import os
import tensorflow as tf
import numpy as np

def load_data(path):
    labels = []
    images = []

    dirs = os.listdir(path)

    index = 0
    count = 0
    for d in dirs:
        dir_path = os.path.join(path, d)
        imgs = os.listdir(dir_path)
        num = len(imgs)
        label = [d] * num
        labels.extend(label)
        count += len(imgs)
        for img in imgs:
            img_path = os.path.join(dir_path, img)
            images.append(img_path)
        index += 1

    print("len(images): ", len(images))
    print("len(labels): ", len(labels))
    print("images[0]: ", images[0])
    print("labels[0]: ", labels[0])

    return images, labels

def get_per_attributes():
    """
    获取每个类别的属性，返回一个字典
    :return:
    """
    f = open("attributes_per_class.txt", "r")
    lines = f.readlines()
    
    per_attributes = {}
    for line in lines:
        l = line.strip().split('\t')
        ty = l[0]
        attributes = l[1:]
        num = []
        for i in range(30):
            num.append(float(attributes[i]))
        per_attributes[ty] = num
        
    return per_attributes
    
def read_and_decode(path, epochs):
    imgs, labels = load_data(path)

    per_attributes = get_per_attributes()
    label_list = []
    for i in range(len(labels)):
        label_list.append(per_attributes[labels[i]])  
    
    #label_list = tf.convert_to_tensor(label_arr, tf.float32)
    img_list = tf.cast(imgs, tf.string)
    filename_queue = tf.train.slice_input_producer([img_list, label_list], shuffle = True, num_epochs = epochs)  # 创建一个管道队列
    read_img = tf.read_file(filename_queue[0])
    decode_img = tf.image.decode_jpeg(read_img, channels = 3)
    resize_img = tf.image.resize_images(decode_img, [64, 64])
    label = filename_queue[1]
    #label = tf.string_to_number(label, tf.float32)

    return resize_img, label

if __name__ == '__main__':
    imgs, labels = read_and_decode('for_train', 5)
    feature_batch, label_batch = tf.train.shuffle_batch(
           [imgs, labels], batch_size = 5,
           capacity = 200, min_after_dequeue = 100)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        threads = tf.train.start_queue_runners(sess = sess)
        for i in range(3):
            print("i: ", i)
            f, l = sess.run([feature_batch, label_batch])
            print("f.shape: ", f.shape)

    