# -*- coding: utf-8 -*-

import os
import tensorflow as tf
import numpy as np
import pandas as pd

def load_data(path, index_file):
    f = open(index_file, "r")
    lines = f.readlines()
    type_index = [line.strip() for line in lines]
    labels = []
    images = []

    dirs = os.listdir(path)

    index = 0
    count = 0
    for d in dirs:
        dir_path = os.path.join(path, d)
        imgs = os.listdir(dir_path)
        num = len(imgs)
        label = [type_index.index(d)] * num
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
    #labels = np.array(labels)
    #labels = np.reshape(labels, [-1, 1])

    return images, labels

def get_per_attributes(attributes_file):
    attributes = pd.read_csv(attributes_file, index_col = 0)
    attributes = attributes.values[:, 1:]  # (230, 30)
    attributes = np.transpose(attributes)  # (30, 230)
    print("attributes.shape: ", attributes.shape)
        
    return attributes
    
def read_and_decode(path, index_file, epochs):
    imgs, labels = load_data(path, index_file)

    imgs_list = tf.cast(imgs, tf.string)
    labels_list = tf.cast(labels, tf.int32)
    filename_queue = tf.train.slice_input_producer([imgs_list, labels_list], shuffle = True, num_epochs = epochs)  # 创建一个管道队列
    read_img = tf.read_file(filename_queue[0])
    decode_img = tf.image.decode_jpeg(read_img, channels = 3)
    resize_img = tf.image.resize_images(decode_img, [224, 224])
    flip1_img = tf.image.random_flip_up_down(resize_img)
    flip2_img = tf.image.random_flip_left_right(flip1_img)
    label_list = filename_queue[1]

    return flip2_img, label_list

if __name__ == '__main__':
    attribute = get_per_attributes("sort_types_attrites.csv")
    imgs, labels = read_and_decode('for_trai', 'type_index.txt', 5)
    feature_batch, label_batch = tf.train.shuffle_batch(
           [imgs, labels], batch_size = 5,
           capacity = 200, min_after_dequeue = 100)
    np_l = tf.one_hot(label_batch, 230, axis = 1)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        threads = tf.train.start_queue_runners(sess = sess)
        for i in range(3):
            print("i: ", i)
            f, l, n = sess.run([feature_batch, label_batch, np_l])
            print("f.shape: ", f.shape)
            print("l: ", l)
            print("n:\n", n)

    
