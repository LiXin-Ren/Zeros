# -*- coding: utf-8 -*-
import os
import tensorflow as tf
import numpy as np
import pandas as pd

def GetFileList(path, index_file):
    """
    :param path: 训练集，按类别存放
    :param index_file: type_index file, 类别名称
    :return: 图像名称列表，以及对应的标签列表
    """
    f = open(index_file, "r")
    lines = f.readlines()
    type_index = [line.strip() for line in lines]
    labels = []
    images = []
    dirs = os.listdir(path)

    index = 0
    for d in dirs:
        dir_path = os.path.join(path, d)
        imgs = os.listdir(dir_path)
        num = len(imgs)
        label = [type_index.index(d)] * num    #标签为index（转为数字）
        labels.extend(label)
        for img in imgs:
            img_path = os.path.join(dir_path, img)
            images.append(img_path)
        index += 1

    return images, labels


def get_per_attributes(attributes_file):
    """
    :param attributes_file: csv 文件，类别与其30个属性
    :return: 类的属性值，shape=(30, N)
    """
    attributes = pd.read_csv(attributes_file, index_col = 0)
    attributes = attributes.values[:, 1:]  # (230, 30)
    attributes = np.transpose(attributes)  # (30, 230)
    print("attributes.shape: ", attributes.shape)
        
    return attributes


def read_and_decode(path, index_file, epochs):
    """
    读取图像，生成队列，裁剪图像
    :param path:  训练集
    :param index_file: 训练集类别名称
    :param epochs:
    :return:
    """
    imgList, labelList = GetFileList(path, index_file)

    imgList = tf.cast(imgList, tf.string)
    labelList = tf.cast(labelList, tf.int32)
    assert imgList.shape[0] == labelList.shape[0], "Dimension Error --> NameList's length is not equal to labelList"

    inputQueue = tf.train.slice_input_producer([imgList, labelList], shuffle = True, num_epochs = epochs)  # 创建一个管道队列
    imgQueue = tf.read_file(inputQueue[0])
    labelQueue = inputQueue[1]
    decode_img = tf.image.decode_jpeg(imgQueue, channels = 3)
    resize_img = tf.image.resize_images(decode_img, [224, 224])
    #flip1_img = tf.image.random_flip_up_down(resize_img)
    #flip2_img = tf.image.random_flip_left_right(flip1_img)

    return resize_img, labelQueue

if __name__ == '__main__':
    attribute = get_per_attributes("sort_types_attrites.csv")
    imgs, labels = read_and_decode('for_train', 'type_index.txt', 5)

    feature_batch, label_batch = tf.train.shuffle_batch(
           [imgs, labels], batch_size = 5,
           capacity = 200, min_after_dequeue = 100)
    np_l = tf.one_hot(label_batch, 200, axis = 1)

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

    
