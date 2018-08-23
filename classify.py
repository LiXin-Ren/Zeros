# -*- coding: utf-8 -*-
from ResNet_Model import encoder, feature_attributes
import pandas as pd
import numpy as np

from scipy import ndimage
from scipy import misc
import tensorflow as tf
import cv2
import csv
import os

Batch_size = 1
test_image_dir = "/home/lisren/Zeros/DatasetA_test/test"
model_dir = "ResNet_model"

def get_per_attributes(attributes_file):
    attributes = pd.read_csv(attributes_file, index_col = 0)
    attributes = attributes.values[:, 1:]  # (230, 30)
    attributes = np.transpose(attributes)  # (30, 230)
    print("attributes.shape: ", attributes.shape)
        
    return attributes


# def get_pre_attributes_test(attributes_file):
#     attributes = pd.read_csv(attributes_file, index_col=0)
#     attributes = attributes.value[191:, 1:]
#     attributes = np.transpose(attributes)
def classfy(attri_pres):
    types = []
    
    for attri_pre in attri_pres:
        ty = max(attri_pre[200:])  
        types.append('ZJL'+str(ty))
    return types

def test_classfy(attri_pre, attris):
    res = []
    for i in range(len(attris)):
        dis = sum([(attri_pre[0][j] - attris[i][j])**2 for j in range(30)])
        res.append(dis)
   
    return 'ZJL'+str(res.index(min(res)) + 211)



def restore_model(sess, saver, model_dir):
    ckpt = tf.train.get_checkpoint_state(model_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        global_step = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1].split('.')[0])
    else:
        raise ValueError('no checkpoint file found!')
    return global_step


def test():
    img_Batch = tf.placeholder(dtype=tf.float32, shape=[None, 224, 224, 3])
    Attribute = tf.placeholder(dtype=tf.float32, shape=[30, 230])
    pred = encoder(img_Batch)
    attribute = get_per_attributes("sort_types_attrites.csv")
   # Attributes = tf.cast(attribute, tf.float32)
    W, logits = feature_attributes(pred, Attribute)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        global_step = restore_model(sess, saver, model_dir)
        print("The Total Training steps: %d" %(global_step))

        with open("DatasetA_test/image.txt", "r") as imgListFile:
            with open("submit.txt", "a") as submitFile:
                line = imgListFile.readline().strip()
                while line:
                    file_dir = os.path.join(test_image_dir, line)
                    print(file_dir)
                    if not os.path.exists(file_dir):
                        print("image not exists")
                        return 0
                    img = cv2.imread(os.path.join(test_image_dir, line))
                    
                   # print(os.path.join(test_image_dir, line))
                   # print(img.shape)

                    resize_img = cv2.resize(img, (224, 224))
                    resize_img = resize_img[np.newaxis, :, :, :] 
                    pred_value = sess.run(logits, feed_dict={img_Batch: resize_img, Attribute: attribute})
                   # print("pred: ", pred_value)
                    #print("pred.shape", pred_value.shape)

                    testAttri = list(pred_value[0][201:])
                   # print(testAttri)
                    res = testAttri.index(max(testAttri))
                    submitFile.write("%s\t%s\n" % (line, 'ZJL'+str(res+201)))
                    line = imgListFile.readline().strip()

def main(argv=None):
    test()


if __name__ == '__main__':
    tf.app.run()

