# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function
import os
import tensorflow as tf
import model

def get_per_attributes(attribute_file):
    """
    获取每个类别的属性，返回一个字典
    :return:
    """
    f = open(attribute_file, "r")
    lines = f.readlines()

    per_attributes = {}
    for line in lines:
        l = line.strip().split('\t')
        ty = l[0]
        attributes = l[1:]
        num = []
        for i in range(24):
            num.append(float(attributes[i]))
        per_attributes[ty] = num
    return per_attributes


def GetFileNameList(imageDir):
    """

    :param path:  训练集
    :return:
    image：
    classLabel:
    attriLabel:
    """
    attriLabel = []
    classLabel = []
    images = []
    attriDictTrain = get_per_attributes("../DatasetA_train/train_class_attribute.txt")
    dirs = os.listdir(imageDir)
    for d in dirs:
        dir_path = os.path.join(imageDir, d)
        imgs = os.listdir(dir_path)
        imgNum = len(imgs)
        classes = [d for i in range(imgNum)]                   #类别标签
        attri = [attriDictTrain[d] for i in range(imgNum)]     #属性标签
        attriLabel.extend(attri)
        classLabel.extend(classes)
        for img in imgs:
            img_path = os.path.join(dir_path, img)
            images.append(img_path)

    # print("len(images): ", len(images))
    # print("len(labels): ", len(labels))
    # print("images[0]: ", images[0])
    # print("labels[0]: ", labels[0])
    return images, classLabel, attriLabel

def GetBatchFromFile_Train(imageList, classLabel, attriLabel, BatchSize, epochs):
    '''i
    Args:
        rawDir: Directory of raw and segmante images
        BatchSize: batch size
    Returns:
        image_batch: 4D tensor [batch_size, width, height, 1], dtype=tf.float32
        label_batch: 4D tensor [batch_size, width, height, 1], dtype=tf.float32
    '''
    # rawImageList, segImageList = GetFileNameList(rawDir)

    imageList = tf.cast(imageList, tf.string)
    classLabel = tf.cast(classLabel, tf.string)
    attriLabel = tf.cast(attriLabel, tf.float32)

    # Make an input queue
    InputQueue = tf.train.slice_input_producer([imageList, classLabel, attriLabel],
                                               num_epochs=epochs,
                                               shuffle=True)

    # Read one example from input queue
    imageContent = tf.read_file(InputQueue[0])
    classContent = InputQueue[1]
    attriContent = InputQueue[2]

    # Decode the jpeg image format
    decode_image = tf.image.decode_image(imageContent, channels=3)
    #resize_image = tf.image.resize_images(decode_image, [64, 64])
    decode_image.set_shape([64,64,3])
    imageBatch, classBatch, attriBatch = tf.train.shuffle_batch([decode_image, classContent, attriContent],
                                                          batch_size=BatchSize,
                                                          capacity=10000,
                                                          min_after_dequeue=200)

    # Normalization
    imageBatch = tf.cast(imageBatch, tf.float32)
    #imageBatch = imageBatch / 255.0
    return imageBatch, classBatch, attriBatch


def GetBatchFromFile_Valid(GIBBSDir, CLEARDir, BatchSize):
    ''' Get batch from files for validation.
    Args:
        GIBBSDir: Directory of GIBBS images
        CLEARDir: Directory of CLEAR images
        BatchSize: batch size
    Returns:
        image_batch: 4D tensor [batch_size, height, width, 1], dtype=tf.float32
        label_batch: 4D tensor [batch_size, height, width, 1], dtype=tf.float32
    '''
    GIBBSImageList = tf.cast(GIBBSDir, tf.string, name='CastGIBBSFileName')
    CLEARImageList = tf.cast(CLEARDir, tf.string, name='CastCLEARFileName')
    if GIBBSImageList.shape[0].value != CLEARImageList.shape[0].value:
        print('Dimension Error --> NameList')
        return

    NUM_EXAMPLES = GIBBSImageList.shape[0].value
    print("Total Training Examples: %d" % NUM_EXAMPLES)

    InputQueue = tf.train.slice_input_producer([GIBBSImageList, CLEARImageList, CLEARImageList],
                                               num_epochs=None,  # validate once or not?
                                               shuffle=False,  # DO NOT shuffle!
                                               capacity=32,
                                               shared_name=None,
                                               name='StringInputProducer')

    # Read one example from input queue

    GIBBSImageContent = tf.read_file(InputQueue[0], name='ReadGIBBSImage')
    CLEARImageContent = tf.read_file(InputQueue[1], name='ReadCLEARImage')
    name = InputQueue[2]
    # Decode the png image format
    GIBBSImage = tf.image.decode_image(GIBBSImageContent, channels=1, name='DecodeGIBBSImage')
    CLEARImage = tf.image.decode_image(CLEARImageContent, channels=1, name='DecodeCLEARImage')

    MIN_FRACTION_EXAMPLE_IN_QUEUE = 0.05
    MIN_EXAMPLES_IN_QUEUE = int(NUM_EXAMPLES * MIN_FRACTION_EXAMPLE_IN_QUEUE)
    print('Filling queue with %d/%d images. This will take a few minutes.' %
          (MIN_EXAMPLES_IN_QUEUE, NUM_EXAMPLES))

    # Set shape for images
    with tf.name_scope('SetShape'):
        GIBBSImage.set_shape([None, None, 1])
        CLEARImage.set_shape([None, None, 1])

    GIBBSImageBatch, CLEARImageBatch, name = tf.train.batch([GIBBSImage, CLEARImage, name],
                                                            batch_size=BatchSize,
                                                            num_threads=1,  # set to 1 to keep order.
                                                            capacity=MIN_EXAMPLES_IN_QUEUE + 10 * BatchSize,
                                                            dynamic_pad=True,
                                                            name='Batch')
    tf.summary.image('val_GIBBS_images', GIBBSImageBatch, max_outputs=4)
    tf.summary.image('val_CLEAR_images', CLEARImageBatch, max_outputs=4)

    # Cast to tf.float32
    with tf.name_scope('CastToFloat'):
        GIBBSImageBatch = tf.cast(GIBBSImageBatch, tf.float32)
        CLEARImageBatch = tf.cast(CLEARImageBatch, tf.float32)

    # Normalization
    with tf.name_scope('Normalization'):
        GIBBSImageBatch = GIBBSImageBatch / 255.0
        CLEARImageBatch = CLEARImageBatch / 255.0

    return GIBBSImageBatch, CLEARImageBatch


if __name__ == '__main__':

    picList, classList, attriList = GetFileNameList("for_train")  # 图像地址list和标签属性list
    imageBatch, classBatch, attriBatch = GetBatchFromFile_Train(picList, classList, attriList, 16)
    predAttriBatch = model.plain_model(imageBatch)    #预测属性

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        threads = tf.train.start_queue_runners(sess = sess)
        for i in range(3):
            print("i: ", i)
            img, classB, trueAttri, pred = sess.run([imageBatch, classBatch, attriBatch,  predAttriBatch])
            print("f.shape: ", img.shape)
            print("classBatch",classB)
            print("trueAttri", trueAttri)
            print("predAttri",pred)

