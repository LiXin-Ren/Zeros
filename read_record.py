import tensorflow as tf
import functools
import random
import os

def get_imgs_labels(path):
    labels = []
    images = []

    dirs = os.listdir(path)

    index = 0
    count = 0
    for d in dirs:
        dir_path = os.path.join(path, d)
        print("dir_path: ", dir_path)
        imgs = os.listdir(dir_path)
        num = len(imgs)
        label = [index] * num
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

def read_and_decode(path, rsize):
    imgs, labels = get_imgs_labels(path)
    img_list = tf.cast(imgs, tf.string)
    label_list = tf.cast(labels, tf.int32)
    filename_queue = tf.train.slice_input_producer([img_list, label_list], shuffle = True, num_epochs = 50)  # 创建一个管道队列
    read_img = tf.read_file(filename_queue[0])
    decode_img = tf.image.decode_jpeg(read_img, channels = 3)
    resize_img = tf.image.resize_images(decode_img, [rsize, rsize])
    label = filename_queue[1]
    #label = tf.string_to_number(label, tf.float32)

    return resize_img, label

if __name__ == '__main__':
    path = "detection_for_train"
    imgs, labels = get_imgs_labels(path)
    img = read_and_decode(imgs, labels)
    #labels = list(labels)

    feature_batch = tf.train.shuffle_batch(
           [img], batch_size = 5,
           capacity = 200, min_after_dequeue = 100)
    init = tf.initialize_all_variables()

    with tf.Session() as sess:
        sess.run(init)
        threads = tf.train.start_queue_runners(sess = sess)
        for i in range(3):
            f = sess.run([feature_batch])
            print("f:\n", f)
