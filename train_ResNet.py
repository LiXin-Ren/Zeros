# -*- coding: utf-8 -*-
from ResNet_Model import ResNet_34, feature_attributes
from utils import read_and_decode, get_per_attributes
import tensorflow as tf
import sklearn
import pandas as pd

def train(path, index_file, lr, epochs, batch_size, attribute):
    """
    :param path:  训练集路径
    :param index_file: ZJL1-ZJL240 类别名称
    :param attribute: 类别对应的属性，shape=[30, None]
    :return:
    """
    Attributes = tf.placeholder(tf.float32, (30, 200))
    imgs, labels = read_and_decode(path, index_file, epochs)
    #batch_X, batch_Y = tf.train.shuffle_batch([imgs, labels], batch_size=batch_size, capacity=10000, min_after_dequeue=200)
    batch_X, batch_Y = tf.train.shuffle_batch([imgs, labels], batch_size=batch_size, capacity=10000, min_after_dequeue=200)
#    print("batch_X.shape", batch_X.shape)   #(batch_size, 224, 224, 3)
#    print("batch_Y.shape", batch_Y.shape)   #(batch_size,)
    ResNet_out = ResNet_34(batch_X)

    logits = tf.nn.softmax(feature_attributes(ResNet_out, Attributes))
#    print("logits.shape", logits.shape)    #(batch_size, 200)
    one_y = tf.one_hot(batch_Y, 200, axis=1)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=one_y))
   # cost = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=one_y)
    optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(cost)
    acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, 1), tf.argmax(one_y, 1)), tf.float32))
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        epoch_i = 0
        print("begin")
        sess.run(optimizer, feed_dict={Attributes: attribute})
 
        max_acc = 0
        min_loss = float('Inf')
        try:
            while not coord.should_stop():
                costs, accs = sess.run([cost, acc], feed_dict={Attributes: attribute})

                if epoch_i % 100 == 0:
                    if max_acc < accs:
                        max_acc = accs
                    if min_loss > costs:
                        min_loss = costs 
                    with open("Records/train_ResNet.txt", "a") as fileName:
                        format_str = "%d\t%.6f\t%.6f\t%.6f\t%.6f\n"
                        fileName.write(format_str %(epoch_i, costs, min_loss, accs, max_acc))
                        print("epoch_i: %d\t cost: %.6f\t acc: %.6f"%(epoch_i, costs, accs))
                    
                if epoch_i % 500 == 0:
                    saver.save(sess, "ResNet_model/model-%s.ckpt" % (str(epoch_i)))
                epoch_i = epoch_i + 1
       # except tf.errors.OutOfRangeError:
        except Exception as e:
            print("Done training -- epoch limit reached")
        finally:
            saver.save(sess, "ResNet_model/model-%s.ckpt" % (str(epoch_i)))
            coord.request_stop()
            coord.join(threads)

if __name__ == '__main__':
    path = 'for_train'
    lr = 0.001
    batch_size = 64
    epochs = 500
    attribute = get_per_attributes("sort_types_attrites.csv")       #(30, 200)
    index_file = "type_index.txt"
    print(attribute)
    train(path, index_file, lr, epochs, batch_size, attribute)

