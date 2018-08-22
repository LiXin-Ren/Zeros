# -*- coding: utf-8 -*-

from vgg19_W import *
from utils import *
import tensorflow as tf
import sklearn
import pandas as pd

def autoencoder_model(path, index_file, lr, epochs, batch_size, attribute):
    X = tf.placeholder(tf.float32, [None, 224, 224, 3], name = 'place_x')
    Y = tf.placeholder(tf.int64, [None], name = 'place_y')
    Attrites = tf.placeholder(tf.float32, [30, 230], name = 'Attrites')
    imgs, labels = read_and_decode(path, index_file, epochs)
    batch_X, batch_Y = tf.train.shuffle_batch([imgs, labels], batch_size = batch_size, capacity = 10000, min_after_dequeue = 200)
    vgg_out = encoder(X, is_training = True)    
    W, logits = feature_attrites(vgg_out, Attrites)

    one_y = tf.one_hot(Y, 230, axis = 1)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = one_y))
    optimizer = tf.train.AdamOptimizer(lr).minimize(cost)
    acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits), tf.argmax(one_y)), tf.float32))
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        print("initialize")
        
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess = sess, coord = coord)
        epoch_i = 0
        print("begin")
        try:
            while not coord.should_stop():
                batch_x, batch_y = sess.run([batch_X, batch_Y])
                _, costs, accs = sess.run([optimizer, cost, acc], feed_dict = {X: batch_x, Y: batch_y, Attrites: attribute}) 
                
                if epoch_i % 100 == 0:
                    print("epoch_i: ", epoch_i, "  costs: ", costs, " accs: ", accs)
                if epoch_i % 500 == 0:
                    saver.save(sess, "vgg19W_model/model-%s.ckpt" %(str(epoch_i)))
                epoch_i = epoch_i + 1
        except tf.errors.OutOfRangeError:
            print("Done training -- epoch limit reached")
        finally:
            saver.save(sess, "vgg19W_model/model-%s.ckpt" %(str(epoch_i)))
            coord.request_stop()    
            coord.join(threads)

if __name__ == '__main__':
    path = 'for_train'
    lr = 0.001
    batch_size = 64
    epochs = 500
    attribute = get_per_attributes("sort_types_attrites.csv")
    index_file = "type_index.txt"
    autoencoder_model(path, index_file, lr, epochs, batch_size, attribute)
