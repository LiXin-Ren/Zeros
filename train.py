# -*- coding: utf-8 -*-
from __future__ import absolute_import
from datetime import datetime
import tensorflow as tf
import inputs
import os
import model
import numpy as np
DATA_TRAIN = "for_train"
DATA_TEST = "for_teset"
BATCH_SIZE = 16
LearningRate = 1e-4
LogDir = "LogDir"
def restore_model(sess, saver, ExistModelDir, global_step):
    log_info = "Restoring Model From %s..." % ExistModelDir
    print(log_info)
    ckpt = tf.train.get_checkpoint_state(ExistModelDir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        init_step = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
        sess.run(tf.assign(global_step, init_step))
    else:
        print('No Checkpoint File Found!')
        return 0
    return init_step

def train():
    global_step = tf.train.get_or_create_global_step()

    with tf.device('/cpu:0'):
        picList, classList, attriList = inputs.GetFileNameList(DATA_TRAIN)     #图像地址list和标签属性list
        imageBatch, classBatch, attriBatch = inputs.GetBatchFromFile_Train(picList, classList, attriList, BATCH_SIZE)

    # Build a Graph that computes the predicted HR images from GIBBS RING CLEAR model.
    predAttriBatch = model.Model(imageBatch)    #预测属性

    # Calculate loss.
    loss = model.loss(attriBatch, predAttriBatch)  #

    # Get the training op for optimizing loss
    TrainOp = model.optimize(loss, LearningRate, global_step)

    # Create a saver.
    saver = tf.train.Saver(tf.global_variables())

    # Start running operations on the Graph. allow_soft_placement must be set to
    # True to build towers on GPU, as some of the ops do not have GPU implementations.
    config = tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)

    with tf.Session(config=config) as sess:
        print("Initializing Variables...")
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        # queue runners, multi threads and coordinator
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        min_loss = float('Inf')
        max_acc = 0
        step = 1
        try:
            print("Starting To Train...")

            sess.run(TrainOp)

            if step % 100 == 0:
                loss_value, classValue, predAttriValue= sess.run([loss, classBatch, predAttriBatch])
                accuracy = model.accuracy(classValue, predAttriValue)
                if min_loss > loss_value:
                    min_loss = loss_value
                if max_acc < accuracy:
                    max_acc = accuracy

                with open("Records/train_records.txt", "a") as file:
                    format_str = "%d\t%.6f\t%.6f\t%.6f\t%.6f\n"
                    file.write(str(format_str) % (
                        step + 1, loss_value, min_loss, accuracy, max_acc))

                print("%s ---- step %d:" % (datetime.now(), step + 1))
                print("\tLOSS = %.6f\tmin_Loss = %.6f" % (loss_value, min_loss))
                print("\tACC = %.4f\tmax_Acc = %.4f" % (accuracy, max_acc))

            if (step == 0) or ((step + 1) % 200 == 0):
                checkpoint_path = os.path.join(LogDir, 'model.ckpt')
                print("saving checkpoint into %s-%d" % (checkpoint_path, step + 1))
                saver.save(sess, checkpoint_path, global_step=step + 1)

        except Exception as e:
            print("exception: ", e)
            coord.request_stop(e)

        finally:
            coord.request_stop()
            coord.join(threads, stop_grace_period_secs=10)


def main(argv=None):  # pylint: disable = unused - argument
    if tf.gfile.Exists(LogDir):
        tf.gfile.DeleteRecursively(LogDir)
    tf.gfile.MakeDirs(LogDir)
    train()


if __name__ == '__main__':
    tf.app.run()













