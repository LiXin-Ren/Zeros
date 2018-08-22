import tensorflow as tf
import cv2
import numpy as np

def res_block(inputs, filters, kernel_size, stride, block_num):
    """
    block_num: the number of blockes
    """
    shortcut = inputs
    for i in range(block_num):
       # conv1 = tf.layers.conv2d(shortcut, filters, kernel_size, padding="same", activation=tf.nn.relu)
        conv1 = conv2d_same(shortcut, filters, kernel_size, stride)
       #  conv2 = tf.layers.conv2d(conv1, filters, kernel_size, padding = "same", activation=tf.nn.relu)
        conv2 = conv2d_same(conv1, filters, kernel_size, stride)
        shortcut = conv2 + shortcut

    return conv2

def conv2d_same(inputs, num_outputs, kernel_size, stride=1):

    """
      inputs: A 4-D tensor of size [batch, height_in, width_in, channels].
      num_outputs: An integer, the number of output filters.
      kernel_size: An int with the kernel_size of the filters.
      stride: An integer, the output stride.
    Returns:
      output: A 4-D tensor of size [batch, height_out, width_out, channels] with
        the convolution output.
        """
    if stride == 1:
        return tf.layers.conv2d(inputs, num_outputs, [kernel_size, kernel_size], [stride, stride], padding="same", activation=tf.nn.relu)
    else:
        pad_total = kernel_size - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])
        return tf.layers.conv2d(inputs, num_outputs, [kernel_size, kernel_size], [stride, stride], padding="valid", activation=tf.nn.relu)

def encoder(inputs):
    """
        resnet_50: 3-4-6-3
        :param inputs:
        :return:
    """
    # conv1
    # input:224*224*3, output:112*112*64
    conv1 = conv2d_same(inputs, 64, 7, 2)       #112*112*64 conv+relu

    #block1   inputs: 112*112*64 output: 56*56*64 块1的channel都为64 重复3次
    pool1 = tf.layers.max_pooling2d(conv1, [2, 2], strides=[2, 2], padding="valid")         #池化：56*56*64
    # block1 = tf.layers.conv2d(pool1, 64, [3, 3], strides=[2, 2], padding="valid", activation=tf.nn.relu) #56*56*64
    block1 = res_block(pool1, filters=64, kernel_size=3, stride=1, block_num=2)       #ouput:56*56*64

    #block2 input: 56*56*64  output: 28*28*128  重复4次
    #先进行一个下采样
    block2_conv1_1 = conv2d_same(block1, num_outputs=128, kernel_size=3, stride=2) #28*28*128
    block2_conv1_2 = conv2d_same(block2_conv1_1, num_outputs=128, kernel_size=3)
    block2 = res_block(block2_conv1_2, filters=128, kernel_size=3, stride=1, block_num=3)

    #block 3: input: 28*28*64 output:14*14*256  重复6次
    block3_conv1_1 = conv2d_same(block2, num_outputs=256, kernel_size=3, stride=2) #14*14*256
    block3_conv1_2 = conv2d_same(block3_conv1_1, num_outputs=256, kernel_size=3)
    block3 = res_block(block3_conv1_2, filters=256, kernel_size=3, stride=1, block_num=5)

    # block 4: input: 14*14*256  output:7*7*512 重复3次
    block4_conv1_1 = conv2d_same(block3, num_outputs=512, kernel_size=3, stride=2) #7*7*512
    block4_conv1_2 = conv2d_same(block4_conv1_1, num_outputs=512, kernel_size=3)
    block4 = res_block(block4_conv1_2, filters=512, kernel_size=3, stride=1, block_num=2)

    flat = tf.reshape(block4, [-1, 7*7*512], name = 'flat')  # 25088
    fcb = tf.layers.dense(flat, 1024, activation = None, name = 'fcb1')

    return fcb


def feature_attrites(vgg_out, attrites):
    W = tf.layers.dense(vgg_out, 30, activation = None, name = 'W')
    logits = tf.matmul(W, attrites, name = "attrites_W")

    return W, logits

if __name__ == "__main__":
    X = tf.placeholder(tf.float32, [None, 224, 224, 3], name="x")
    img = cv2.imread("1.jpeg")

    #decode_img = tf.image.decode_image(img, channels = 3)
    resize_image = cv2.resize(img, (224, 224))
    resize_image = resize_image[np.newaxis, :, :, :]
    resize_image = tf.cast(resize_image, tf.float32)
    res = encoder(X)
    #res = resnet_v1.resnet_v1_50(X)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        res_value = sess.run(res, feed_dict={X: resize_image})

        print(res_value)

