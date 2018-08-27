import tensorflow as tf
import cv2
import numpy as np
import inputs

def plain_model(inputs):
    """
    实现一个简单的模型
    :param inputs: （None, 64, 64, 3)
    :return:
    """
    # in: 64x64x3 out: 32x32x32
    conv1 = tf.layers.conv2d(inputs, 32, (3, 3), padding='same', activation=tf.nn.relu, kernel_initializer=tf.initializers.truncated_normal)
    pool1 = tf.layers.max_pooling2d(conv1, (2, 2), (2, 2), padding='same')

    # in: 32x32x32 out: 16x16x64
    conv2 = tf.layers.conv2d(pool1, 64, (3, 3), padding='same', activation=tf.nn.relu, kernel_initializer=tf.initializers.truncated_normal)
    pool2 = tf.layers.max_pooling2d(conv2, (2, 2), (2, 2), padding='same')

    # in: 16x16x64 out: 8x8x128
    conv3 = tf.layers.conv2d(pool2, 128, (3, 3), padding='same', activation=tf.nn.relu, kernel_initializer=tf.initializers.truncated_normal)
    pool3 = tf.layers.max_pooling2d(conv3, (2, 2), (2, 2), padding='same')

    # in: 8x8x128 out: 4x4x512
    conv4 = tf.layers.conv2d(pool3, 512, (3, 3), padding='same', activation=tf.nn.relu, kernel_initializer=tf.initializers.truncated_normal)
    pool4 = tf.layers.max_pooling2d(conv4, (2, 2), (2, 2), padding='same')

    # dense
    flat = tf.reshape(pool4, [-1, 4 * 4 * 512], name='flat')  # 8192
    fcb1 = tf.layers.dense(flat, 1024, activation=tf.nn.relu, name='fcb1')
    fcb2 = tf.layers.dense(fcb1, 24, activation=tf.nn.sigmoid)  # (None, 30)

    return fcb2


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

def ResNet_34(inputs):
    """
        resnet_50: 3-4-6-3
        :param inputs:
        :return:
    """
    # conv1
    # input:224*224*3, output:112*112*64
    conv1 = conv2d_same(inputs, 64, 7, 2)       #112*112*64 conv+relu

    #block1   inputs: 112*112*64 output: 56*56*64 块1的channel都为64 重复3次
    pool1 = tf.layers.max_pooling2d(conv1, [3, 3], strides=[2, 2], padding="same")         #池化：56*56*64
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
    print("block4.shape:", block4.shape)
    flat = tf.reshape(block4, [-1, 7*7*512], name = 'flat')  # 25088
    fcb1 = tf.layers.dense(flat, 1024, activation = tf.nn.relu, name = 'fcb1')
    fcb2 = tf.layers.dense(fcb1, 30, activation = tf.nn.relu)     #(None, 30)
   # print("res_out.shape", fcb2.shape)
    return fcb2

def feature_attributes(net_out, attributes):
    """
    建立网络的输出(视觉潜入)与属性（语义潜入）的联系。
    :param net_out:
    :param attributes:
    :return:
    """
    logits = tf.matmul(net_out, attributes, name = "attributes_W")    #(None, 200)
   # logits = tf.nn.sigmoid(logits)
 #   print("attribute:", attributes.shape)     #(30, 200)
 #   print("logit: ", logits.shape)
    return logits
"""
def loss(logits, labels):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels))
"""

def loss(logits, labels):
    return tf.losses.mean_squared_error(labels, logits)

def accuracy(classLabel, predAttri):
    """
    classLabel: [Batch_size, 1]
    predAttri: [Batch_size, 30]
    """
    attributionDict = inputs.get_per_attributes("../DatasetA_train/train_class_attribute.txt")
    res = {}
    acc = 0
   # print("classLabel.shape:", classLabel.shape)     #(batch, )
   # print("predAttri.shape: ", predAttri.shape)      #(batch, 30)
    for i in range(len(classLabel)):
        for (category, attri) in attributionDict.items():
    #        print(attri[0])
     #       print(predAttri[0])
           # print(type(attri), type(attri[0]))
            #print("attri.shape: ", len(attri))   #30
            dis = sum([(predAttri[i][j] - attri[j]) ** 2 for j in range(24)])
            res[category] = dis     #每种类别的距离
        predClass = max(res, key=res.get)
        print("predClass: ", predClass)
        print("classLabel[i]: ", classLabel[i])
       # print("predClass is ", predClass)
       # print("labelCLass is ", classLabel[i])
        if predClass == classLabel[i]:
            acc += 1
    return acc/len(classLabel)

def optimize(loss, learning_rate, global_step):
    """Sets up the training Ops.

    Args:
        loss: Loss tensor, from loss().
        lr: The learning rate to use for gradient descent.

    Returns:
        train_op: The Op for training.
    """

    with tf.name_scope('optimization'):
#        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)
 
       # with a global_step to track the global step.
        train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op


if __name__ == "__main__":
    X = tf.placeholder(tf.float32, [None, 224, 224, 3])
    img = cv2.imread("1.jpeg")

    #decode_img = tf.image.decode_image(img, channels = 3)
    resize_image = cv2.resize(img, (224, 224))
    resize_image = resize_image[np.newaxis, :, :, :]
    res = ResNet_34(X)
    #res = resnet_v1.resnet_v1_50(X)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        res_value = sess.run(res, feed_dict={X: resize_image})

        print(res_value)

