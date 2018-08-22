import tensorflow as tf

def vgg_encoder_black(input_x, filters, black_name, is_training):
    conv1 = tf.layers.conv2d(input_x, filters, (3, 3), padding = 'same', activation = None, name = black_name + 'conv1')
    norm1 = tf.layers.batch_normalization(conv1, training = is_training, name = black_name + 'norm1')
    action1 = tf.nn.relu(norm1, name = black_name + 'action1')
    conv2 = tf.layers.conv2d(action1, filters, (3, 3), padding = 'same', activation = None, name = black_name + 'conv2')
    norm2 = tf.layers.batch_normalization(conv2, training = is_training, name = black_name + 'norm2')
    action2 = tf.nn.relu(norm2, name = black_name + 'action2')
    conv3 = tf.layers.conv2d(action2, filters, (3, 3), padding = 'same', activation = None, name = black_name + 'conv3')
    norm3 = tf.layers.batch_normalization(conv3, training = is_training, name = black_name + 'norm3')
    action3 = tf.nn.relu(norm3, name = black_name + 'action3')
    conv4 = tf.layers.conv2d(action3, filters, (3, 3), padding = 'same', activation = None, name = black_name + 'conv4')
    norm4 = tf.layers.batch_normalization(conv4, training = is_training, name = black_name + 'norm4')
    action4 = tf.nn.relu(norm4, name = black_name + 'action4')
    pool1 = tf.layers.max_pooling2d(action4, (2, 2), (2, 2), padding = 'same', name = black_name + 'pool1')

    return pool1


def encoder(input_x, is_training):
    vgg_1 = vgg_encoder_black(input_x, 16, "vgg1", is_training)  # 112x112x16
    vgg_2 = vgg_encoder_black(vgg_1, 32, "vgg2", is_training)    # 56x56x32
    vgg_3 = vgg_encoder_black(vgg_2, 32, "vgg3", is_training)    # 28x28x32
    vgg_4 = vgg_encoder_black(vgg_3, 64, "vgg4", is_training)    # 14x14x64
    vgg_5 = vgg_encoder_black(vgg_4, 64, "vgg5", is_training)    # 7x7x64
    flat = tf.reshape(vgg_5, [-1, 7*7*64], name = 'flat')  # 3136
    fcb = tf.layers.dense(flat, 1024, activation = None, name = 'fcb1')

    return fcb

def vgg_decoder_black(input_x, filters,size, black_name, is_training):
    decoder_unsample1 = tf.image.resize_nearest_neighbor(input_x, (size, size), name = black_name + 'unsample1')
    decoder_conv1 = tf.layers.conv2d(decoder_unsample1, filters, (3, 3), padding = 'same', activation = None, name = black_name + 'decoder_conv1')
    decoder_norm1 = tf.layers.batch_normalization(decoder_conv1, training = is_training, name = black_name + 'decoder_norm1')
    decoder_action1 = tf.nn.relu(decoder_norm1, name = black_name + 'decoder_action1')
    decoder_conv2 = tf.layers.conv2d(decoder_action1, filters, (3, 3), padding = 'same', activation = None, name = black_name + 'decoder_conv2')
    decoder_norm2 = tf.layers.batch_normalization(decoder_conv2, training = is_training, name = black_name + 'decoder_norm2')
    decoder_action2 = tf.nn.relu(decoder_norm2, name = black_name + 'decoder_action2')
    decoder_conv3 = tf.layers.conv2d(decoder_action2, filters, (3, 3), padding = 'same', activation = None, name = black_name + 'decoder_conv3')
    decoder_norm3 = tf.layers.batch_normalization(decoder_conv3, training = is_training, name = black_name + 'decoder_norm3')
    decoder_action3 = tf.nn.relu(decoder_norm3, name = black_name + 'decoder_action3')
    decoder_conv4 = tf.layers.conv2d(decoder_action3, filters, (3, 3), padding = 'same', activation = None, name = black_name + 'decoder_conv4')
    decoder_norm4 = tf.layers.batch_normalization(decoder_conv4, training = is_training, name = black_name + 'decoder_norm4')
    decoder_action4 = tf.nn.relu(decoder_norm4, name = black_name + 'decoder_action4')

    return decoder_action4


def decoder(input_x, is_training):
    # in: 4x4x64 out 8x8x64
    decoder_1 = vgg_decoder_black(input_x, 64, 8, "decoder1", is_training)
    decoder_2 = vgg_decoder_black(decoder_1, 32, 16, "decoder2", is_training)
    decoder_3 = vgg_decoder_black(decoder_2, 32, 32, "decoder3", is_training)
    decoder_4 = vgg_decoder_black(decoder_3, 16, 64, "decoder4", is_training)
    decoder_out = tf.layers.conv2d(decoder_4, 3, (3, 3), padding = 'same', activation = tf.nn.relu, name = 'decoder_conv16')

    return decoder_out

def feature_attrites(vgg_out, attrites):
    W = tf.layers.dense(vgg_out, 30, activation = None, name = 'W')
    logits = tf.matmul(W, attrites, name = "attrites_W")

    return W, logits 
