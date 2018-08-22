from utils import *
#from vgg19_W import *
from ResNet_Model import  *
import tensorflow as tf
import sklearn
from classify import *
from scipy import misc
import math
import cv2

def mini_batches(X, mini_batch_size = 64, seed = 0):
    m = X.shape[0]                  # number of training examples
    mini_batches = []
    permutation = list(range(m))

    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch = permutation[k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch = permutation[num_complete_minibatches * mini_batch_size : m]
        mini_batches.append(mini_batch)
    
    return mini_batches

def load_predict_data(path, indexfile):
    f = open(indexfile, "r")
    imgs = f.readlines()
    imgs_index = [] 
    predict_img_batch = np.zeros((len(imgs), 224, 224, 3))
    resize_imgs = []
    with tf.Session() as sess:
        for img in range(0, len(imgs)):
            img_index = imgs[img].strip()
            imgs_index.append(img_index)
            img_path = os.path.join(path, img_index)
            read_img = cv2.imread(img_path) / 255.   # 灰度图读出来也是三维的
            read_img = cv2.resize(read_img, (224, 224)) 
            #read_img = (misc.imread(img_path)/255.).astype(np.float32)
            #print("read_img,shape: ", read_img.shape)
            #read_img = read_img[np.newaxis, :, :, :]    # 灰度图用这种方法处理会出错
            #read_img = tf.read_file(img_path)
            #decode_img = tf.image.decode_jpeg(read_img, channels = 3)
            #resize_img = tf.image.resize_images(decode_img, [64, 64])
            #resize_img = tf.reshape(resize_img, [1, 64, 64, 3])
            #img_data = sess.run(resize_img)
            predict_img_batch[img, :, :, :] = read_img
    
    return imgs_index, predict_img_batch

def model_predict(path, attribute, outfile, predict_file, batch_size):
    X = tf.placeholder(tf.float32, [None, 224, 224, 3], name = 'place_x')
    Y = tf.placeholder(tf.int64, [None], name = 'place_y')
    Attrites = tf.placeholder(tf.float32, [30, 230], name = 'Attrites')
    vgg_out = encoder(X)    
    W, logits = feature_attrites(vgg_out, Attrites)

    saver = tf.train.Saver()
    sess = tf.Session()
    saver.restore(sess, '/home/lisren/Zeros/vgg19_model/model-61500.ckpt')
   
    f = open(outfile, "w") 
    count_correct_nums = 0
    test_nums = 0
    imgs_name, resize_imgs = load_predict_data(path,indexfile)
    imgs_batch = mini_batches(resize_imgs, batch_size) 
    ec = 0
    for img_batch in imgs_batch:
        print("ec: ", ec)
        ec += 1
        pred = sess.run([logits], feed_dict = {X: resize_imgs[img_batch]}) 
        pred_label = classfy(pred)
        for img_i in range(len(img_batch)):
            f.write(imgs_name[img_batch[img_i]] + '\t' + pred_label[img_i] + '\n')

    f.close()

if __name__ == '__main__':
    path = 'test'
    batch_size = 128
    attribute = get_per_attributes("sort_types_attrites.csv")
    index_file = "type_index.txt"
    outfile = "822.txt"
    model_predict(path, attribute, outfile, "image.txt", batch_size)
