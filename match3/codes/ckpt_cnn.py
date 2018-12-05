# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd
import random
import time

IMAGE_WIDTH = 160
IMAGE_HEIGHT = 64
MAX_NUM = 3
CHAR_SET_LEN = 10
    
def text2vec(digit):
    text = str(digit)
    vector=np.zeros(CHAR_SET_LEN*MAX_NUM)
    if len(text)==1:
        vector[0]=1
        vector[10]=1
        vector[20+digit]=1
    elif len(text)==2:
        vector[0]=1
        vector[10+digit//10]=1
        vector[20+digit%10]=1
    else:
        vector[digit//100]=1
        vector[10+(digit//10%10)]=1
        vector[20+(digit%10)]=1
    return vector 


    
#生成一个训练batch    
def get_next_batch(batch_size, step, type='train'):
    batch_x = np.zeros([batch_size, IMAGE_WIDTH*IMAGE_HEIGHT])
    batch_y = np.zeros([batch_size, CHAR_SET_LEN*MAX_NUM])
    if type == 'train':
        index = [ i for i in train_text_array]
    elif type == 'valid':
        index = [ i for i in valid_text_array]

#    np.random.shuffle(index)
    totalNumber = len(index) 
    indexStart = step * batch_size   
    for i in range(batch_size):
        idx = index[(i + indexStart) % totalNumber]
        jpg_path = './train/' + str(idx) + '.jpg' 
        img = Image.open(jpg_path)
        img = img.resize((IMAGE_WIDTH, IMAGE_HEIGHT), Image.ANTIALIAS)
        img = img.convert('L')
        img = np.array(img)
        img = img.flatten() / 255

        text = text_val[idx]
        text = text2vec(text)
        batch_x[i,:] = img
        batch_y[i,:] = text 
    return batch_x, batch_y
    
#构建卷积神经网络并训练
def train_data_with_CNN():
    def weight_variable(shape, name='weight'):
        w_alpha=0.01
#        init = w_alpha*tf.truncated_normal(shape, stddev=0.1)
        init = w_alpha*tf.random_normal(shape)
        var = tf.Variable(initial_value=init, name=name)
        return var
    #初始化偏置    
    def bias_variable(shape, name='bias'):
        b_alpha=0.1
        init = b_alpha * tf.random_normal(shape)
        var = tf.Variable(init, name=name)
        return var
    #卷积    
    def conv2d(x, W, name='conv2d'):
        return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME', name=name)
    #池化 
    def max_pool_2X2(x, name='maxpool'):
        return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name=name)  

    X = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT * IMAGE_WIDTH], name='data-input')
    Y = tf.placeholder(tf.float32, [None, MAX_NUM * CHAR_SET_LEN], name='label-input')    
    x_input = tf.reshape(X, [-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1], name='x-input')
    #dropout,防止过拟合
    #请注意 keep_prob 的 name，在测试model时会用到它
    keep_prob = tf.placeholder(tf.float32, name='keep-prob')
    #第一层卷积
    W_conv1 = weight_variable([3,3,1,32], 'W_conv1')
    B_conv1 = bias_variable([32], 'B_conv1')
    conv1 = tf.nn.relu(conv2d(x_input, W_conv1, 'conv1') + B_conv1)
    conv1 = max_pool_2X2(conv1, 'conv1-pool')
    conv1 = tf.nn.dropout(conv1, keep_prob)
    #第二层卷积
    W_conv2 = weight_variable([3,3,32,64], 'W_conv2')
    B_conv2 = bias_variable([64], 'B_conv2')
    conv2 = tf.nn.relu(conv2d(conv1, W_conv2,'conv2') + B_conv2)
    conv2 = max_pool_2X2(conv2, 'conv2-pool')
    conv2 = tf.nn.dropout(conv2, keep_prob)
    #第三层卷积
    W_conv3 = weight_variable([3,3,64,64], 'W_conv3')
    B_conv3 = bias_variable([64], 'B_conv3')
    conv3 = tf.nn.relu(conv2d(conv2, W_conv3, 'conv3') + B_conv3)
    conv3 = max_pool_2X2(conv3, 'conv3-pool')
    conv3 = tf.nn.dropout(conv3, keep_prob)
    #全链接层
    #每次池化后，图片的宽度和高度均缩小为原来的一半，进过上面的三次池化，宽度和高度均缩小8倍
    W_fc1 = weight_variable([20*8*64, 1024], 'W_fc1')
    B_fc1 = bias_variable([1024], 'B_fc1')
    fc1 = tf.reshape(conv3, [-1, 20*8*64])
    fc1 = tf.nn.relu(tf.add(tf.matmul(fc1, W_fc1), B_fc1))
    fc1 = tf.nn.dropout(fc1, keep_prob)
    #输出层
    W_fc2 = weight_variable([1024, MAX_NUM * CHAR_SET_LEN], 'W_fc2')
    B_fc2 = bias_variable([MAX_NUM * CHAR_SET_LEN], 'B_fc2')
    output = tf.add(tf.matmul(fc1, W_fc2), B_fc2, 'output')


    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=Y, logits=output))
    optimizer = tf.train.AdamOptimizer(0.0005).minimize(loss)
    
    predict = tf.reshape(output, [-1, MAX_NUM, CHAR_SET_LEN], name='predict')
    labels = tf.reshape(Y, [-1, MAX_NUM, CHAR_SET_LEN], name='labels')
    #预测结果
    #请注意 predict_max_idx 的 name，在测试model时会用到它
    predict_max_idx = tf.argmax(predict, axis=2, name='predict_max_idx')
    labels_max_idx = tf.argmax(labels, axis=2, name='labels_max_idx')
    predict_correct_vec = tf.equal(predict_max_idx, labels_max_idx)
    accuracy = tf.reduce_mean(tf.cast(predict_correct_vec, tf.float32))
    
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        steps = 0
        ckpt = tf.train.get_checkpoint_state('./models/')
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess,ckpt.model_checkpoint_path)
        while True:
            train_data, train_label = get_next_batch(32, steps, 'train')
            _,loss_ = sess.run([optimizer,loss], feed_dict={X : train_data, Y : train_label, keep_prob:0.75})
            print("step:%d loss:%f" % (steps,loss_))
            if steps % 100 == 0:
                valid_data, valid_label = get_next_batch(100, steps, 'valid')
                acc = sess.run(accuracy, feed_dict={X : valid_data, Y : valid_label, keep_prob:1.0})
                print("steps=%d, accuracy=%f" % (steps, acc))
                saver.save(sess, "./models/cnn.model", global_step=steps)
                if acc > 0.99:
                    break
            steps += 1


            
if __name__ == '__main__':    
    
    random.seed(time.time())
    #打乱顺序
    label = pd.read_csv('./train_labels.csv')
    text_val = np.array(label['y'])
    text_id = np.array(label['id'])

    TRAIN_SIZE = 0.7
    TRAIN_NUM = int(len(text_id) * TRAIN_SIZE)
    random.shuffle(text_id)
    train_text_array = text_id[:TRAIN_NUM]
    valid_text_array = text_id[TRAIN_NUM:]

    train_data_with_CNN()    
 

