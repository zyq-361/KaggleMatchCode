# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 11:03:52 2018

@author: zyq
"""

import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd
#import matplotlib.pyplot as plt 
 
CAPTCHA_LEN = 3
IMAGE_WIDTH = 96
IMAGE_HEIGHT = 64
MODEL_SAVE_PATH = './model/'
TEST_IMAGE_PATH = './test/'
 

 
def model_test():
    #加载graph
    saver = tf.train.import_meta_graph(MODEL_SAVE_PATH+"crack_captcha.model-5900.meta")
    graph = tf.get_default_graph()
    #从graph取得 tensor，他们的name是在构建graph时定义的(查看上面第2步里的代码)
    input_holder = graph.get_tensor_by_name("data-input_1:0")
    keep_prob_holder = graph.get_tensor_by_name("keep-prob_1:0")
    predict_max_idx = graph.get_tensor_by_name("predict_max_idx:0")
    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint(MODEL_SAVE_PATH))
#        count = 0
        digit_list = []
        for i in range(20000):
            img_path = TEST_IMAGE_PATH + str(i) + '.jpg'
            img = Image.open(img_path)
#            img = img.resize((IMAGE_WIDTH, IMAGE_HEIGHT), Image.BILINEAR)
            img = img.convert("L")       
            img_array = np.array(img)    
            img_data = img_array.flatten()/255
            
            predict = sess.run(predict_max_idx, feed_dict={input_holder:[img_data], keep_prob_holder : 1.0})            
#            filePathName = img_path
#            print(filePathName)
#            img = Image.open(filePathName)
#            plt.imshow(img)
#            plt.axis('off')
#            plt.show()
            predictValue = np.squeeze(predict)
            digit_list.append(predictValue)
#            print("预测值：{}".format(predictValue))
        return digit_list
        #     if np.array_equal(predictValue, rightValue):
        #         result = '正确'
        #         count += 1
        #     else: 
        #         result = '错误'            
        #     print('实际值：{}， 预测值：{}，测试结果：{}'.format(rightValue, predictValue, result))
        #     print('\n')
            
        # print('正确率：%.2f%%(%d/%d)' % (count*100/totalNumber, count, totalNumber))

def list2digit(y_list):
    arr = []
    for j in range(len(y_list)):
        text = y_list[j]
        digit = 0
        for i,c in enumerate(text):
            digit += pow(10,2-i)*c
        arr.append(digit)
    return arr  
if __name__ == '__main__':
    arr = model_test()
    y_list = list2digit(arr)
    data = pd.DataFrame(y_list)
    data.to_csv('test_labels.csv')
    