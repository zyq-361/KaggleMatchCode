# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 18:16:05 2018

@author: zyq
"""
#load datasets
import numpy as np
import pandas as pd
    
marketX=pd.read_csv('F:/PyWork/match/trainx.csv',header=None)
marketY=pd.read_csv('F:/PyWork/match/trainy.csv',header=None)
TestX=pd.read_csv('F:/PyWork/match/testx.csv',header=None)

marketX=np.array(marketX)
marketY=np.array(marketY)
feature_names=marketX[0]
marketX=np.array(marketX[1:])
marketY=np.array(marketY[1:])
TestX=np.array(TestX[1:])
#print(feature_names)
#print(marketY)

#preprocessing the data
from sklearn import preprocessing
impute=preprocessing.Imputer(strategy='median')
marketX=impute.fit_transform(marketX)
TestX=impute.fit_transform(TestX)

#first, we will separate training and testing data
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test=train_test_split(marketX,marketY,test_size=0.25,random_state=33)
#feature selection



# Training a decision tree classifier
from sklearn import tree
clf=tree.DecisionTreeClassifier(max_depth=2,min_samples_leaf=6)
clf=clf.fit(X_train,y_train)

#print(clf.predict(X_train))

#last we will define a helper function to measure the performance of a classifier
from sklearn import metrics
def measure_performance(X,y,clf,show_accuracy=True):
    y_pred=clf.predict(X)
    if show_accuracy:
        print("Accuracy:{0:.3f}".format(metrics.accuracy_score(y,y_pred)),"\n")
measure_performance(X_train,y_train,clf,show_accuracy=True)
measure_performance(X_test,y_test,clf,show_accuracy=True)

y_pred=clf.predict(TestX)
print(y_pred)


data1 = pd.DataFrame(y_pred)
data1.to_csv('data1.csv')

























# =============================================================================
# import numpy as np
# import matplotlib.pyplot as plt
# 
# 
# def load_datasets(file_name):
#     data_mat =[]
#     with open(file_name) as fr:
#         lines=fr.readlines()
#     for line in lines:
#         cur_line = line.strip().split("\t")
#         flt_line =list(map(lambda x:float(x),cur_line))
#         data_mat.append(flt_line)
#     return np.array(data_mat)
# 
# 
# print(load_datasets('F:\\PyWork\\a.txt'))
# =============================================================================
#import numpy as np
#import matplotlib.pyplot as plt
#from sklearn import tree



































    
    
