# coding: utf-8


#导入必要的工具包和数据集
import pandas as pd
import numpy as np
train_data=pd.read_csv('F:/PyWork/match2/trainx.csv',header=None)
test_data=pd.read_csv('F:/PyWork/match2/testx.csv')



#使训练集测试集格式一致：给训练集加上特征名，分离出标签列
train_y=train_data[14]
train_data=train_data.drop(columns=[14])
feature_names=test_data.columns
train_data.columns=feature_names




#改变capital-gain/loss的范围，进行对数转换
skewed = ['capital-gain', 'capital-loss']
train_set = pd.DataFrame(data = train_data)
test_set = pd.DataFrame(data = test_data)
train_set[skewed] = train_data[skewed].apply(lambda x: np.log(x + 1))
test_set[skewed] = test_data[skewed].apply(lambda x: np.log(x + 1))




#正则化数值特征，将其规范到0-1区间
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler() # default=(0, 1)
numerical = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']

train_set[numerical] = scaler.fit_transform(train_set[numerical])

test_set[numerical] = scaler.fit_transform(test_set[numerical])




#删除不重要的特征
train=train_set.drop(columns=['fnlwgt','education','native-country'])
test=test_set.drop(columns=['fnlwgt','education','native-country'])


#填补缺失值
train['workclass']=train['workclass'].replace(' ?',' Private')
train['occupation']=train['occupation'].replace(' ?',' Prof-specialty')
# train['native-country']=train['native-country'].replace(' ?',' United-States')

test['workclass']=test['workclass'].replace(' ?',' Private')
test['occupation']=test['occupation'].replace(' ?',' Prof-specialty')
# test['native-country']=test['native-country'].replace(' ?',' United-States')


#对标签列进行编码
income_set = set(train_y)
#print(income_set)
train_y= train_y.map({' <=50K': 0, ' >50K': 1}).astype(int)
# train_y


#编码类别特征
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
for i in ['workclass','marital-status','occupation','relationship','race','sex']:
    train[i]=le.fit_transform(train[i])

for i in ['workclass','marital-status','occupation','relationship','race','sex']:
    test[i]=le.fit_transform(test[i])



#进行独热编码
train = pd.get_dummies(train)
test=pd.get_dummies(test)



#数据集划分
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(train,train_y,test_size=0.6,random_state=33)



#进行模型的学习，并用格点搜索选择最优参数组合
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings(module='sklearn*', action='ignore', category=DeprecationWarning)



#使用格点搜索调参
#cv_params={'n_estimators':[375,400,425]}
#other_params = {
#    'n_estimators': 400,
#    'booster': 'gbtree',
#    'objective': 'binary:logistic',
#    'max_depth': 5,
#    'subsample': 1.0,
#    'colsample_bytree': 0.3,
#    'min_child_weight': 1,
#    'learning_rate': 0.1,
#    'gamma':0.3
#} 
#model=xgb.XGBClassifier(**other_params)
#optimized_GBM=GridSearchCV(estimator=model,param_grid=cv_params,scoring='f1',cv=5,verbose=1)
#optimized_GBM.fit(X_train,y_train)
#evalute_result=optimized_GBM.grid_scores_
#print('每轮迭代运行结果:{0}'.format(evalute_result))
#print('参数的最佳取值：{0}'.format(optimized_GBM.best_params_))
#print('最佳模型得分:{0}'.format(optimized_GBM.best_score_))



#使用最优参数组合进行模型的学习以及评价
model=xgb.XGBClassifier(learning_rate=0.1, n_estimators=400, max_depth=5, min_child_weight=1, 
                        subsample=1.0, colsample_bytree=0.3, gamma=0.3)
model.fit(X_train, y_train)

ans = model.predict(X_test)
from sklearn.metrics import f1_score
ans = model.predict(X_test)
print(f1_score(y_test, ans))



#使用模型预测并导出到csv
pred = model.predict(test)
print(pred)
data1 = pd.DataFrame(pred)
data1.to_csv('data11.csv')

