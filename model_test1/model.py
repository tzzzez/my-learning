import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

#导入原始数据
data = pd.read_csv(r"D:\my-learning\预报洪水模型测试一\日均流量洪水数据集.csv",engine='python')

"""
对于原始数据缺失的四个特征，先采用均值填补，查看模型得分；再进行中值填补，查看模型得分
"""

#将缺失的四列取出来
cl1_nan = data.iloc[:,11].values.reshape(-1,1)
cl2_nan = data.iloc[:,12].values.reshape(-1,1)
cl3_nan = data.iloc[:,13].values.reshape(-1,1)
cl4_nan = data.iloc[:,14].values.reshape(-1,1)

#使用均值填补
cl1_mean = SimpleImputer(strategy="mean",fill_value='nan').fit_transform(cl1_nan)
cl2_mean = SimpleImputer(strategy="mean",fill_value='nan').fit_transform(cl2_nan)
cl3_mean = SimpleImputer(strategy="mean",fill_value='nan').fit_transform(cl3_nan)
cl4_mean = SimpleImputer(strategy="mean",fill_value='nan').fit_transform(cl4_nan)

#用中位数填补
cl1_median = SimpleImputer(strategy="median",fill_value='nan').fit_transform(cl1_nan)
cl2_median = SimpleImputer(strategy="median",fill_value='nan').fit_transform(cl2_nan)
cl3_median = SimpleImputer(strategy="median",fill_value='nan').fit_transform(cl3_nan)
cl4_median = SimpleImputer(strategy="median",fill_value='nan').fit_transform(cl4_nan)

#test1使用均值填补，test2使用中位数填补
data_test1 = data.copy()
data_test2 = data.copy()

data_test1.iloc[:,11] = cl1_mean
data_test1.iloc[:,12] = cl2_mean
data_test1.iloc[:,13] = cl3_mean
data_test1.iloc[:,14] = cl4_mean

data_test2.iloc[:,11] = cl1_median
data_test2.iloc[:,12] = cl2_median
data_test2.iloc[:,13] = cl3_median
data_test2.iloc[:,14] = cl4_median

#删除部分缺失值以及时间特征
data_test1 = data_test1.dropna(axis=0)
data_test1.drop(["时间"],inplace=True,axis=1)

data_test2 = data_test2.dropna(axis=0)
data_test2.drop(["时间"],inplace=True,axis=1)

X_test1 = data_test1.iloc[:,0:-1]
y_test1 = data_test1.iloc[:,-1]

X_test2 = data_test2.iloc[:,0:-1]
y_test2 = data_test2.iloc[:,-1]

rfr = RFR(n_estimators=183,random_state=90)
rfr_s1 = cross_val_score(rfr,X_test1,y_test1,cv=10).mean()
print("利用均值填补的数据建造的模型的准确率:")
print(rfr_s1)

rfr = RFR(n_estimators=223,random_state=90)
rfr_s2 = cross_val_score(rfr,X_test2,y_test2,cv=10).mean()
print("利用中位数填补的数据建造的模型的准确率:")
print(rfr_s2)

print("------------------------------------------------------")
#先调整均值填补的数据模型
#画出n_estimators的学习曲线
print("调整均值模型的参数网格搜索结果")
score = []
for i in range(100,300,2):
    rfr = RFR(n_estimators=i
              ,random_state=90
              ,n_jobs=-1
              )
    once = cross_val_score(rfr, X_test1, y_test1, cv=5).mean()
    score.append(once)
#plt.plot(range(150,350,5),score)
#plt.show()
print("n_estimators调整后的最佳值:")
print(max(score),[*range(100,300)][score.index(max(score))])
#使用网格搜索调整参数
#调整max_depth
param_grid_md = {'max_depth':np.arange(30,50,1)}
rfr = RFR(n_estimators=183
          ,n_jobs=-1
          ,random_state=90
          )
GS_1 = GridSearchCV(rfr,param_grid_md,cv=10)
GS_1.fit(X_test1,y_test1)
print("max_depth调整后的最佳值:")
print(GS_1.best_params_)
print(GS_1.best_score_)

#调整max_features
param_grid_mf = {'max_features':np.arange(4,15,1)}
rfr = RFR(n_estimators=183
          ,n_jobs=-1
          ,random_state=90
          )
GS_2 = GridSearchCV(rfr,param_grid_mf,cv=10)
GS_2.fit(X_test1,y_test1)
print("max_features调整后的最佳值:")
print(GS_2.best_params_)
print(GS_2.best_score_)

#调整min_samples_leaf
param_grid_msl = {'min_samples_leaf':np.arange(1,20,1)}
rfr = RFR(n_estimators=183
          ,n_jobs=-1
          ,random_state=90
          )
GS_3 = GridSearchCV(rfr,param_grid_msl,cv=10)
GS_3.fit(X_test1,y_test1)
print("min_samples_leaf调整后的最佳值:")
print(GS_3.best_params_)
print(GS_3.best_score_)

#调整min_samples_split
param_grid_mss = {'min_samples_leaf':np.arange(1,20,1)}
rfr = RFR(n_estimators=183
          ,n_jobs=-1
          ,random_state=90
          )
GS_4 = GridSearchCV(rfr,param_grid_mss,cv=10)
GS_4.fit(X_test1,y_test1)
print("min_samples_split调整后的最佳值:")
print(GS_4.best_params_)
print(GS_4.best_score_)

print("-----------------------------------------------")

#中位数模型
print("中位数模型调整结果")
score_l = []
for i in range(100,300,2):
    rfr = RFR(n_estimators=i
              ,random_state=90
              ,n_jobs=-1
              )
    once = cross_val_score(rfr, X_test1, y_test1, cv=5).mean()
    score_l.append(once)
#plt.plot(range(150,350,5),score)
#plt.show()
print("n_estimators调整后的最佳值:")
print(max(score_l),[*range(100,300)][score_l.index(max(score_l))])

#使用网格搜索调整参数
#调整max_depth
param_grid_md_2 = {'max_depth':np.arange(30,50,1)}
rfr = RFR(n_estimators=223
          ,n_jobs=-1
          ,random_state=90
          )
GS_1_1 = GridSearchCV(rfr,param_grid_md_2,cv=10)
GS_1_1.fit(X_test2,y_test2)
print("max_depth调整后的最佳值:")
print(GS_1_1.best_params_)
print(GS_1_1.best_score_)

#调整max_features
param_grid_mf_2 = {'max_features':np.arange(4,15,1)}
rfr = RFR(n_estimators=223
          ,n_jobs=-1
          ,random_state=90
          )
GS_2_2 = GridSearchCV(rfr,param_grid_mf_2,cv=10)
GS_2_2.fit(X_test2,y_test2)
print("max_features调整后的最佳值:")
print(GS_2_2.best_params_)
print(GS_2_2.best_score_)

#调整min_samples_leaf
param_grid_msl_2 = {'min_samples_leaf':np.arange(1,20,1)}
rfr = RFR(n_estimators=223
          ,n_jobs=-1
          ,random_state=90
          )
GS_3_3 = GridSearchCV(rfr,param_grid_msl_2,cv=10)
GS_3_3.fit(X_test2,y_test2)
print("min_samples_leaf调整后的最佳值:")
print(GS_3_3.best_params_)
print(GS_3_3.best_score_)

#调整min_samples_split
param_grid_mss_2 = {'min_samples_leaf':np.arange(1,20,1)}
rfr = RFR(n_estimators=223
          ,n_jobs=-1
          ,random_state=90
          )
GS_4_4 = GridSearchCV(rfr,param_grid_mss_2,cv=10)
GS_4_4.fit(X_test2,y_test2)
print("min_samples_split调整后的最佳值:")
print(GS_4_4.best_params_)
print(GS_4_4.best_score_)