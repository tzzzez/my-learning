import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

#读取数据
data = pd.read_csv(r"C:\Users\LENOVO\Desktop\model_test2\test.csv",engine='python',encoding='gbk')

#删除空信息
data.dropna(axis=0,inplace=True)

#删除时间列
data.drop(['时间'],axis=1,inplace=True)

#分出数据特征和数据结果
X = data.iloc[:,0:-1]
y = data.iloc[:,-1]

#事先画出学习曲线调整n_estimators参数
rfr = RFR(n_estimators=100
          ,n_jobs=-1
          ,random_state=90
          )
rfr_s = cross_val_score(rfr,X,y,cv=10).mean()
print("未调整参数前模型的准确率:")
print(rfr_s)
print("------------------------------------------")


#进行网格搜索调整参数

#调整n_estimators
param_grid_ne = {'n_estimators':np.arange(50,100,1)}
rfr = RFR(n_jobs=-1
          ,random_state=90
          )
GS = GridSearchCV(rfr,param_grid_ne,cv=10)
GS.fit(X,y)
print("n_estimators调整后的最佳结果:")
print(GS.best_params_)
print(GS.best_score_)

#调整max_depth
param_grid_md = {'max_depth':np.arange(20,80,1)}
rfr = RFR(n_estimators=70
          ,n_jobs=-1
          ,random_state=90
          )
GS = GridSearchCV(rfr,param_grid_md,cv=10)
GS.fit(X,y)
print("max_depth调整后的最佳结果:")
print(GS.best_params_)
print(GS.best_score_)

#调整min_samples_leaf
param_grid_msl = {'min_samples_leaf':np.arange(1,50,1)}
rfr = RFR(n_estimators=70
          ,n_jobs=-1
          ,random_state=90
          )
GS = GridSearchCV(rfr,param_grid_msl,cv=10)
GS.fit(X,y)
print("min_samples_leaf调整后的最佳结果:")
print(GS.best_params_)
print(GS.best_score_)

#调整min_samples_split
param_grid_mss = {'min_samples_split':np.arange(2,50,1)}
rfr = RFR(n_estimators=70
          ,n_jobs=-1
          ,random_state=90
          )
GS = GridSearchCV(rfr,param_grid_mss,cv=10)
GS.fit(X,y)
print("min_samples_split调整后的最佳结果:")
print(GS.best_params_)
print(GS.best_score_)

