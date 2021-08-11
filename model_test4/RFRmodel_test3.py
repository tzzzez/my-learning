import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

data = pd.read_csv(r'./data3.csv',engine='python',encoding='gbk')

data.dropna(axis=0,inplace=True)

data.drop(['时间'],axis=1,inplace=True)

X = data.iloc[:,0:-1]
y = data.iloc[:,-1]

param_grid_ne = {
    'n_estimators':np.arange(1,151,1)
}
rfr = RFR(n_jobs=-1
          ,random_state=70
          )
GS = GridSearchCV(rfr,param_grid_ne,cv=10)
GS.fit(X,y)
print("n_estimators调整后的最佳结果:")
print(GS.best_params_)
print(GS.best_score_)

#调整max_depth
param_grid_md = {'max_depth':np.arange(20,60,1)}
rfr = RFR(n_estimators=48
          ,n_jobs=-1
          ,random_state=70
          )
GS = GridSearchCV(rfr,param_grid_md,cv=10)
GS.fit(X,y)
print("max_depth调整后的最佳结果:")
print(GS.best_params_)
print(GS.best_score_)

#调整min_samples_leaf
param_grid_msl = {'min_samples_leaf':np.arange(1,50,1)}
rfr = RFR(n_estimators=48
          ,n_jobs=-1
          ,random_state=70
          )
GS = GridSearchCV(rfr,param_grid_msl,cv=10)
GS.fit(X,y)
print("min_samples_leaf调整后的最佳结果:")
print(GS.best_params_)
print(GS.best_score_)

#调整min_samples_split
param_grid_mss = {'min_samples_split':np.arange(2,50,1)}
rfr = RFR(n_estimators=48
          ,n_jobs=-1
          ,random_state=70
          )
GS = GridSearchCV(rfr,param_grid_mss,cv=10)
GS.fit(X,y)
print("min_samples_split调整后的最佳结果:")
print(GS.best_params_)
print(GS.best_score_)

print("-----------------------------------")
#将X，y归一化处理后的最佳调参
print("归一化后的最佳调参:")

data = data.values
scaler = MinMaxScaler(feature_range=[0,1])
data_s = scaler.fit_transform(data)

data_s = pd.DataFrame(data_s)
X = data_s.iloc[:,0:-1]
y = data_s.iloc[:,-1]

param_grid_ne = {
    'n_estimators':np.arange(1,151,1)
}
rfr = RFR(n_jobs=-1
          ,random_state=70
          )
GS = GridSearchCV(rfr,param_grid_ne,cv=10)
GS.fit(X,y)
print("n_estimators调整后的最佳结果:")
print(GS.best_params_)
print(GS.best_score_)

#调整max_depth
param_grid_md = {'max_depth':np.arange(20,60,1)}
rfr = RFR(n_estimators=48
          ,n_jobs=-1
          ,random_state=70
          )
GS = GridSearchCV(rfr,param_grid_md,cv=10)
GS.fit(X,y)
print("max_depth调整后的最佳结果:")
print(GS.best_params_)
print(GS.best_score_)

#调整min_samples_leaf
param_grid_msl = {'min_samples_leaf':np.arange(1,50,1)}
rfr = RFR(n_estimators=48
          ,n_jobs=-1
          ,random_state=70
          )
GS = GridSearchCV(rfr,param_grid_msl,cv=10)
GS.fit(X,y)
print("min_samples_leaf调整后的最佳结果:")
print(GS.best_params_)
print(GS.best_score_)

#调整min_samples_split
param_grid_mss = {'min_samples_split':np.arange(2,50,1)}
rfr = RFR(n_estimators=48
          ,n_jobs=-1
          ,random_state=70
          )
GS = GridSearchCV(rfr,param_grid_mss,cv=10)
GS.fit(X,y)
print("min_samples_split调整后的最佳结果:")
print(GS.best_params_)
print(GS.best_score_)

