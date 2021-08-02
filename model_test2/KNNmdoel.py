from sklearn.neighbors import KNeighborsRegressor as KNR
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

data = pd.read_csv(r"./test.csv",engine='python',encoding='gbk')

data.dropna(axis=0,inplace=True)

data.drop(['时间'],axis=1,inplace=True)

X = data.iloc[:,0:-1]
y = data.iloc[:,-1]

#未调参前
knr = KNR()
score = cross_val_score(knr,X,y,cv=10).mean()
print(score)

#网格搜索调整参数
param_grid = [
    {
        'weights':['uniform'],
        'n_neighbors':[k for k in range(1,20)]
    },
    {
        'weights':['distance'],
        'n_neighbors':[k for k in range(1,20)],
        'p':[p for p in range(1,20)]
    }
]

knr = KNR()
GS = GridSearchCV(knr,param_grid,cv=10)
GS.fit(X,y)
print("网格搜索后的最佳结果:")
print(GS.best_params_)
print(GS.best_score_)

