import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor as DTR
from sklearn.model_selection import GridSearchCV

data = pd.read_csv(r"./test.csv",engine='python',encoding='gbk')

data.dropna(axis=0,inplace=True)

data.drop(['时间'],axis=1,inplace=True)

X = data.iloc[:,0:-1]
y = data.iloc[:,-1]

param_grid = {
    "criterion":['mse','mae'],
    "max_depth":range(1,20),
    "min_samples_leaf":range(2,20),
    "min_samples_split":range(2,20)
}

dtr = DTR(random_state=70)

'''
dtr = DTR(criterion="mse"
          ,random_state=30
          ,splitter="random"
          ,max_depth=5
          ,min_samples_leaf=10
          ,min_samples_split=10
          )
score = cross_val_score(dtr,X,y,cv=10).mean()
print(score)
'''
GS = GridSearchCV(dtr,param_grid,cv=10)
GS.fit(X,y)
print("网格搜索后的最佳结果:")
print(GS.best_params_)
print(GS.best_score_)




