import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor as DTR
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

data = pd.read_csv(r"./test.csv",engine='python',encoding='gbk')

data.dropna(axis=0,inplace=True)

data.drop(['时间'],axis=1,inplace=True)

X = data.iloc[:,0:-1]
y = data.iloc[:,-1]

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

#网格搜索调整最佳参数
param_grid = {
    "criterion":['mse','mae'],
    "max_depth":range(1,20),
    "min_samples_leaf":range(2,20),
    "min_samples_split":range(2,20)
}

dtr = DTR(random_state=70)

GS = GridSearchCV(dtr,param_grid,cv=10)
GS.fit(X,y)
print("网格搜索后的最佳结果:")
print(GS.best_params_)
print(GS.best_score_)
'''

'''
网格搜索后的最佳结果:
{'criterion': 'mae', 'max_depth': 8, 'min_samples_leaf': 7, 'min_samples_split': 17}
0.9442709028401598
'''
dtr = DTR(
    criterion='mae'
    ,max_depth=8
    ,min_samples_leaf=7
    ,min_samples_split=17
)

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.5, random_state=70)

score_s = cross_val_score(dtr,X,y,cv=10).mean()
print("交叉验证模型得分:")
print(score_s)

print("--------------------------")

#得出均方误差
mse = cross_val_score(dtr,X_test,Y_test,scoring="neg_mean_squared_error",cv=10).mean()
print("均方误差:")
print(-mse)
print("---------------------------")

#得出平方误差
mae = cross_val_score(dtr,X_test,Y_test,scoring="neg_mean_absolute_error",cv=10).mean()
print("平方绝对误差:")
print(-mae)
print("----------------------------")

dtr.fit(X_train,Y_train)
#分割比例5:2:3
X_test, X_target, Y_test, Y_target = train_test_split(X_test,Y_test,test_size=0.6, random_state=70)
Y_predict = dtr.predict(X_target)
#画出验证集的图像
Y_target = Y_target.values
X_target = X_target.values

plt.plot(range(0,648),Y_target,color="red",label="real")
plt.plot(range(0,648),Y_predict,color="blue",label="predict")
plt.xticks(range(0,648))
plt.legend()
plt.show()

