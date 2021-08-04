from sklearn.neighbors import KNeighborsRegressor as KNR
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
<<<<<<< HEAD
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
=======
>>>>>>> c48976412e2354a526dc68e7b8668d1dba70d554

data = pd.read_csv(r"./test.csv",engine='python',encoding='gbk')

data.dropna(axis=0,inplace=True)

data.drop(['时间'],axis=1,inplace=True)

X = data.iloc[:,0:-1]
y = data.iloc[:,-1]

<<<<<<< HEAD
'''
=======
>>>>>>> c48976412e2354a526dc68e7b8668d1dba70d554
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
<<<<<<< HEAD
'''

'''
网格搜索后的最佳结果:
{'n_neighbors': 7, 'p': 1, 'weights': 'distance'}
0.9499653412950108
'''
knr = KNR(
    n_neighbors=7
    ,p=1
    ,weights='distance'
)

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.5, random_state=70)

score_s = cross_val_score(knr,X,y,cv=10).mean()
print("交叉验证模型得分:")
print(score_s)

print("--------------------------")

#得出均方误差
mse = cross_val_score(knr,X_test,Y_test,scoring="neg_mean_squared_error",cv=10).mean()
print("均方误差:")
print(-mse)
print("---------------------------")

#得出平方误差
mae = cross_val_score(knr,X_test,Y_test,scoring="neg_mean_absolute_error",cv=10).mean()
print("平方绝对误差:")
print(-mae)
print("----------------------------")

knr.fit(X_train,Y_train)
#分割比例5:2:3
X_test, X_target, Y_test, Y_target = train_test_split(X_test,Y_test,test_size=0.6, random_state=70)
Y_predict = knr.predict(X_target)
#画出验证集的图像
Y_target = Y_target.values
X_target = X_target.values

plt.plot(range(0,648),Y_target,color="red",label="real")
plt.plot(range(0,648),Y_predict,color="blue",label="predict")
plt.xticks(range(0,648))
plt.legend()
plt.show()
=======

>>>>>>> c48976412e2354a526dc68e7b8668d1dba70d554

