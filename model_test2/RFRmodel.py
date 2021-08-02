import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error #均方误差
from sklearn.metrics import mean_absolute_error #平方绝对误差

#读取数据
data = pd.read_csv(r"D:\my-learning\model_test2\test.csv",engine='python',encoding='gbk')

#删除空信息
data.dropna(axis=0,inplace=True)

#删除时间列
data.drop(['时间'],axis=1,inplace=True)

#分出数据特征和数据结果
X = data.iloc[:,0:-1]
y = data.iloc[:,-1]

#训练集，验证数据集和测试数据集
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.5, random_state=70)



rfr = RFR(n_estimators=71
          ,max_depth=22
          ,min_samples_leaf=4
          ,n_jobs=-1
          )

rfr.fit(X_train,Y_train)
score = cross_val_score(rfr,X,y,cv=10).mean()

print("验证机交叉验证得分:")
print(score)

print("--------------------------")

#得出均方误差
mse = cross_val_score(rfr,X_test,Y_test,scoring="neg_mean_squared_error",cv=10).mean()
print("均方误差:")
print(-mse)
print("---------------------------")
'''
mse = mean_squared_error(Y_target,Y_predict)
print("均方误差:")
print(mse)
print("---------------------------")
'''

#得出平方误差
mae = cross_val_score(rfr,X_test,Y_test,scoring="neg_mean_absolute_error",cv=10).mean()
print("平方绝对误差:")
print(-mae)
print("----------------------------")

'''
mae = mean_absolute_error(Y_target,Y_predict)
print("平方绝对误差:")
print(mae)
print("----------------------------")
'''

#分割比例5:2:3
X_test, X_target, Y_test, Y_target = train_test_split(X_test,Y_test,test_size=0.6, random_state=70)
Y_predict = rfr.predict(X_target)
#画出验证集的图像
Y_target = Y_target.values
X_target = X_target.values

plt.plot(range(0,648),Y_target,color="red",label="real")
plt.plot(range(0,648),Y_predict,color="blue",label="predict")
plt.xticks(range(0,648))
plt.legend()
plt.show()



