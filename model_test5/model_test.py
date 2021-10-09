import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sko.PSO import PSO

#获取数据
data = pd.read_csv(r'./data1.csv',engine='python',encoding='gbk')

data.dropna(axis=0,inplace=True)

data.drop(['时间'],axis=1,inplace=True)

X = data.iloc[:,0:-1]
y = data.iloc[:,-1]

#创建模型，并返回预测
def demo_func(x):
    rfr = RFR(n_estimators=x[0].astype(np.int)
              ,n_jobs=-1
              ,random_state=70
              ,max_depth=x[1].astype(np.int)
              ,min_samples_leaf=x[2].astype(np.int)
              ,min_samples_split=x[3].astype(np.int)
              )
    score = cross_val_score(rfr,X,y,cv=10,scoring='r2').mean()
    return score

if __name__ == '__main__':
    pso = PSO(func=demo_func,n_dim=4,pop=50,max_iter=200,lb=[1,20,1,2],ub=[150,60,50,50],w=0.8,c1=2.0,c2=2.0)
    pso.run()
    print('best_x is',pso.gbest_x,'best_y is',pso.gbest_y)