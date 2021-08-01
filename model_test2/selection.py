import pandas as pd
import numpy as np

data = pd.read_csv(r"C:\Users\LENOVO\Desktop\model_test2\test.csv",engine='python',encoding='gbk')

data_dsw = data.iloc[:,0:2]
data_thc = data.iloc[:,2:4]
data_lms = data.iloc[:,4:6]
data_ss = data.iloc[:,6:8]
data_lx = data.iloc[:,8:10]
data_ylg = data.iloc[:,10:12]
data_ch = data.iloc[:,12:14]
data_target = data.iloc[:,14:16]

#删除空值
data_dsw.dropna(axis=0,inplace=True)
data_thc.dropna(axis=0,inplace=True)
data_lms.dropna(axis=0,inplace=True)
data_ss.dropna(axis=0,inplace=True)
data_lx.dropna(axis=0,inplace=True)
data_ylg.dropna(axis=0,inplace=True)
data_ch.dropna(axis=0,inplace=True)
data_target.dropna(axis=0,inplace=True)

#转化目标值的时间
data_target_time = pd.to_datetime(data_target['时间'],format='%Y/%m/%d %H:%M')
data_target_y = data_target.iloc[:,1]

#转换时间
data_dsw['岛石坞时间'] = pd.to_datetime(data_dsw['岛石坞时间'],format='%Y/%m/%d %H:%M')
data_thc['桃花村时间'] = pd.to_datetime(data_thc['桃花村时间'],format='%Y/%m/%d %H:%M')
data_lms['龙门寺时间'] = pd.to_datetime(data_lms['龙门寺时间'],format='%Y/%m/%d %H:%M')
data_ss['双石时间'] = pd.to_datetime(data_ss['双石时间'],format='%Y/%m/%d %H:%M')
data_lx['岭下时间'] = pd.to_datetime(data_lx['岭下时间'],format='%Y/%m/%d %H:%M')
data_ylg['昱岭关时间'] = pd.to_datetime(data_ylg['昱岭关时间'],format='%Y/%m/%d %H:%M')
data_ch['昌化时间'] = pd.to_datetime(data_ch['昌化时间'],format='%Y/%m/%d %H:%M')

days = [0,31,28,31,30,31,30,31,31,30,31,30,31]

#存放流量
col1 = []
#存放雨量
col2 = []
'''
for i in range(0,data_target.shape[0]):
    once_y = data_target_y[i]
    col1.append(once_y)
    year = data_target_time[i].year
    month = data_target_time[i].month
    day = data_target_time[i].day
    hour = data_target_time[i].hour
    if(hour <= 3):
        hour = hour + 24 - 4
        if(day == 1):
            if(month == 1):
                month = 12
                year = year - 1
            else:
                month = month - 1
            day = days[month]
        else:
            day = day - 1
    else:
        hour = hour - 4
    data_tmp = data_dsw[data_dsw['岛石坞时间'].dt.year == year]
    data_tmp = data_tmp[data_tmp['岛石坞时间'].dt.month == month]
    data_tmp = data_tmp[data_tmp['岛石坞时间'].dt.day == day]
    data_tmp = data_tmp[data_tmp['岛石坞时间'].dt.hour == hour]
    once = data_tmp['岛石坞雨量'].values
    col2.append(once)

col1 = pd.DataFrame(col1)
col2 = pd.DataFrame(col2)
data_new = pd.concat([data_target_time, col1, col2], axis=1)
'''

for i in range(0,data_target.shape[0]):
    once_y = data_target_y[i]
    col1.append(once_y)
    year = data_target_time[i].year
    month = data_target_time[i].month
    day = data_target_time[i].day
    hour = data_target_time[i].hour
    if(hour <= 3):
        hour = hour + 24 - 4
        if(day == 1):
            if(month == 1):
                month = 12
                year = year - 1
            else:
                month = month - 1
            day = days[month]
        else:
            day = day - 1
    else:
        hour = hour - 4
    data_tmp = data_ch[data_ch['昌化时间'].dt.year == year]
    data_tmp = data_tmp[data_tmp['昌化时间'].dt.month == month]
    data_tmp = data_tmp[data_tmp['昌化时间'].dt.day == day]
    data_tmp = data_tmp[data_tmp['昌化时间'].dt.hour == hour]
    once = data_tmp['昌化雨量'].values
    col2.append(once)

col1 = pd.DataFrame(col1)
col2 = pd.DataFrame(col2)
data_new = pd.concat([data_target_time, col1, col2], axis=1)

data_new.to_csv(r'C:\Users\LENOVO\Desktop\model_test2\out2.csv')
print('创建完成')