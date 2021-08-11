import pandas as pd
import numpy as np

data = pd.read_csv(r"./data.csv",engine='python',encoding='gbk')

data_dsw = data.iloc[:,0:2]
data_thc = data.iloc[:,2:4]
data_lms = data.iloc[:,4:6]
data_ss = data.iloc[:,6:8]
data_lx = data.iloc[:,8:10]
data_ylg = data.iloc[:,10:12]
data_ch = data.iloc[:,12:14]
data_target = data.iloc[:,14:16]
#窄溪的蒸发量
data_zx = data.iloc[:,17:19]
#土壤的含水量
data_tr = data.iloc[:,19:-1]

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
data_zx['窄溪时间'] = pd.to_datetime(data_zx['窄溪时间'],format='%Y/%m/%d %H:%M')

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
    if(hour <= 6):
        hour = hour + 24 - 7
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
        hour = hour - 7
    data_tmp = data_dsw[data_dsw['岛石坞时间'].dt.year == year]
    data_tmp = data_tmp[data_tmp['岛石坞时间'].dt.month == month]
    data_tmp = data_tmp[data_tmp['岛石坞时间'].dt.day == day]
    data_tmp = data_tmp[data_tmp['岛石坞时间'].dt.hour == hour]
    once = data_tmp['岛石坞雨量'].values
    col2.append(once)
'''

'''
for i in range(0,data_target.shape[0]):
    once_y = data_target_y[i]
    col1.append(once_y)
    year = data_target_time[i].year
    month = data_target_time[i].month
    day = data_target_time[i].day
    hour = data_target_time[i].hour
    if(hour <= 6):
        hour = hour + 24 - 7
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
        hour = hour - 7
    data_tmp = data_thc[data_thc['桃花村时间'].dt.year == year]
    data_tmp = data_tmp[data_tmp['桃花村时间'].dt.month == month]
    data_tmp = data_tmp[data_tmp['桃花村时间'].dt.day == day]
    data_tmp = data_tmp[data_tmp['桃花村时间'].dt.hour == hour]
    once = data_tmp['桃花村雨量'].values
    col2.append(once)
'''
'''
for i in range(0,data_target.shape[0]):
    once_y = data_target_y[i]
    col1.append(once_y)
    year = data_target_time[i].year
    month = data_target_time[i].month
    day = data_target_time[i].day
    hour = data_target_time[i].hour
    if(hour <= 6):
        hour = hour + 24 - 7
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
        hour = hour - 7
    data_tmp = data_lms[data_lms['龙门寺时间'].dt.year == year]
    data_tmp = data_tmp[data_tmp['龙门寺时间'].dt.month == month]
    data_tmp = data_tmp[data_tmp['龙门寺时间'].dt.day == day]
    data_tmp = data_tmp[data_tmp['龙门寺时间'].dt.hour == hour]
    once = data_tmp['龙门寺雨量'].values
    col2.append(once)
'''
'''
for i in range(0,data_target.shape[0]):
    once_y = data_target_y[i]
    col1.append(once_y)
    year = data_target_time[i].year
    month = data_target_time[i].month
    day = data_target_time[i].day
    hour = data_target_time[i].hour
    if(hour <= 6):
        hour = hour + 24 - 7
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
        hour = hour - 7
    data_tmp = data_ss[data_ss['双石时间'].dt.year == year]
    data_tmp = data_tmp[data_tmp['双石时间'].dt.month == month]
    data_tmp = data_tmp[data_tmp['双石时间'].dt.day == day]
    data_tmp = data_tmp[data_tmp['双石时间'].dt.hour == hour]
    once = data_tmp['双石雨量'].values
    col2.append(once)
'''
'''
for i in range(0,data_target.shape[0]):
    once_y = data_target_y[i]
    col1.append(once_y)
    year = data_target_time[i].year
    month = data_target_time[i].month
    day = data_target_time[i].day
    hour = data_target_time[i].hour
    if(hour <= 6):
        hour = hour + 24 - 7
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
        hour = hour - 7
    data_tmp = data_lx[data_lx['岭下时间'].dt.year == year]
    data_tmp = data_tmp[data_tmp['岭下时间'].dt.month == month]
    data_tmp = data_tmp[data_tmp['岭下时间'].dt.day == day]
    data_tmp = data_tmp[data_tmp['岭下时间'].dt.hour == hour]
    once = data_tmp['岭下雨量'].values
    col2.append(once)
'''
'''
for i in range(0,data_target.shape[0]):
    once_y = data_target_y[i]
    col1.append(once_y)
    year = data_target_time[i].year
    month = data_target_time[i].month
    day = data_target_time[i].day
    hour = data_target_time[i].hour
    if(hour <= 6):
        hour = hour + 24 - 7
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
        hour = hour - 7
    data_tmp = data_ch[data_ch['昌化时间'].dt.year == year]
    data_tmp = data_tmp[data_tmp['昌化时间'].dt.month == month]
    data_tmp = data_tmp[data_tmp['昌化时间'].dt.day == day]
    data_tmp = data_tmp[data_tmp['昌化时间'].dt.hour == hour]
    once = data_tmp['昌化雨量'].values
    col2.append(once)
'''
'''
#窄溪的蒸发量只需要年月日相对应即可
for i in range(0,data_target.shape[0]):
    once_y = data_target_y[i]
    col1.append(once_y)
    year = data_target_time[i].year
    month = data_target_time[i].month
    day = data_target_time[i].day
    data_tmp = data_zx[data_zx['窄溪时间'].dt.year == year]
    data_tmp = data_tmp[data_tmp['窄溪时间'].dt.month == month]
    data_tmp = data_tmp[data_tmp['窄溪时间'].dt.day == day]
    once = data_tmp['蒸发量'].values
    col2.append(once)

col1 = pd.DataFrame(col1)
col2 = pd.DataFrame(col2)
data_new = pd.concat([data_target_time, col1, col2], axis=1)
'''
'''
#筛选土壤含水量
#目标时间
data_target_time = data.iloc[:,0]

#土壤含水量
data_tr = data.iloc[:,1:]

data_tr.dropna(axis=0,inplace=True)

data_target_time = pd.to_datetime(data_target_time,format='%Y/%m/%d %H:%M')

line1_1 = data_tr[['上层1']].values
line1_2 = data_tr[['上层2']].values
line1_3 = data_tr[['上层3']].values
line1_4 = data_tr[['上层4']].values
line1_5 = data_tr[['上层5']].values
line1_6 = data_tr[['上层6']].values
line1_7 = data_tr[['上层7']].values
line2_1 = data_tr[['下层1']].values
line2_2 = data_tr[['下层2']].values
line2_3 = data_tr[['下层3']].values
line2_4 = data_tr[['下层4']].values
line2_5 = data_tr[['下层5']].values
line2_6 = data_tr[['下层6']].values
line2_7 = data_tr[['下层7']].values
line3_1 = data_tr[['深层1']].values
line3_2 = data_tr[['深层2']].values
line3_3 = data_tr[['深层3']].values
line3_4 = data_tr[['深层4']].values
line3_5 = data_tr[['深层5']].values
line3_6 = data_tr[['深层6']].values
line3_7 = data_tr[['深层7']].values

#上层土壤含水量平均值
col1 = []

#中层土壤含水量平均值
col2 = []

#深层土壤含水量平均值
col3 = []

data_time = data_tr[['土壤时间']].values


for i in range(0,len(data_target_time)):
    target_year = data_target_time[i].year
    target_month = data_target_time[i].month
    target_day = data_target_time[i].day
    flag = 0
    for i in range(0,len(data_tr)):
        time = (int)(data_time[i])
        year = (int)(time / 10000)
        time = (int)(time % 10000)
        month = (int)(time / 100)
        day = (int)(time % 100)
        if(year==target_year and month==target_month and target_day==day):
            #上层土壤含水量平均值
            once1 = (line1_1[i] + line1_2[i] + line1_3[i] + line1_4[i] + line1_5[i] + line1_6[i] + line1_7[i]) * 1.0 / 7
            once2 = (line2_1[i] + line2_2[i] + line2_3[i] + line2_4[i] + line2_5[i] + line2_6[i] + line2_7[i]) * 1.0 / 7
            once3 = (line3_1[i] + line3_2[i] + line3_3[i] + line3_4[i] + line3_5[i] + line3_6[i] + line3_7[i]) * 1.0 / 7
            col1.append(once1)
            col2.append(once2)
            col3.append(once3)
            flag = 1
            break
    if(flag == 0):
        col1.append(None)
        col2.append(None)
        col3.append(None)


col1 = pd.DataFrame(col1)
col2 = pd.DataFrame(col2)
col3 = pd.DataFrame(col3)
data_new = pd.concat([data_target_time,col1,col2,col3],axis=1)
data_new.to_csv(r"D:\my-learning\model_test4\土壤含水量汇总.csv")
print('创建完成')
'''
'''
data_new.to_csv(r'./out2.csv')
print('创建完成')
'''