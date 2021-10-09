import pandas as pd
import numpy as np

data = pd.read_csv(r"./data.csv",engine='python',encoding='gbk')

#降雨量
data_dsw = data.iloc[:,0:2]
data_thc = data.iloc[:,2:4]
data_lms = data.iloc[:,4:6]
data_ss = data.iloc[:,6:8]
data_lx = data.iloc[:,8:10]
data_ylg = data.iloc[:,10:12]
data_ch = data.iloc[:,12:14]

#昌化流量和水位
data_target = data.iloc[:,14:17]

#窄溪蒸发量
data_zx = data.iloc[:,17:19]

#初始土壤含水量
data_hsl = data.iloc[:,19:-1]

#以防万一删除空值
data_dsw.dropna(axis=0,inplace=True)
data_thc.dropna(axis=0,inplace=True)
data_lms.dropna(axis=0,inplace=True)
data_ss.dropna(axis=0,inplace=True)
data_ylg.dropna(axis=0,inplace=True)
data_ch.dropna(axis=0,inplace=True)
data_target.dropna(axis=0,inplace=True)
data_zx.dropna(axis=0,inplace=True)
data_hsl.dropna(axis=0,inplace=True)

data_target_time = pd.to_datetime(data_target['时间'],format='%Y/%m/%d %H:%M')
data_target_y = data_target.iloc[:,2]

#筛选各个基站的前4-7h的降雨量
