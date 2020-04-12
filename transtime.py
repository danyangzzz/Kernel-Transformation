encoding = 'utf-8'

import pandas as pd
import math
import numpy as np
from math import radians, sqrt,tan
import datetime


sigma = 0.5678020243184039
s = 0.5 / (sigma ** 2)


# 读取原文件中坐标
ini = pd.read_csv("F:\zhudanyang\LBS9.30\Flbsdataset\simplegeo data.csv", index_col=0)
l = ini.shape[0]

lon = np.zeros(l)
lat = np.zeros(l)
X = np.zeros(l)
Y = np.zeros(l)


# 转换平面坐标函数
def coord4(lon1,lat1):
    lon0, lat0 = 151.195,-33.8855
    # simplegeo151.195,-33.8855,151.18,-33.86
    # yelp-116.218,36.2114 -116,36
    lon0, lat0, lon1, lat1, = map(radians, [lon0, lat0, lon1, lat1])
    a = 6371
    c1 = pow(tan(lat0), 2.0)
    d1 = pow(1/tan(lat0), 2.0)
    x = a / sqrt(1 + 1 * c1)
    y = a / sqrt(1 + 1 * d1)
    c2 = pow(tan(lat1), 2.0)
    if tan(lat1) == 0:
        return 0
    else:
        d2 = pow(1/tan(lat1), 2.0)
    m = a / sqrt(1 + 1 * c2)
    n = a / sqrt(1 + 1 * d2)
    yl = sqrt(((m-x)**2+(n-y)**2))
    xl = x * (lon1 - lon0)
    return xl,yl


starttime = datetime.datetime.now()
# 读取经纬度转换为平面坐标
for i in range(0, l):
    lon[i] = (float(ini.ix[i]["lon"]))
    lat[i] = (float(ini.ix[i]["lat"]))
    X[i],Y[i] = coord4(lon[i],lat[i])
endtime = datetime.datetime.now()
print(endtime - starttime)


X0 = np.zeros(l)
Y0 = np.zeros(l)
Xmin = min(X)
Ymin = min(Y)
Xmax = max(X)
Ymax = max(Y)
min = min(Xmin,Ymin)
max = max(Xmax,Ymax)
Trange = max - min



# 归一化平面坐标
for i in range(0, l):
    X0[i] = (X[i] - min) / Trange
    Y0[i] = (Y[i] - min) / Trange
    ini.iloc[i,4] = X0[i]
    ini.iloc[i,5] = Y0[i]


k = 8

def trans(x0):
    x = np.zeros(k)
    x[0] = math.exp(-s * (x0 ** 2))
    for i in range(1,k):
           x[i] = x[0] * math.sqrt(((2 * s) ** i) / math.factorial(i)) * (x0 ** i)
    # x2 = x1 * math.sqrt(2 * s / math.factorial(1)) * x0
    # x3 = x1 * math.sqrt(((2 * s) ** 2) / math.factorial(2)) * (x0 ** 2)
    # x4 = x1 * math.sqrt(((2 * s) ** 3) / math.factorial(3)) * (x0 ** 3)
    return x


# 变换后的坐标
X1 = np.zeros([l,k])
Y1 = np.zeros([l,k])
starttime = datetime.datetime.now()
for h in range(0, l):
    X1[h] = trans(X0[h])
    Y1[h] = trans(Y0[h])
    for i in range(0,k):
        ini.iloc[h,6+i] = X1[h][i]
        ini.iloc[h,14+i] = Y1[h][i]

endtime = datetime.datetime.now()
print(endtime - starttime)

# 把变换的坐标存放到表里
# ini.to_csv("F:\zhudanyang\LBS-FOG\Flbsdataset\yelp data.csv")
ini.to_csv("F:\zhudanyang\LBS9.30\Flbsdataset\simplegeo data.csv")