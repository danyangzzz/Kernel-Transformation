encoding = 'utf-8'

from math import radians, cos, tan, asin, sqrt,log
import pandas as pd
import random
import numpy as np
import datetime
import math


# 用户和兴趣点
user = pd.read_csv("F:\zhudanyang\LBS9.30\Flbsdataset\simplegeo data.csv", index_col=0)
lu = user.shape[0]
poi1data = pd.read_csv("F:\zhudanyang\LBS9.30\Flbsdataset\simplegeo data.csv", index_col=0)
# poi1data = pd.read_csv("F:\zhudanyang\LBS-FOG\FLBSdata\SimpleRestaurant.csv", index_col=0)
lenp = poi1data.shape[0]


# result = pd.read_csv("F:\zhudanyang\LBS-FOG\FLBSdata\\result\dimension-rq.csv", index_col=0)
# 定义区域内兴趣点坐标

PX1 = np.zeros([lenp, 8])
PY1 = np.zeros([lenp, 8])
for b in range(0, lenp):
    PX1[b] = np.array([poi1data.ix[b]["x1"], poi1data.ix[b]["x2"], poi1data.ix[b]["x3"], poi1data.ix[b]["x4"],poi1data.ix[b]["x5"], poi1data.ix[b]["x6"], poi1data.ix[b]["x7"], poi1data.ix[b]["x8"]],
                      dtype=np.float)
    PY1[b] = np.array([poi1data.ix[b]["y1"], poi1data.ix[b]["y2"], poi1data.ix[b]["y3"], poi1data.ix[b]["y4"],poi1data.ix[b]["y5"], poi1data.ix[b]["y6"], poi1data.ix[b]["y7"], poi1data.ix[b]["y8"]],
                      dtype=np.float)


s = 1.5508726289713732
k = 8


def trans(lx0):
    lt = np.zeros(k)
    lt[0] = math.exp(-s * (lx0 ** 2))
    for i in range(1,k):
        lt[i] = lt[0] * math.sqrt(((2 * s) ** i) / math.factorial(i)) * (lx0 ** i)
    return lt


# 转换为平面坐标simplegeo
def coord4(lon1,lat1):
    lon0, lat0 = 151.195, -33.8855
    lon0, lat0, lon1, lat1 = map(radians, [lon0, lat0, lon1, lat1])
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
    return xl, yl


def adjust_r(x, y, x1, y1, lr):
    MIN, Trange = -3.7546724212, 9.1918757882
    # 经纬度换成平面坐标
    x, y = coord4(x, y)
    sx = np.zeros(4)
    sy = np.zeros(4)
    sx1 = np.zeros([4, 8])
    sy1 = np.zeros([4, 8])
    sr = 0
    lr = lr/1000
    for q in range(0, 4):
        j = 2 * random.random() * math.pi
        sx[q] = x + lr * math.cos(j)
        sy[q] = y + lr * math.sin(j)
        sx[q] = (sx[q] - MIN) / Trange
        sy[q] = (sy[q] - MIN) / Trange
        sx1[q] = trans(sx[q])
        sy1[q] = trans(sy[q])
        sr = sr + sqrt((-1/s) * (log(np.dot(x1.T, sx1[q])) + log(np.dot(y1.T, sy1[q]))))
    return sr/4



# 1000个用户的测量结果取平均
UX1 = np.zeros([100, 8])
UY1 = np.zeros([100, 8])
UX = np.zeros(100)
UY = np.zeros(100)
starttime = datetime.datetime.now()
for i in range(0, 100):
    h = random.randint(1, lu)
    UX[i] = float(user.ix[h]["lon"])
    UY[i] = float(user.ix[h]["lat"])
    UX1[i] = np.array([user.ix[h]["x1"], user.ix[h]["x2"], user.ix[h]["x3"], user.ix[h]["x4"],user.ix[h]["x5"], user.ix[h]["x6"], user.ix[h]["x7"], user.ix[h]["x8"]], dtype=np.float)
    UY1[i] = np.array([user.ix[h]["y1"], user.ix[h]["y2"], user.ix[h]["y3"], user.ix[h]["y4"],user.ix[h]["y5"], user.ix[h]["y6"], user.ix[h]["y7"], user.ix[h]["y8"]], dtype=np.float)
    dis1 = []
    srr = adjust_r(UX[i], UY[i], UX1[i], UY1[i], 400)
    PX = []
    for v in range(0, lenp):
        dis1.append(sqrt((-1/s)*((log(np.dot(UX1[i].T,PX1[v])) + log(np.dot(UY1[i].T,PY1[v]))))))
        if dis1[v] <= srr:
            PX.append([PX1[v],PY1[v]])


endtime = datetime.datetime.now()
print(endtime - starttime)






