encoding = 'utf-8'

from math import radians, cos, tan, asin, sqrt,log
import pandas as pd
import random
import numpy as np
from geopy.distance import vincenty
import math

UX = np.zeros(1000)
UY = np.zeros(1000)
UX1 = np.zeros([1000, 6])
UY1 = np.zeros([1000, 6])

#input the data
user = pd.read_csv("F:\zhudanyang\LBS1222\Flbsdataset2\\professionals.csv", index_col=0)
lu = user.shape[0]
poi1data = pd.read_csv("F:\zhudanyang\LBS1222\Flbsdataset2\SimpleRestaurant.csv", index_col=0)
#poi1data = pd.read_csv("F:\zhudanyang\LBS-FOG\FLBSdata\Health Services.csv",SimpleRestaurant.csv index_col=0)
lenp = poi1data.shape[0]


PX = np.zeros(lenp)
PY = np.zeros(lenp)
PX1 = np.zeros([lenp, 6])
PY1 = np.zeros([lenp, 6])
for b in range(0, lenp):
    t = b+1
    PX[b] = poi1data.ix[t]["lon"]
    PY[b] = poi1data.ix[t]["lat"]
    PX1[b] = np.array(
        [poi1data.ix[t]["x1"], poi1data.ix[t]["x2"], poi1data.ix[t]["x3"], poi1data.ix[t]["x4"], poi1data.ix[t]["x5"],
         poi1data.ix[t]["x6"]],
        dtype=np.float)
    PY1[b] = np.array(
        [poi1data.ix[t]["y1"], poi1data.ix[t]["y2"], poi1data.ix[t]["y3"], poi1data.ix[t]["y4"], poi1data.ix[t]["y5"],
         poi1data.ix[t]["y6"]],
        dtype=np.float)


s = 1.5508726289713732
k = 6
#store the result of our method
result = pd.read_csv("F:\zhudanyang\LBS1222\Flbsdataset2\\result\dimension-rq-simplegeo-6.csv.csv", index_col=0)


def trans(lx0):
    lt = np.zeros(k)
    lt[0] = math.exp(-s * (lx0 ** 2))
    for i in range(1,k):
        lt[i] = lt[0] * math.sqrt(((2 * s) ** i) / math.factorial(i)) * (lx0 ** i)
    return lt


def coord4(lon1,lat1):
    lon0, lat0 = 151.195,-33.8855
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

#transform the query distance r in the geographic space
def adjust_r(x, y, x1, y1, lr):
    MIN, Trange = -3.7544408256527677,18.382617960587897
    x, y = coord4(x, y)
    sx = np.zeros(4)
    sy = np.zeros(4)
    sx1 = np.zeros([4, 6])
    sy1 = np.zeros([4, 6])
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


def query(UX1, UY1, UX, UY, PX1, PY1, PX, PY,kr):
    dis1 = []
    dis2 = []
    poi1 = []
    poi2 = []
    for v in range(0, lenp):
        dis1.append(sqrt((-1/s)*((log(np.dot(UX1.T,PX1[v])) + log(np.dot(UY1.T,PY1[v]))))))
        dis2.append(vincenty((UY, UX), (PY[v],  PX[v])).meters)
    srr = adjust_r(UX, UY, UX1, UY1, kr)
    for n in range(0, lenp):
        if dis1[n] <= srr:
            poi1.append([PX[n], PY[n]])
        if dis2[n] <= kr:
            poi2.append([PX[n], PY[n]])
    res1 = commen(poi2, poi1)
    tmp1 = [val for val in poi2 if val in poi1]
    if len(poi1) == 0 & len(poi2) == 0:
        rec1 = 0.99999999
    elif len(poi1) == 0:
        return 0
    else:
        rec1 = len(tmp1) / len(poi1)
	#返回相似性和召回率
    return res1, rec1


def commen(p1, p2):
    tmp = [val for val in p1 if val in p2]
    if len(p2) == 0 & len(p1) == 0:
        return 0.99999999
    elif len(p1) == 0:
        return 0
    else:
        return len(tmp) / len(p1)
    print(tmp)


res = np.zeros(1000)
rec = np.zeros(1000)
kc = 0

for distance in range(100, 1100, 100):
    for i in range(0, 1000):
        h = random.randint(1, lu)
        UX[i] = float(user.ix[h]["lon"])
        UY[i] = float(user.ix[h]["lat"])
        UX1[i] = np.array(
            [user.ix[h]["x1"], user.ix[h]["x2"], user.ix[h]["x3"], user.ix[h]["x4"], user.ix[h]["x5"], user.ix[h]["x6"]], dtype=np.float)
        UY1[i] = np.array(
            [user.ix[h]["y1"], user.ix[h]["y2"], user.ix[h]["y3"], user.ix[h]["y4"], user.ix[h]["y5"], user.ix[h]["y6"]], dtype=np.float)
        res[i], rec[i] = query(UX1[i], UY1[i], UX[i], UY[i], PX1, PY1, PX, PY, distance)
    resemblance[kc] = sum(res) / 1000
    recall[kc] = sum(rec) / 1000
    result.iloc[kc, 4] = resemblance[kc]
    result.iloc[kc, 5] = recall[kc]
    kc = kc+1

result.to_csv("F:\zhudanyang\LBS1222\Flbsdataset2\\result\dimension-rq-simplegeo-6.csv.csv")