encoding = 'utf-8'

from math import sqrt, log
import pandas as pd
import random
import numpy as np
import datetime


s = 1.5508726289713732


# 定义matrics，存储不同的k对应的结果
# 100个用户的测量结果取平均
c2 = np.zeros(100)
d2 = np.zeros(100)

UX1 = np.zeros([100, 8])
UY1 = np.zeros([100, 8])

user = pd.read_csv("F:\zhudanyang\LBS9.30\Flbsdataset\simplegeo data.csv", index_col=0)
lu = user.shape[0]
poi1data = pd.read_csv("F:\zhudanyang\LBS9.30\Flbsdataset\simplegeo data.csv", index_col=0)
lenp = poi1data.shape[0]

PX1 = np.zeros([lenp, 8])
PY1 = np.zeros([lenp, 8])
for b in range(0, lenp):
    PX1[b] = np.array([poi1data.ix[b]["x1"], poi1data.ix[b]["x2"], poi1data.ix[b]["x3"], poi1data.ix[b]["x4"],poi1data.ix[b]["x5"], poi1data.ix[b]["x6"], poi1data.ix[b]["x7"], poi1data.ix[b]["x8"]],
                      dtype=np.float)
    PY1[b] = np.array([poi1data.ix[b]["y1"], poi1data.ix[b]["y2"], poi1data.ix[b]["y3"], poi1data.ix[b]["y4"],poi1data.ix[b]["y5"], poi1data.ix[b]["y6"], poi1data.ix[b]["y7"], poi1data.ix[b]["y8"]],
                      dtype=np.float)

# 测试K的个数对实验的影响，1-50
starttime = datetime.datetime.now()
for i in range(0, 100):
    # 定义区域内兴趣点坐标
    dis1=[]
    h = random.randint(1, lu)
    UX1[i] = np.array([user.ix[h]["x1"], user.ix[h]["x2"], user.ix[h]["x3"], user.ix[h]["x4"],user.ix[h]["x5"], user.ix[h]["x6"], user.ix[h]["x7"], user.ix[h]["x8"]], dtype=np.float)
    UY1[i] = np.array([user.ix[h]["y1"], user.ix[h]["y2"], user.ix[h]["y3"], user.ix[h]["y4"],user.ix[h]["y5"], user.ix[h]["y6"], user.ix[h]["y7"], user.ix[h]["y8"]], dtype=np.float)
    for v in range(0, lenp):
        dis1.append(sqrt((-1/s)*((log(np.dot(UX1[i].T,PX1[v])) + log(np.dot(UY1[i].T,PY1[v]))))))
    dis1 = np.array(dis1)
    sorteddis1 = dis1.argsort()
    PX=[]
    for j in range(0, 20):
        PX.append([PX1[sorteddis1[j]], PY1[sorteddis1[j]]])

endtime = datetime.datetime.now()
print(endtime - starttime)