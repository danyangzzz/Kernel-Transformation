encoding = 'utf-8'

from math import sqrt, log
import pandas as pd
import random
import numpy as np
from geopy.distance import vincenty

UX = np.zeros(1000)
UY = np.zeros(1000)
UX1 = np.zeros([1000, 6])
UY1 = np.zeros([1000, 6])


user = pd.read_csv("F:\zhudanyang\LBS1222\Flbsdataset2\\restaurant.csv", index_col=0)
lu = user.shape[0]
#poi1data = pd.read_csv("F:\zhudanyang\LBS9.30\Flbsdataset\\Nightlife2.csv", index_col=0)
# poi1data = pd.read_csv("F:\zhudanyang\LBS9.30\Flbsdataset\SimpleRestaurant.csv", index_col=0)
poi1data = pd.read_csv("F:\zhudanyang\LBS1222\Flbsdataset2\\Nightlife.csv", index_col=0)
lenp = poi1data.shape[0]


result = pd.read_csv("F:\zhudanyang\LBS1222\Flbsdataset2\\result\denmension-knn-yelp6.csv", index_col=0)


PX = np.zeros(lenp)
PY = np.zeros(lenp)
PX1 = np.zeros([lenp, 6])
PY1 = np.zeros([lenp, 6])
for b in range(0, lenp):
    t = b+1
    PX[b] = poi1data.ix[t]["lon"]
    PY[b] = poi1data.ix[t]["lat"]
    PX1[b] = np.array([poi1data.ix[t]["x1"], poi1data.ix[t]["x2"], poi1data.ix[t]["x3"], poi1data.ix[t]["x4"],poi1data.ix[t]["x5"], poi1data.ix[t]["x6"]],
                      dtype=np.float)
    PY1[b] = np.array([poi1data.ix[t]["y1"], poi1data.ix[t]["y2"], poi1data.ix[t]["y3"], poi1data.ix[t]["y4"],poi1data.ix[t]["y5"], poi1data.ix[t]["y6"]],
                      dtype=np.float)
 

s = 1.5508726289713732

def query(UX1, UY1, UX, UY, PX1, PY1, PX, PY,kp):
    dis1 = []
    dis2 = []
    poi1 = []
    poi2 = []
    for v in range(0, lenp):
        dis1.append(sqrt((-1/s)*((log(np.dot(UX1.T,PX1[v])) + log(np.dot(UY1.T,PY1[v]))))))
        dis2.append(vincenty((UY, UX), (PY[v],  PX[v])).meters)

    dis1 = np.array(dis1)
    dis2 = np.array(dis2)
    sorteddis1 = dis1.argsort()
    sorteddis2 = dis2.argsort()
    totald2 = 0
    totald1 = 0
    for j in range(0, kp):
        poi1.append([PX[sorteddis1[j]], PY[sorteddis1[j]]])
        poi2.append([PX[sorteddis2[j]], PY[sorteddis2[j]]])
        totald2 = dis2[sorteddis2[j]]+totald2
        totald1 = totald1 + (vincenty((UY, UX), (PY[sorteddis1[j]], PX[sorteddis1[j]])).meters)
    kdi = (totald1 - totald2) / len(poi2)

    kpc = commen(poi2, poi1)
    return kpc, kdi



def commen(p1, p2):
    tmp = [val for val in p1 if val in p2]
    return len(tmp) / len(p1)



resemblance = np.zeros(50)
displacement = np.zeros(50)
c2 = np.zeros(1000)
d2 = np.zeros(1000)
for k in range(1,51):
    for i in range(0, 1000):
        h = random.randint(1, lu)
        UX[i] = float(user.ix[h]["lon"])
        UY[i] = float(user.ix[h]["lat"])
        UX1[i] = np.array([user.ix[h]["x1"], user.ix[h]["x2"], user.ix[h]["x3"], user.ix[h]["x4"],user.ix[h]["x5"], user.ix[h]["x6"]], dtype=np.float)
        UY1[i] = np.array([user.ix[h]["y1"], user.ix[h]["y2"], user.ix[h]["y3"], user.ix[h]["y4"],user.ix[h]["y5"], user.ix[h]["y6"]], dtype=np.float)
        c2[i],d2[i] = query(UX1[i], UY1[i], UX[i], UY[i], PX1, PY1, PX, PY, k)
    o = k - 1
    resemblance[o] = sum(c2) / 1000
    displacement[o] = sum(d2) / 1000
    result.iloc[o, 0] = resemblance[o]
    result.iloc[o, 1] = displacement[o]


print(resemblance)
print(displacement)
result.to_csv("F:\zhudanyang\LBS1222\Flbsdataset2\\result\denmension-knn-yelp6.csv")