encoding = 'utf-8'

import pandas as pd
import random
import math
import numpy as np
from geopy.distance import vincenty
from math import radians, sqrt,tan,sin,cos,log

sigma = 0.5678020243184039
#  random.uniform(0.4, 0.7)

s = 0.5 / (sigma ** 2)

ini = pd.read_csv("F:\zhudanyang\LBS1222\Flbsdataset2\simplegeo data.csv", index_col=0)
l = ini.shape[0]

lon = np.zeros(l)
lat = np.zeros(l)
X = np.zeros(l)
Y = np.zeros(l)


#from geographic space to euclidean space
def coord4(lon1,lat1):
    lon0, lat0 = 151.195,-33.8855
    # simplegeo,-116,36151.18,-33.86
    # yelp-116.218,36.2114#
    lon0, lat0, lon1, lat1, = map(radians, [lon0, lat0, lon1, lat1])
    a = 6371
    yl = a*(lat1 - lat0)
    x = a * cos(lat0)
    xl = x * (lon1 - lon0)
    return xl,yl

#transform the data on the map
for i in range(0, l):
    lon[i] = (float(ini.ix[i]["lon"]))
    lat[i] = (float(ini.ix[i]["lat"]))
    X[i],Y[i] = coord4(lon[i],lat[i])


X0 = np.zeros(l)
Y0 = np.zeros(l)
Xmin = min(X)
Ymin = min(Y)
Xmax = max(X)
Ymax = max(Y)
min = min(Xmin,Ymin)
max = max(Xmax,Ymax)
Trange = (max - min)*2

#normalization
for i in range(0, l):
    X0[i] = (X[i] - min) / Trange
    Y0[i] = (Y[i] - min) / Trange
    ini.iloc[i,4] = X0[i]
    ini.iloc[i,5] = Y0[i]

#kernel transformation function
k = 6
def trans(x0):
    x = np.zeros(k)
    x[0] = math.exp(-s * (x0 ** 2))
    for i in range(1,k):
           x[i] = x[0] * math.sqrt(((2 * s) ** i) / math.factorial(i)) * (x0 ** i)
    return x


#transform the 2d points to multidimensional space
X1 = np.zeros([l,k])
Y1 = np.zeros([l,k])
for h in range(0, l):
    X1[h] = trans(X0[h])
    Y1[h] = trans(Y0[h])
    for i in range(0,k):
        ini.iloc[h,6+i] = X1[h][i]
        ini.iloc[h,14+i] = Y1[h][i]


ini.to_csv("J:\LBS\Flbsdataset2\yelp data1.csv")
ini.to_csv("F:\zhudanyang\LBS9.30\Flbsdataset\simplegeo data.csv")

#deal the POI in the file
def poidata(sname, filename):
    data = pd.read_csv("F:\zhudanyang\9.17\dataset\yelp data.csv", index_col=0)
    l = data.shape[0]
    pdata = pd.read_csv("F:\zhudanyang\9.17\dataset\%s.csv" % filename,index_col=0)
    j = 0
    for i in range(0, l):
        if data.ix[i]["poi"] == sname:
            j = j+1
            pdata.loc[j,:] = data.loc[i]
    pdata.to_csv("F:\zhudanyang\9.17\dataset\%s.csv" % filename)

poidata('Restaurants', 'restaurant')
poidata('Nightlife', 'Nightlife')
poidata('Beauty & Spas', 'Beauty')
poidata('Professional', 'professionals1')
poidata('Shopping', 'shopping')
poidata('Restaurant', 'SimpleRestaurant')

print("finish")
print(s)
print(min,Trange,max)
print(sigma)