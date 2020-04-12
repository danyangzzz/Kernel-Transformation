import pandas as pd



def poidata(sname, filename):
    data = pd.read_csv("F:\zhudanyang\LBS1222\Flbsdataset2\simplegeo data.csv", index_col=0)
    # data = pd.read_csv("F:\zhudanyang\LBS9.30\Flbsdataset\simplegeo data.csv", index_col=0)
    l = data.shape[0]
    pdata = pd.read_csv("F:\zhudanyang\LBS1222\Flbsdataset2\%s.csv" % filename,index_col=0)
    j = 0
    for i in range(0,l):
        if data.ix[i]["poi"] == sname:
            j = j+1
            pdata.loc[j,:] = data.loc[i]
    pdata.to_csv("F:\zhudanyang\LBS1222\Flbsdataset2\%s.csv" % filename)

# poidata('Restaurants', 'restaurant1')
# poidata('Nightlife', 'Nightlife1')
# poidata('Beauty & Spas', 'Beauty1')
# poidata('Professional', 'professionals')
# poidata('Shopping', 'shopping')
# poidata('Restaurant', 'SimpleRestaurant')
poidata('Health Services','Health Services')