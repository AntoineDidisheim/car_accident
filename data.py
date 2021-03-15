import pandas as pd
import numpy as np
from parameters import *
import didipack as didi

class Data:
    def __init__(self,par: Params):
        self.par = par

    def load_all(self, reload=False):
        if reload:
            df = []
            for n in ['accidents_2005_to_2007','accidents_2009_to_2011','accidents_2012_to_2014']:
                df.append(self.load_year(n))
            df=pd.concat(df)

            ##################
            # create regions
            ##################
            # we will create some squared regions of uk randomly to predict
            df['r'] = 0
            nb_square = 10
            q_long = np.linspace(df['longitude'].min(), df['longitude'].max(), 10)
            q_lat = np.linspace(df['latitude'].min(), df['latitude'].max(), 10)
            long = df['r'].values.copy()
            lat = df['r'].values.copy()

            for i in range(nb_square):
                long[df['longitude'] > q_long[i]] = long[df['longitude'] > q_long[i]] + 1
                lat[df['latitude'] > q_lat[i]] = lat[df['latitude'] > q_lat[i]] + 1

            # create a unique name
            r = []
            for i in range(len(long)):
                r.append(str(long[i]) + '-' + str(lat[i]))

            df['r'] = r


            df.to_pickle(self.par.data.dir+'all.p')
        else:
            df = pd.read_pickle(self.par.data.dir+'all.p')
        return df

    def load_year(self,name):
        df = pd.read_csv(f'{self.par.data.dir}{name}.csv')
        df.columns = [x.lower() for x in df.columns]
        df['date'] = pd.to_datetime(df['date'])
        df['month'] = df['date'].dt.year * 100 + df['date'].dt.month
        col_to_del = ['location_easting_osgr', 'location_northing_osgr','lsoa_of_accident_location', 'junction_detail', 'junction_control', 'carriageway_hazards', 'special_conditions_at_site']
        for c in col_to_del:
            del df[c]
        return df
self = Data(Params())

# lsoa_of_accident_location