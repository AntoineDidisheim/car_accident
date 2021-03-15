import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import didipack as didi
import geopandas
from parameters import *
from data import *
from map import *
import os

##################
# set parameters
##################
par = Params()


# this function allow us to change in one point all plt.show with plt.close
def plt_show():
    if Constant.SHOW_PLOT:
        plt.show()
    else:
        plt.close()


##################
# create saving directory
##################
save_dir = par.model.res_dir + '/exploration/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

##################
# load data
##################
# if we load the data for the first time, we need to set reload=True, hence the try catch
try:
    df = Data(par).load_all()
except:
    df = Data(par).load_all(True)

##################
# View map dispersion
##################
map = MapPlotter(par)
# since there is a lot of accident each day we arbitrairly select one day to plot
rnd_date = np.random.choice(df['date'])
t = df.loc[df['date'] == rnd_date, :]
map.plot(t)
plt.title(f'Accident on {str(rnd_date).split("T")[0]}')
plt.tight_layout()
plt.savefig(save_dir + 'rnd_date_map.png')
plt_show()

##################
# number of casualities and severity
##################
# histogram of number of death per accident
plt.hist(df['number_of_casualties'], bins=100)
plt_show()
# the many accidents with extremes makes it hard to see, we therefore look at the distribution conditional on death>0 and death <10
plt.hist(df.loc[(df['number_of_casualties'] <= 10), 'number_of_casualties'], bins=12)
plt.savefig(save_dir + 'nb_casuality_hist.png')
plt_show()

# same on accident severity
plt.hist(df['accident_severity'], bins=10)
plt.savefig(save_dir + 'nb_casuality_hist.png')
plt_show()

##################
# nb accident depending on conditions
##################
save_dir_cond = save_dir + 'cond/'
if not os.path.exists(save_dir_cond):
    os.makedirs(save_dir_cond)
df['month_nb'] = df['date'].dt.month
df.head()
col = ['day_of_week', 'road_type', 'pedestrian_crossing-human_control', 'pedestrian_crossing-physical_facilities',
       'light_conditions', 'weather_conditions', 'road_surface_conditions', 'did_police_officer_attend_scene_of_accident','month_nb',
       'speed_limit','urban_or_rural_area']
for c in col:

    t = df.groupby(c)['accident_index'].count()
    print(t)
    plt.bar(t.index, t.values)
    try:
        ml = np.max([len(x) for x in t.index])
    except:
        ml = 1
    if ml > 4:
        plt.xticks(rotation=90)
    plt.title(c)
    plt.tight_layout()
    plt.savefig(save_dir_cond + c + '.png')
    plt_show()

t = df.groupby('time')['accident_index'].count().plot()
plt.savefig(save_dir_cond + 'time.png')
plt_show()


##################
# time series
##################
save_dir_ts = save_dir + 'ts/'
if not os.path.exists(save_dir_ts):
    os.makedirs(save_dir_ts)

df.groupby('date')['accident_index'].count().rolling(252).mean().dropna().plot()
plt.savefig(save_dir_ts+'nb_accident.png')
plt_show()

for s in df['accident_severity'].unique():
    df.loc[df['accident_severity']==s,:].groupby('date')['accident_index'].count().rolling(252).mean().dropna().plot()
    plt.savefig(save_dir_ts+f'nb_accident_severity{s}.png')
    plt_show()

##################
# illustrate the region
##################

r = df['r'].unique()[0]
ax = map.plot(gdf=df.loc[df['r']==r,:].head(100),color_=None,legend_=r)
for r in df['r'].unique()[1:]:
    ax = map.plot(gdf=df.loc[df['r'] == r, :].head(10000),color_=None, legend_=r,ax=ax)

plt.savefig(save_dir+f'region_illustration.png')
plt_show()

