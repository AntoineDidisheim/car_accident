import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import didipack as didi
import geopandas
from parameters import *
from data import *
from map import *
import os
from sklearn.ensemble import RandomForestRegressor as RF
from sklearn.linear_model import LinearRegression as OLS
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score

##################
# set parameters
##################
par = Params()
par.update_model_name()

##################
# create save directory
##################
save_dir = par.model.res_dir + f'/{par.name}/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


# this function allow us to change in one point all plt.show with plt.close
def plt_show():
    if Constant.SHOW_PLOT:
        plt.show()
    else:
        plt.close()

# load the
df = pd.read_pickle(save_dir + '/all.p')

def r2(c,df_):
    return 1-((df_['y']-df_[c])**2).sum()/((df_['y']-df_['y'].mean())**2).sum()

def r2_all(df_):
    return pd.Series({'RF':r2('pred_rf',df_),'OLS':r2('pred_ols',df_),'Lasso':r2('pred_lasso',df_),'Benchmark':r2('bench',df_)})

print('Overall R^2')
print(r2_all(df))
r2_all(df).to_csv(save_dir+'overall_r2.csv')
# We see that: a) OLS overfit, b) the Lasso does not even manage to match the simple benchmark performance, c) the random forest outperform the benchmark out of sample

##################
# Computing performance through time
##################
res = []
for date in df[['date']].drop_duplicates().iloc[:,0]:
    date_end = date +pd.DateOffset(months=12)
    ind = (df['date']>=date) & (df['date']<date_end)

    # we don't compute the r² is there is not enough date as the data is missing in 2008
    if (date_end <= df['date'].max()) & (len(df.loc[ind,'date'].unique())>=250):
        r=r2_all(df.loc[ind,:])
        r.name = date_end
        res.append(r)

res=pd.DataFrame(res)
res.drop(columns='OLS').plot()
plt.ylabel(r'out-of-sample $R^2$')
plt.savefig(save_dir+'r2_across_time.png')
plt_show()

# we note that the performance off all models take a hit in 2013. We remember from explortaory analysis that
# in 2013, the trend changed and for the first time in years, the number of accident increased by a lot.
# to see if the RF got more suprised than the benchmark we will plot the relative performance

res['d'] = res['RF']-res['Benchmark']
res['d'].plot()
plt.ylabel(r'out-of-sample $R^2_{RF}-R^2_{Benchmark}$')
plt.savefig(save_dir+'r2_diff_across_time.png')
plt_show()
# we see that, while our model still outperform the benchmark in 2013, the relative performance dropped.
# unsurpsingly, this "black swan" affected our more complex model more than simple historical mean.


##################
# computing the performance per region
##################
t=df.groupby('r').apply(lambda x: r2_all(x)).dropna().sort_values('RF').reset_index(drop=True)

plt.scatter(t['Benchmark'],t['RF'],marker='+',color='k')
plt.hlines(0.0, t['Benchmark'].min(),t['Benchmark'].max(),colors='r')
plt.xlabel(r'Benchmark $R^2$')
plt.ylabel(r'Random Forest $R^2$')
plt.savefig(save_dir+'r2_regions_benchmark_v_RF.png')
plt_show()

# this plot showed us that the RF has some blind splot. For a few specific regions, the RF performance is much worst than the benchmark


##################
# understanding the poor region performance
##################

# e merge to the performance per region the average number of accident per region
t=df.groupby('r').apply(lambda x: r2_all(x)).dropna().sort_values('RF').reset_index()
t=t.merge(df.groupby('r')['y'].mean().reset_index())


plt.scatter(t['y'],t['RF'],marker='+',color='k')
plt.hlines(0.0, t['y'].min(),t['y'].max(),colors='r')
plt.xlabel(r'average #accident per region')
plt.ylabel(r'Random Forest $R^2$')
plt.savefig(save_dir+'r2_region_per_accident_size.png')
plt_show()

# this plot showed that the "poor" performance of the RF is entierly concentrated on regions with a small number of accidents
