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

# this function allow us to change in one point all plt.show with plt.close
def plt_show():
    if Constant.SHOW_PLOT:
        plt.show()
    else:
        plt.close()


##################
# create saving directory
##################
save_dir = par.model.res_dir + f'/{par.name}/'
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
# create a matrix of predictors and target
##################

# we compute the average per region of the weather conditions and road_surfac econdition. We also keep the month and the day of the week tout take into account cyclical patterns
final = df.loc[:, ['r', 'date']]
for c in ['weather_conditions','road_surface_conditions','month','day_of_week','r']:
    for u in df[c].unique():
        final[u] = (df[c] == u) * 1
final=final.groupby(['r', 'date']).mean().reset_index()

# we add our target variable --> the total number of accident per region per day.
final=final.merge(df.groupby(['r', 'date'])['accident_index'].count().reset_index().rename(columns={'accident_index': 'y'}))

##################
# expanding window cross-validation procedure
##################
# create time chunk of 6 months
final['T']= (final['date'].dt.month<=6)*1+final['date'].dt.year*10

T = final['T'].sort_values().unique()
rf_models = []

dash = '-' * 150
ft = '{:<10s}{:>14s}{:>14s}{:>14s}{:>14s}{:>14s}{:>14s}{:>14s}{:>14s}'
f = '{:<10s}{:>14f}{:>14f}{:>14f}{:>14f}{:>14f}{:>14f}{:>14f}{:>14f}'
print(dash)
print(ft.format('Time\R^2','RF is', 'RF oos','OLS is', 'OLS oos','LASSO is', 'LASSO oos'
                ,'Bench is', 'Bench oos'))
print(dash)
res = []
for t in range(1,len(T)):
    # create train and test sample (and split it in x,y)
    train=final.loc[final['T']<=T[t-1],:]
    test=final.loc[final['T']==T[t],:]

    train_x = train.drop(columns=['y','date','r','T']).values
    train_y = train[['y']].values.flatten()

    test_res = test[['date','r','y','T']].copy()
    test_x = test.drop(columns=['y','date','r','T']).values
    test_y = test[['y']].values.flatten()

    m=RF(max_depth=par.model.rf_max_depth,n_jobs=-1)
    m.fit(train_x,train_y)
    test_res['pred_rf']=m.predict(test_x)
    is_rf = m.score(train_x,train_y) # we compute both the in and out of sample r² of each model to get a feel during training of the performance
    oos_rf = np.clip(m.score(test_x,test_y),-1,1) # we clip the r² to -1 to avoid printing very large negative numbers

    m=OLS(fit_intercept=True)
    m.fit(train_x,train_y)
    test_res['pred_ols']=m.predict(test_x)
    is_ols = m.score(train_x,train_y)
    oos_ols = np.clip(m.score(test_x,test_y),-1,1)

    m=Lasso(fit_intercept=True,alpha=par.model.lasso_penalization)
    m.fit(train_x,train_y)
    test_res['pred_lasso']=m.predict(test_x)
    is_lasso = m.score(train_x,train_y)
    oos_lasso = np.clip(m.score(test_x,test_y),-1,1)

    ##################
    # compute benchmark performance
    ##################
    # we predict the mean of the region in the training sample and use it as a prediction in the training and test sample
    temp = train.groupby('r')['y'].mean().reset_index().rename(columns={'y':'bench'})
    test_res = test_res.merge(temp)
    temp = train.merge(temp)

    is_benchmark = r2_score(temp['y'],temp['bench'])
    oos_benchmark = np.clip(r2_score(test_res['y'],test_res['bench']), -1, 1)

    print(f.format(str(T[t]), is_rf, oos_rf, is_ols, oos_ols, is_lasso, oos_lasso, is_benchmark,oos_benchmark))

    # save the results
    res.append(test_res)




# save the vector containing all prediciton

pd.concat(res).to_pickle(save_dir+'all.p')

