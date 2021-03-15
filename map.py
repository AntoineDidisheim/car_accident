import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import didipack as didi
import geopandas
from parameters import *

class MapPlotter:
    def __init__(self, par:Params):
        self.world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
        self.world = self.world[self.world['name'] == 'United Kingdom']
        self.par = par

    def plot(self,gdf, color_ = 'blue',marker_='+', legend_ = '', ax=None):
        if not type(gdf) == geopandas.geodataframe.GeoDataFrame:
            gdf = self.to_geopandas(gdf)
        if ax is None:
            ax =self.world.plot(color='white', edgecolor='black')
        if color_ is not None:
            gdf.plot(ax=ax, color=color_, marker=marker_, label=legend_)
        else:
            gdf.plot(ax=ax, marker=marker_, label=legend_)
        return ax

    def to_geopandas(self,df_):
        gdf = geopandas.GeoDataFrame(df_, geometry=geopandas.points_from_xy(df_['longitude'], df_['latitude']))
        return gdf