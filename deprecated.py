#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 16:09:39 2022

@author: michelev
"""
import numpy as np
import pandas as pd
import geopandas as gpd

#import statsmodels.api as sm
from scipy import stats

import ast

#import linearmodels as lm
#from causaldata import gapminder
from scipy.sparse import csr_matrix, load_npz
import csv

import libpysal
from esda.moran import Moran
import seaborn as sns

from libpysal.weights import Kernel
import os

import statsmodels.formula.api as sm

import matplotlib.pyplot as plt
from matplotlib import rc
from stargazer.stargazer import Stargazer

from shapely.ops import unary_union
from shapely.geometry import MultiPolygon

from Base import reshape_, createCirclesGrid
from geo import GeoData
from yelp import YelpClient
from grubhub1 import GrubhubClient

#%%

machine = '/Users/michelev/spatial-competition-food/'
    
GH = GrubhubClient()
GD = GeoData()

census_df = GD.census_df
CRS_LATLON = 'GCS_WGS_1984'
CRS_M = 'EPSG:32118'

census_df.geometry = census_df.geometry.set_crs(CRS_LATLON).to_crs(CRS_M)

YP = YelpClient()
YP.updateDate('2022-10-31')
YP.retrieveData()
YP.yelp_data = YP.yelp_data.sort_index(axis=0)
#YP.yelp_data.categories = YP.yelp_data.categories.apply(lambda x: sorted(x))

#YP.yelp_data['BoroCT2020'] = YP.yelp_data['BoroCT2020'].astype(int)
#YP.yelp_data['concatenated_categories'] = YP.yelp_data.categories.apply("+".join)
YP.yelp_data = YP.yelp_data.join(pd.get_dummies(YP.yelp_data.transactions.apply(ast.literal_eval).explode()).groupby(level=0).sum())

rests_df = YP.yelp_data.merge(pd.read_csv(YP.machine + 'data/' + 'new_merged_dataset.csv', index_col=[0]),on='id',how='left')
rests_df.loc[rests_df.isGH!=True,'isGH']=False
rests_df.set_index('id',inplace=True)

mask = ((YP.yelp_data.pickup==1) | (YP.yelp_data.delivery==1))

dummies_CT =  pd.get_dummies(YP.yelp_data.BoroCT2020.apply(pd.Series).stack()).groupby(level=0).sum()
dummies_CT = dummies_CT[mask]
dummies_CT.columns = dummies_CT.columns.astype(int)

CT_columns = dummies_CT.columns

dummies_CT_sparse = csr_matrix(dummies_CT)

filtering_matrix_CT_sparse = dummies_CT_sparse.dot(dummies_CT_sparse.T)


#dummies_CT = 


##
dummies_CT_del = pd.read_csv(machine + '/grubhub_data/data_analysis/where_restaurants_deliver_by_Yelp_id.csv', index_col=[0])

dummies_CT_del = pd.read_csv(machine + '/grubhub_data/data_analysis/where_restaurants_deliver_by_GH_id.csv', index_col=[0])
GH_data = pd.read_csv(GH.machine + GH.folder + GH.analysis_folder + 'restaurants_delivering_from_tracts.csv')

#delivery_mode_est = pd.read_csv(GH.machine + GH.folder + GH.analysis_folder + 'restaurants_platform_delivery.csv')

dummies_CT_del = dummies_CT_del.reset_index().merge(GH_data[['restaurant_id', 'merchant_id', 'delivery_mode']], on = 'restaurant_id')
#dummies_CT_del = dummies_CT_del.reset_index().merge(delivery_mode_est, on = 'merchant_id')
#dummies_CT_del = dummies_CT_del.merge(delivery_network.PPD, on = 'merchant_id')
#dummies_CT_del = dummies_CT_del.merge(delivery_sums, on = 'merchant_id')
#dummies_CT_del=dummies_CT_del.set_index('id')
del_mode = 'SUPPLEMENTAL_DELIVERY_AS_GHD'
dummies_CT_del = dummies_CT_del.loc[dummies_CT_del.delivery_mode==del_mode]
dummies_CT_del.pop('delivery_mode')
dummies_CT_del.pop('merchant_id')
#dummies_CT_del.pop('restaurant_id')
#dummies_CT_del.pop('index')
#dummies_CT_del.pop('EstimatedPlatformDelivery')

##
dummies_CT_del.set_index('restaurant_id',inplace=True)

GH_data.set_index('restaurant_id',inplace=True)

missing_kwds = dict(color='lightgrey', hatch = '///')
for idx in dummies_CT_del.index:
  
    boro = int(GH_data.loc[idx,'BoroCT2020'])
    
    if os.path.exists(machine + 'figures/' + del_mode + '/' + str(boro) + '/' + str(idx) + '.png'):
        print('Already generated image')
        continue
    
    if not os.path.exists(machine + 'figures/' + del_mode + '/' + str(boro) + '/'):
        os.makedirs(machine + 'figures/' + del_mode + '/' + str(boro) + '/')

        int(GH_data.loc[idx,'BoroCT2020'])
    covered_CTs = pd.DataFrame(dummies_CT_del.loc[idx])
    covered_CTs.rename_axis(index='BoroCT2020', inplace=True)
    covered_CTs.rename({idx:'DeliveringTo'}, axis=1,inplace=True)
    covered_CTs.loc[boro, 'DeliveringTo'] = 2
    covered_CTs.index=covered_CTs.index.astype(int)
    
    cc = census_df.merge(covered_CTs, on = 'BoroCT2020', how = 'left')
    #cc.loc[cc.DeliveringTo.isna(), 'DeliveringTo']=0
    #cc.loc[cc.SituatedIn.isna(), 'SituatedIn']=0
    
    fig, ax = plt.subplots(figsize =(25,25))
    
    cc.plot(column = 'DeliveringTo', ax=ax, missing_kwds = missing_kwds)
    fig.savefig(machine + 'figures/' + del_mode + '/' + str(boro) + '/' + str(idx) + '.png', bbox_inches='tight')
    #cc.plot(column = 'SituatedIn', ax=ax, missing_kwds = missing_kwds)



##




# dummies_CT_del.columns = dummies_CT_del.columns.astype(int)
# dummies_CT_del = dummies_CT_del.loc[:,dummies_CT.columns]
# dummies_CT_del = pd.concat((pd.DataFrame(YP.yelp_data.index, index=YP.yelp_data.index),dummies_CT_del),axis=1)
# dummies_CT_del = dummies_CT_del[mask]
# dummies_CT_del.pop('id')
# dummies_CT_del = dummies_CT_del.sort_index(axis=1)
# dummies_CT_del = dummies_CT_del.sort_index(axis=0)
# dummies_CT_del = dummies_CT_del.fillna(0)
# dummies_del_sparse = csr_matrix(dummies_CT_del)
# #delivering_matrix =  dummies_CT_sparse.dot(dummies_del_sparse.T)

# #del dummies_CT, dummies_CT_del

# # category_dummies = pd.get_dummies(df.categories.apply(pd.Series).stack()).groupby(level=0).sum()
# # #category_dummies[category_dummies>0]=1
# # #category_dummies.loc['laotian','vietnamese']=1
# #pd.get_dummies(df.categories.apply(pd.Series).stack()).groupby(level=0)
# #category_dummies = category_dummies.drop_duplicates()
# #category_dummies = category_dummies.join(df, how = 'left')
# #category_dummies = category_dummies.groupby('concatenated_categories').sum()

# h_matrix = load_npz(machine + 'yelp_data/similarity_network/restaurant_similarity_angles.npz')


# h_matrix = h_matrix[mask,:]
# h_matrix = h_matrix[:,mask]



# communities_dataset = pd.read_csv('communities_BoroCT2020.csv', index_col=[0])
# communities_dataset = communities_dataset[mask]
# communities_dataset = communities_dataset.reset_index()

# communities_dataset.loc[communities_dataset['community_1.5']==9,'community_1'] = 9
# communities_dataset.loc[communities_dataset['community_1.5']==9,'community_1.5'] = 14
# communities_dataset.loc[communities_dataset['community_1.5']==9,'community_2'] = 16
 
# CT_community_count = communities_dataset.groupby(['BoroCT2020', 'community_1.5']).count()['id']
# CT_count = communities_dataset.groupby('BoroCT2020').count()['id']

# CT_community_frac = pd.DataFrame(CT_community_count).join(pd.DataFrame(CT_count), on = 'BoroCT2020', lsuffix = '_com', rsuffix = '_tot')
# CT_community_frac['pi'] = CT_community_frac['id_com']/CT_community_frac['id_tot']
# CT_community_frac['pi_div_tot'] = CT_community_frac['id_com']/CT_community_frac['id_tot']**2
# CT_community_frac['pi2'] = CT_community_frac['pi']**2
# #CT_community_frac = CT_community_frac.reset_index()
# # CT_pi = CT_community_frac['pi'].copy()
# # for col in CT_pi['community_1.5']:
# #     CT_pi.loc[:,col] = CT_pi.loc[CT_pi['community_1.5']==col,'pi']
# # CT_pi = CT_pi.set_index('BoroCT2020')
# # CT_pi.pop('community_1.5')
# # CT_pi.pop('pi')
# # CT_pi = CT_pi.sort_index(axis=1)
# # CT_pi = CT_pi.fillna(0)
# # CT_pi = CT_pi.groupby(level=0).sum()
# # CT_HHI = CT_community_frac.groupby('BoroCT2020').sum('pi2')

# pi_vector = communities_dataset.merge(CT_community_frac, on = ['BoroCT2020','community_1.5']).set_index('id')
# pi_vector = pi_vector.sort_index(axis=0)
# pi_vector = pi_vector['pi_div_tot']
# #pi_vector = CT_pi.values
# pi_matrix = np.outer(pi_vector,pi_vector)


# dummies_CT_sparse_0 = dummies_CT_sparse.copy()
# dummies_CT_sparse_0.setdiag(0)

# h_CT = dummies_CT_sparse.T.dot(h_matrix).dot(dummies_CT_sparse)

# p_CT = dummies_CT_sparse.T.dot(pi_matrix)
# p_CT = p_CT.dot(dummies_CT_sparse_0.todense())

# hp_matrix = h_matrix.multiply(pi_matrix)

# hp_CT = dummies_CT_sparse.T.dot(hp_matrix).dot(dummies_CT_sparse)


# h_CT = h_CT.todense()
# p_CT = p_CT
# hp_CT = hp_CT.todense()


# #h_del_matrix = delivering_matrix.multiply(h_matrix)

# #h_del_CT = dummies_CT_sparse.T.dot(h_del_matrix).dot(dummies_CT_sparse)

# #p_del_matrix = delivering_matrix.multiply(pi_matrix)

# #p_del_CT = dummies_CT_sparse.T.dot(p_del_matrix)
# #p_del_CT = p_del_CT.dot(dummies_CT_sparse.todense())


# #hp_del_matrix = delivering_matrix.multiply(hp_matrix)

# #hp_del_CT = dummies_CT_sparse.T.dot(hp_del_matrix).dot(dummies_CT_sparse)

# #h_del_CT = h_del_CT.todense()
# #p_del_CT = p_del_CT
# #hp_del_CT = hp_del_CT.todense()



# areas = census_df.loc[CT_columns, 'Shape_Area']/1e6
# h_CT_R = {}
# p_CT_R = {}
# hp_CT_R = {}

# # h_del_CT_R = {}
# # p_del_CT_R = {}
# # hp_del_CT_R = {}

# num_physical_competitors_R = {}
# num_physical_delivery_competitors_R = {}
# num_delivery_competitors_R = {}

# num_physical_competitors_circle = {}
# num_physical_delivery_competitors_circle = {}
# num_delivery_competitors_circle = {}

# den_physical_competitors_R = {}
# den_physical_delivery_competitors_R = {}
# den_delivery_competitors_R = {}

# den_physical_competitors_circle = {}
# den_physical_delivery_competitors_circle = {}
# den_delivery_competitors_circle = {}

# num_physical_competitors_R[0] = reshape_(dummies_CT_sparse.sum(0))
# num_delivery_competitors_R[0] = reshape_(dummies_del_sparse.sum(0))
# num_physical_delivery_competitors_R[0] = reshape_(dummies_CT_sparse.T.dot(dummies_del_sparse.sum(1)>0))

# den_physical_competitors_R[0] = num_physical_competitors_R[0]/areas
# den_delivery_competitors_R[0] = num_delivery_competitors_R[0]/areas
# den_physical_delivery_competitors_R[0] = num_delivery_competitors_R[0]/areas


# h_CT_R[0] = np.diagonal(h_CT)/(num_physical_competitors_R[0]*(num_physical_competitors_R[0]-1))
# p_CT_R[0] = np.diagonal(p_CT)
# hp_CT_R[0] = np.diagonal(hp_CT)

# # h_del_CT_R[0] = np.diagonal(h_del_CT)/num_physical_competitors_R[0]
# # p_del_CT_R[0] = np.diagonal(p_del_CT)
# # hp_del_CT_R[0] = np.diagonal(hp_del_CT)

# r=0

# # B=40
# # bootstrap_folder = 'bootstrap_quantile_num/'
# # bstrap = pd.read_csv(machine + bootstrap_folder + '/diversity_' + str(0) + '.csv', index_col=[0])
# # h_rel = pd.DataFrame(h_CT_R[0]/bstrap['disparity'])
# # p_rel = pd.DataFrame(p_CT_R[0]/bstrap['balance'])
# # hp_rel = pd.DataFrame(hp_CT_R[0]/bstrap['diversity'])

# # for b in range(1,B):
# #     bstrap = pd.read_csv(machine + bootstrap_folder + '/diversity_' + str(b) + '.csv', index_col=[0])
# #     h_rel = pd.concat((h_rel,pd.DataFrame(h_CT_R[0]/bstrap['disparity'])),axis=1)
# #     p_rel = pd.concat((h_rel,pd.DataFrame(p_CT_R[0]/bstrap['balance'])),axis=1)
# #     hp_rel = pd.concat((h_rel,pd.DataFrame(hp_CT_R[0]/bstrap['diversity'])),axis=1)
# #     #avg_df += 
# # #avg_df /= B

# # h_rel = h_rel.mean(1).values
# # p_rel = p_rel.mean(1).values
# # hp_rel = hp_rel.mean(1).values

# #%%

# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "sans-serif",
#     "font.sans-serif": "Helvetica",
# })

# rests_df = YP.yelp_data.merge(pd.read_csv(YP.machine + 'data/' + 'new_merged_dataset.csv', index_col=[0]),on='id',how='left')
# rests_df.loc[rests_df.isGH!=True,'isGH']=False
# rests_df.set_index('id',inplace=True)
# h_matrix_=h_matrix.todense()

# mask_d = (rests_df.loc[mask,'review_count_x']>0) & (rests_df.loc[mask,'rating_count']>0) & (rests_df.loc[mask,'isGH'])
# mask_nd = (rests_df.loc[mask,'review_count_x']>0) & (~rests_df.loc[mask,'isGH'])
# h_sum = h_matrix.sum(1)

# fact_h = pd.DataFrame({'hsum':reshape_(h_sum), 'del':mask_d.values, 'no_del':mask_nd.values})

# sns.histplot(data=fact_h, x='hsum', hue='del', stat='density', alpha = 0.25, kde=True, common_norm=False)
# #p=sns.histplot(data=h_nd, color ='blue',stat='density', alpha = 0.05, kde=True)
# plt.legend(loc='upper left', labels=['$i$ on the platform', '$i$ not on the platform'])
# plt.xlabel('$\sum_j h_{ij}$')
# plt.savefig('figures/disparity_distribution.png', dpi=300)

# communities_dataset.rename({'community_1.5':'community15'},inplace=True, axis=1)
# communities_dataset.community15=np.where(communities_dataset.community15>8,communities_dataset.community15-1,communities_dataset.community15)
# fact_p = pd.DataFrame((communities_dataset.loc[communities_dataset.isGH].groupby('community15').count()/sum(communities_dataset.isGH))['id'])
# fact_p=fact_p.merge(pd.DataFrame((communities_dataset.loc[~communities_dataset.isGH].groupby('community15').count()/sum(~communities_dataset.isGH))['id']), on = 'community15', suffixes=('-del', '-nondel'), how = 'left')
# fact_p=fact_p.rename({'id-del': 'Yes', 'id-nondel': 'No'},axis=1)
# fact_p.reset_index(inplace=True)
# fact_p = pd.melt(fact_p, id_vars=['community15'])
# fact_p=fact_p.rename({'variable': 'On the platform'},axis=1)

# plt.figure(0)
# ax=sns.catplot(x = 'community15', y='value', hue = 'On the platform',data=fact_p, kind='bar')
# plt.xlabel('Community')
# plt.ylabel('Frequency')
# plt.savefig('figures/community_catplot.png', dpi=300)

# #%%

# ### Show boundaries of delivery areas found with method 'findDeliveryAreas' in 'grubhub.py'

# for res in [0.75, 1, 1.25]:
#     delivery_areas = pd.read_csv(GH.machine + GH.folder + GH.network_folder + 'communities_CT_weighted_full_' + str(res) + '.csv', index_col='BoroCT2020')
#     delivery_areas = gpd.GeoDataFrame(delivery_areas.merge(census_df, on='BoroCT2020'))
    
#     fig, ax = plt.subplots(figsize =(25,25))
#     delivery_areas.plot(column='community', ax=ax)
#     fig.savefig('figures/communities_CT_weighted_full_' + str(res) +'.png', bbox_inches='tight')
    
# #%%

# #census_df.set_index('BoroCT2020', inplace=True)

# for res in [1,1.25]:
#     delivery_areas = pd.read_csv(GH.machine + GH.folder + GH.network_folder + 'communities_CT_weighted_full_' + str(res) + '.csv', index_col='BoroCT2020')
#     delivery_areas = gpd.GeoDataFrame(delivery_areas.merge(census_df, on='BoroCT2020'))
    
#     delivery_areas = delivery_areas.reset_index().merge(delivery_areas[['NTA2020','community']].groupby('NTA2020').agg(lambda x: pd.Series.mode(x)[0]), on = 'NTA2020', suffixes = ('', '_NTA'))
#     delivery_areas.community_NTA = delivery_areas.community_NTA.astype(int)
#     fig, ax = plt.subplots(figsize =(25,25))
#     delivery_areas.plot(column='community_NTA', ax=ax)
#     delivery_areas.sort_values(by = 'BoroCT2020', inplace=True)
    
#     delivery_network = pd.read_csv(GH.machine + GH.folder + GH.network_folder + 'where_firms_deliver_by_restaurant_id.csv')
#     #delivery_network = pd.read_csv(GH.machine + GH.folder + GH.analysis_folder + 'where_restaurants_deliver_by_GH_id.csv')
#     #all_rests = pd.read_csv(GH.machine + GH.folder + GH.analysis_folder + 'restaurants_delivering_from_tracts.csv', usecols = ['restaurant_id', 'merchant_id', 'BoroCT2020'])
#     #delivery_network = delivery_network.merge(all_rests, on = 'restaurant_id')
#     delivery_mode_est = pd.read_csv(GH.machine + GH.folder + GH.analysis_folder + 'restaurants_platform_delivery.csv')
#     delivery_network = delivery_network.merge(delivery_mode_est[['merchant_id', 'EstimatedPlatformDelivery']], on = 'merchant_id')
#     delivery_network = delivery_network.loc[delivery_network.EstimatedPlatformDelivery==True]
    
#     dummies_comm = pd.get_dummies(delivery_areas['community'])
#     dummies_comm = dummies_comm.dot(dummies_comm.T)
#     dummies_comm = ~dummies_comm.astype(bool)
#     dummies_comm = dummies_comm.astype(int)
#     dummies_comm.index = delivery_areas.BoroCT2020
#     dummies_comm.columns = delivery_areas.BoroCT2020
    
#     dummies_comm = delivery_network[['merchant_id', 'BoroCT2020']].merge(dummies_comm, on = 'BoroCT2020')
#     dummies_comm.set_index('merchant_id', inplace=True)
#     dummies_comm.pop('BoroCT2020')
#     dummies_comm.sort_index(axis=0, inplace=True)
#     dummies_comm.sort_index(axis=1, inplace=True)
    
#     delivery_network.set_index('merchant_id', inplace=True)
#     #delivery_network.pop('restaurant_id')
#     delivery_network.pop('BoroCT2020')
#     delivery_network.pop('EstimatedPlatformDelivery')
#     delivery_network.sort_index(axis=0, inplace=True)
#     delivery_network.sort_index(axis=1, inplace=True)
    
#     filtered_dummies_CT_del_tot = csr_matrix(delivery_network).multiply(csr_matrix(dummies_comm))
    
    
#     census_df.sort_index(axis=0,inplace=True)
#     census_df['number_outside']= reshape_(filtered_dummies_CT_del_tot.sum(0))
#     census_df['share_outside']= reshape_(filtered_dummies_CT_del_tot.sum(0))/delivery_network.sum(0).values
#     fig, ax = plt.subplots(figsize =(25,25))
#     census_df.plot('number_outside', ax=ax, cmap = 'magma_r', legend=True)
#     fig.savefig('figures/number_outside_' + str(res) +'.png', bbox_inches='tight')
    
#     fig, ax = plt.subplots(figsize =(25,25))
#     census_df.plot('share_outside', ax=ax, cmap = 'magma_r', legend=True)
#     fig.savefig('figures/share_outside_' + str(res) +'.png', bbox_inches='tight')
# #%%

# res = 0.75
# delivery_areas = pd.read_csv(GH.machine + GH.folder + GH.network_folder + 'communities_CT_weighted_full_' + str(res) + '.csv', index_col='BoroCT2020')
# delivery_areas = gpd.GeoDataFrame(delivery_areas.merge(census_df, on='BoroCT2020'))

# delivery_areas = delivery_areas.reset_index().merge(delivery_areas[['NTA2020','community']].groupby('NTA2020').agg(lambda x: pd.Series.mode(x)[0]), on = 'NTA2020', suffixes = ('', '_NTA'))
# delivery_areas.community_NTA = delivery_areas.community_NTA.astype(int)
# fig, ax = plt.subplots(figsize =(25,25))
# delivery_areas.plot(column='community_NTA', ax=ax)
# delivery_areas.sort_values(by = 'BoroCT2020', inplace=True)

# delivery_network = pd.read_csv(GH.machine + GH.folder + GH.network_folder + 'where_firms_deliver_by_restaurant_id.csv')
# #delivery_network = pd.read_csv(GH.machine + GH.folder + GH.analysis_folder + 'where_restaurants_deliver_by_GH_id.csv')
# #all_rests = pd.read_csv(GH.machine + GH.folder + GH.analysis_folder + 'restaurants_delivering_from_tracts.csv', usecols = ['restaurant_id', 'merchant_id', 'BoroCT2020'])
# #delivery_network = delivery_network.merge(all_rests, on = 'restaurant_id')
# delivery_mode_est = pd.read_csv(GH.machine + GH.folder + GH.analysis_folder + 'restaurants_platform_delivery.csv')
# delivery_network = delivery_network.merge(delivery_mode_est[['merchant_id', 'EstimatedPlatformDelivery']], on = 'merchant_id')
# #delivery_network = delivery_network.loc[delivery_network.EstimatedPlatformDelivery==True]

# dummies_comm = pd.get_dummies(delivery_areas['community'])
# dummies_comm = dummies_comm.dot(dummies_comm.T)
# dummies_comm = ~dummies_comm.astype(bool)
# dummies_comm = dummies_comm.astype(int)
# dummies_comm.index = delivery_areas.BoroCT2020
# dummies_comm.columns = delivery_areas.BoroCT2020

# dummies_comm = delivery_network[['merchant_id', 'BoroCT2020']].merge(dummies_comm, on = 'BoroCT2020')
# dummies_comm.set_index('merchant_id', inplace=True)
# dummies_comm.pop('BoroCT2020')
# dummies_comm.sort_index(axis=0, inplace=True)
# dummies_comm.sort_index(axis=1, inplace=True)

# delivery_network.set_index('merchant_id', inplace=True)
# #delivery_network.pop('restaurant_id')
# delivery_network.pop('BoroCT2020')
# delivery_network.pop('EstimatedPlatformDelivery')
# delivery_network.sort_index(axis=0, inplace=True)
# delivery_network.sort_index(axis=1, inplace=True)

# filtered_dummies_CT_del_tot = csr_matrix(delivery_network).multiply(csr_matrix(dummies_comm))


# census_df.sort_index(axis=0,inplace=True)
# census_df['number_outside']= reshape_(filtered_dummies_CT_del_tot.sum(0))
# census_df['share_outside']= reshape_(filtered_dummies_CT_del_tot.sum(0))/delivery_network.sum(0).values
# fig, ax = plt.subplots(figsize =(25,25))
# census_df.plot('number_outside', ax=ax, cmap = 'magma_r', legend=True)
# fig.savefig('figures/number_outside_' + str(res) +'.png', bbox_inches='tight')

# fig, ax = plt.subplots(figsize =(25,25))
# census_df.plot('share_outside', ax=ax, cmap = 'magma_r', legend=True)
# fig.savefig('figures/share_outside_' + str(res) +'.png', bbox_inches='tight')

# #%%

# ### CREATING GIS FILES AND PLOTS FOR NUMBER OF INTERSECTING CIRCLES
# ## EACH DELIVERY AREA (Manhattan, Brooklyn + Queens, Bronx, Staten Island) separately

# # resolution = distance between each point on the grid
# resolution = 500
# # radius = of each circle drawn from each point of the grid
# radius = 5000

# all_census_df = None

# # importing GIS data in which only the largest polygon of the census tract multipolygon is retrieved
# # (operation necessary because of problem with Manhattan)
# census_df = gpd.read_file(machine + 'nyc_geodata/census_tracts_boundaries/census_tracts_largest.shp')
# CRS_LATLON = 'GCS_WGS_1984'
# CRS_M = 'EPSG:32118'
# census_df.geometry = census_df.geometry.set_crs(CRS_LATLON).to_crs(CRS_M)

# for delivery_area in ['Manhattan', 'BrooklynQueens', 'Bronx', 'Staten Island']:
#     if delivery_area!= 'BrooklynQueens':
#         polygon = MultiPolygon(census_df.loc[census_df.BoroName==delivery_area,'geometry'].values)
#     else:
#         polygon = MultiPolygon(census_df.loc[(census_df.BoroName=='Brooklyn') | (census_df.BoroName=='Queens'),'geometry'].values)
    
#     # create dataframe with polygons created by the intersection of the circles
#     # dataframe contains no. of intersections
#     circle_df = createCirclesGrid(polygon, resolution, radius)
#     # save dataframe
#     circle_df.to_file(GH.machine + GH.folder + GH.network_folder + 'overlapping_circles_boroughs/overlapping_circles_' + str(delivery_area) +  '_radius_' + str(radius) + '_resolution_' + str(resolution) + '.shp')
#     # plot geodataframe and save it
#     fig, ax = plt.subplots(figsize =(25,25))
#     circle_df.plot(ax = ax, column = 'intersections', legend=True)
#     fig.savefig('figures/overlapping_circles_boroughs/overlapping_circles_' + str(delivery_area) + '_radius_' + str(radius) + '_resolution_' + str(resolution) + '.png', bbox_inches='tight')

#     all_census_df = pd.concat((all_census_df, circle_df), axis=0)
    
# all_census_df.to_file(GH.machine + GH.folder + GH.network_folder + 'overlapping_circles_boroughs/overlapping_circles_all_radius_' + str(radius) + '_resolution_' + str(resolution) + '.shp')
# # plot geodataframe and save it
# fig, ax = plt.subplots(figsize =(25,25))
# all_census_df.plot(ax = ax, column = 'intersections', legend=True)
# fig.savefig('figures/overlapping_circles_boroughs/overlapping_circles_all_radius_' + str(radius) + '_resolution_' + str(resolution) + '.png', bbox_inches='tight')

# #%%

# ## WHOLE NEW YORK CITY
# polygon = MultiPolygon(census_df['geometry'].values)
# # resolution = distance between each point on the grid
# resolution = 500
# # radius = of each circle drawn from each point of the grid
# radius = 5000
# # create dataframe with polygons created by the intersection of the circles
# # dataframe contains no. of intersections
# circle_df = createCirclesGrid(polygon, resolution, radius)
# # save dataframe
# circle_df.to_file(GH.machine + GH.folder + GH.network_folder + 'overlapping_circles_all/overlapping_circles_all_'+ str(radius) + '_resolution_' + str(resolution) + '.shp')
# # plot geodataframe and save it
# fig, ax = plt.subplots(figsize =(25,25))
# circle_df.plot(ax = ax, column = 'intersections', legend=True)
# fig.savefig('figures/overlapping_circles_all/overlapping_circles_all_radius_' + str(radius) + '_resolution_' + str(resolution) + '.png', bbox_inches='tight')

# #%%

# ## UNION OF DELIVERY AREAS (Manhattan, Brooklyn + Queens, Bronx, Staten Island) separately
# resolution = 500
# radius = 5000
# circle_df = gpd.read_file(GH.machine + GH.folder + GH.network_folder + 'overlapping_circles_boroughs/overlapping_circles_all_radius_'+ str(radius) + '_resolution_' + str(resolution) + '.shp')
# circle_df['area'] = circle_df.area
# # copy census_df
# census_df_ = census_df.copy()
# census_df_['intersections']=0
# for idx in census_df_.index:
#     CT = census_df.loc[idx,'geometry']
#     intersecting_regions = circle_df['geometry'].intersects(CT)
#     intersecions_CT = (intersecting_regions*circle_df['area']*circle_df['intersecti']).sum()/(intersecting_regions*circle_df['area']).sum()
#     census_df_.loc[idx, 'intersections'] = intersecions_CT
    
# census_df_.to_file(GH.machine + GH.folder + GH.network_folder + 'overlapping_circles_boroughs/census_tracts_'+ str(radius) + '_resolution_' + str(resolution) + '.shp')
# fig, ax = plt.subplots(figsize =(25,25))
# census_df_.plot(ax = ax, cmap = 'seismic', column = 'intersections', legend=True)
# fig.savefig('figures/overlapping_circles_boroughs/census_tracts_radius_' + str(radius) + '_resolution_' + str(resolution) + '.png', bbox_inches='tight')

# ## WHOLE NEW YORK CITY
# resolution = 500
# radius = 5000
# circle_df = gpd.read_file(GH.machine + GH.folder + GH.network_folder + 'overlapping_circles_all/overlapping_circles_all_'+ str(radius) + '_resolution_' + str(resolution) + '.shp')
# circle_df['area'] = circle_df.area
# # copy census_df
# census_df_ = census_df.copy()
# census_df_['intersections']=0
# for idx in census_df_.index:
#     CT = census_df.loc[idx,'geometry']
#     intersecting_regions = circle_df['geometry'].intersects(CT)
#     intersecions_CT = (intersecting_regions*circle_df['area']*circle_df['intersecti']).sum()/(intersecting_regions*circle_df['area']).sum()
#     census_df_.loc[idx, 'intersections'] = intersecions_CT
    
# census_df_.to_file(GH.machine + GH.folder + GH.network_folder + 'overlapping_circles_all/census_tracts_'+ str(radius) + '_resolution_' + str(resolution) + '.shp')
# fig, ax = plt.subplots(figsize =(25,25))
# census_df_.plot(ax = ax, cmap = 'seismic', column = 'intersections', legend=True)
# fig.savefig('figures/overlapping_circles_all/census_tracts_radius_' + str(radius) + '_resolution_' + str(resolution) + '.png', bbox_inches='tight')

# #%%
# GD = GeoData()
# euclidean_distances_corrected = pd.read_csv(GD.machine + GD.distances_folder + 'euclidean_distances_corrected.csv', index_col=[0])
# hausdorff_distances_corrected = pd.read_csv(GD.machine + GD.distances_folder + 'hausdorff_distances_corrected.csv', index_col=[0])
# hausdorff_distances_corrected.columns = hausdorff_distances_corrected.columns.astype(int)
# euclidean_distances_corrected.columns = euclidean_distances_corrected.columns.astype(int)

# times_CT = pd.DataFrame(CT_columns, columns = ['BoroCT2020']).merge(timetable,on='BoroCT2020', how='left')
# times_CT = times_CT.set_index('BoroCT2020')
# times_CT = times_CT.loc[CT_columns,CT_columns]
# #times_CT.values[[np.arange(times_CT.shape[0])]*2] = 0
# #dists_method = 'euclidean_polygon'
# #dists_method = 'hausdorff_polygon'
# #dists_method = 'euclidean_centroids'
# dists_method = 'grubhub_distances'
# #dists_method = 'grubhub_times'
# if dists_method == 'euclidean_polygon':
#     dists_CT = euclidean_distances_corrected.loc[CT_columns,CT_columns]
# elif dists_method == 'hausdorff_polygon':
#     dists_CT = hausdorff_distances_corrected.loc[CT_columns,CT_columns]
# elif dists_method == 'grubhub_distances':
#     dists_CT = pd.DataFrame(CT_columns, columns = ['BoroCT2020']).merge(distancetable,on='BoroCT2020', how='left')
#     dists_CT = dists_CT.set_index('BoroCT2020')
#     dists_CT = dists_CT.loc[CT_columns,CT_columns]
#     #dists_CT.values[[np.arange(dists_CT.shape[0])]*2] = 0
#     dists_CT = dists_CT*1000
#     dists_CT = dists_CT*1.60934
# else:
#     dists_CT = times_CT.copy()

# #dists_CT = dists_CT.fillna(0)
# dists_CT.sort_index(axis=0)
# dists_CT.sort_index(axis=1)
# dists_CT_sparse = csr_matrix(dists_CT.values)
        

# #%%

# if dists_method == 'euclidean_polygon':
#     R = np.array([0,5,500,1000,1500,2000,2500,3000,3500,4000,4500,5000])
# elif (dists_method == 'hausdorff_polygon'):
#     R = np.array([0,500,1000,1500,2000,2500,3000,3500,4000,4500,5000])
# elif (dists_method == 'grubhub_distances'):
#     R = np.array([0,500,1000,1500,2000,2500,3000,3500,4000,4500,5000,7500,10000,15000,20000])
# else:
#     R = np.array([0, 10, 20, 30, 40, 50, 60, 90, 120, 240, 480])

# dists_bands_CT = {}
# dists_circles_CT = {}

# for r in range(len(R)):
    
#     if r==0:
#         dists_bands_CT[R[r]] = (dists_CT_sparse>0) - (dists_CT_sparse>R[r])
#     else:
#         dists_bands_CT[R[r]] = (dists_CT_sparse>0) - (dists_CT_sparse>R[r]) - dists_bands_CT[R[r-1]]

# for r in range(len(R)):
    
#     if r==0:
#         dists_circles_CT[R[r]] = (dists_CT_sparse>0) - (dists_CT_sparse>R[r])
#     else:
#         dists_circles_CT[R[r]] = (dists_CT_sparse>0) - (dists_CT_sparse>R[r]) #- dists_bands_CT[R[r-1]]
        
#     dists_circles_CT[R[r]].setdiag(1)
    
# for r in range(len(R)):
    
#     # h_CT_R[R[r]] = reshape_(np.max(dists_bands_CT[R[r]].multiply(h_CT).todense(),axis=1))#/dists_bands_CT[R[r]].sum(1))
#     # p_CT_R[R[r]] = reshape_(np.max(dists_bands_CT[R[r]].multiply(p_CT).todense(),axis=1))#.sum(1))#/dists_bands_CT[R[r]].sum(1))
#     # hp_CT_R[R[r]] = reshape_(np.max(dists_bands_CT[R[r]].multiply(hp_CT).todense(),axis=1))#.sum(1))#/dists_bands_CT[R[r]].sum(1))
    
#     # h_del_CT_R[R[r]] = reshape_(np.max(dists_bands_CT[R[r]].multiply(h_del_CT).todense(),axis=1))#.sum(1))#/dists_bands_CT[R[r]].sum(1))
#     # p_del_CT_R[R[r]] = reshape_(np.max(dists_bands_CT[R[r]].multiply(p_del_CT).todense(),axis=1))#.sum(1))#/dists_bands_CT[R[r]].sum(1))
#     # hp_del_CT_R[R[r]] = reshape_(np.max(dists_bands_CT[R[r]].multiply(hp_del_CT).todense(),axis=1))#.sum(1))#/dists_bands_CT[R[r]].sum(1))

#     # num_physical_competitors_R[R[r]] = dists_bands_CT[R[r]].dot(reshape_(dummies_CT_sparse.sum(0)))#/reshape_(dists_bands_CT[R[r]].sum(1))
#     # num_delivery_competitors_R[R[r]] = dists_bands_CT[R[r]].dot(reshape_(dummies_del_sparse.sum(0)))#/reshape_(dists_bands_CT[R[r]].sum(1))
    
#     if r!=0:
#         h_CT_R[R[r]] = reshape_(dists_bands_CT[R[r]].multiply(h_CT).sum(1)/dists_bands_CT[R[r]].sum(1))
#         p_CT_R[R[r]] = reshape_(dists_bands_CT[R[r]].multiply(p_CT).sum(1)/dists_bands_CT[R[r]].sum(1))
#         hp_CT_R[R[r]] = reshape_(dists_bands_CT[R[r]].multiply(hp_CT).sum(1)/dists_bands_CT[R[r]].sum(1))
        
#         #h_del_CT_R[R[r]] = reshape_(dists_bands_CT[R[r]].multiply(h_del_CT).sum(1)/dists_bands_CT[R[r]].sum(1))
#         #p_del_CT_R[R[r]] = reshape_(dists_bands_CT[R[r]].multiply(p_del_CT).sum(1)/dists_bands_CT[R[r]].sum(1))
#         #hp_del_CT_R[R[r]] = reshape_(dists_bands_CT[R[r]].multiply(hp_del_CT).sum(1)/dists_bands_CT[R[r]].sum(1))
        
#         num_physical_delivery_competitors_R[R[r]] = 0

#     # h_CT_R[R[r]] = dists_bands_CT[R[r]].dot(h_CT_R[0])/reshape_(dists_bands_CT[R[r]].sum(1))#reshape_(dists_bands_CT[R[r]].multiply(h_CT).sum(1)/dists_bands_CT[R[r]].sum(1))
#     # p_CT_R[R[r]] = dists_bands_CT[R[r]].dot(p_CT_R[0])/reshape_(dists_bands_CT[R[r]].sum(1))#reshape_(dists_bands_CT[R[r]].multiply(p_CT).sum(1)/dists_bands_CT[R[r]].sum(1))
#     # hp_CT_R[R[r]] = dists_bands_CT[R[r]].dot(hp_CT_R[0])/reshape_(dists_bands_CT[R[r]].sum(1))#reshape_(dists_bands_CT[R[r]].multiply(hp_CT).sum(1)/dists_bands_CT[R[r]].sum(1))
    
#     # h_del_CT_R[R[r]] = dists_bands_CT[R[r]].dot(h_del_CT_R[0])/reshape_(dists_bands_CT[R[r]].sum(1))#reshape_(dists_bands_CT[R[r]].multiply(h_del_CT).sum(1)/dists_bands_CT[R[r]].sum(1))
#     # p_del_CT_R[R[r]] = dists_bands_CT[R[r]].dot(p_del_CT_R[0])/reshape_(dists_bands_CT[R[r]].sum(1))#reshape_(dists_bands_CT[R[r]].multiply(p_del_CT).sum(1)/dists_bands_CT[R[r]].sum(1))
#     # hp_del_CT_R[R[r]] = dists_bands_CT[R[r]].dot(hp_del_CT_R[0])/reshape_(dists_bands_CT[R[r]].sum(1))#reshape_(dists_bands_CT[R[r]].multiply(hp_del_CT).sum(1)/dists_bands_CT[R[r]].sum(1))

#         num_physical_competitors_R[R[r]] = dists_bands_CT[R[r]].dot(reshape_(dummies_CT_sparse.sum(0)))#/reshape_(dists_bands_CT[R[r]].sum(1))
#     #num_delivery_competitors_circle[R[r]] = dists_circles_CT[R[r]].dot(reshape_(dummies_del_sparse.sum(0)))#/reshape_(dists_bands_CT[R[r]].sum(1))
#         #num_delivery_competitors_R[R[r]] = dummies_del_sparse.T.dot(dummies_CT_sparse.dot(dists_bands_CT[R[r]])).diagonal()
#         num_delivery_competitors_R[R[r]] = reshape_(((dummies_CT_sparse.dot(dists_bands_CT[R[r]])).multiply(dummies_del_sparse)).sum(0))
        
        
#         den_physical_competitors_R[R[r]] = num_physical_competitors_R[R[r]]/dists_bands_CT[R[r]].dot(areas)#/reshape_(dists_bands_CT[R[r]].sum(1))
#     #num_delivery_competitors_circle[R[r]] = dists_circles_CT[R[r]].dot(reshape_(dummies_del_sparse.sum(0)))#/reshape_(dists_bands_CT[R[r]].sum(1))
#         den_delivery_competitors_R[R[r]] = num_delivery_competitors_R[R[r]]/dists_bands_CT[R[r]].dot(areas)
    
#     num_physical_competitors_circle[R[r]] = dists_circles_CT[R[r]].dot(reshape_(dummies_CT_sparse.sum(0)))#/reshape_(dists_bands_CT[R[r]].sum(1))
#     #num_delivery_competitors_R[R[r]] = dists_bands_CT[R[r]].dot(reshape_(dummies_del_sparse.sum(0)))#/reshape_(dists_bands_CT[R[r]].sum(1))
#     #num_delivery_competitors_circle[R[r]] = dummies_del_sparse.T.dot(dummies_CT_sparse.dot(dists_circles_CT[R[r]])).diagonal()                                  
#     num_delivery_competitors_circle[R[r]] = reshape_(((dummies_CT_sparse.dot(dists_circles_CT[R[r]])).multiply(dummies_del_sparse)).sum(0))    

    
#     den_physical_competitors_circle[R[r]] = num_physical_competitors_circle[R[r]]/dists_circles_CT[R[r]].dot(areas)
#     den_delivery_competitors_circle[R[r]] = num_delivery_competitors_circle[R[r]]/dists_circles_CT[R[r]].dot(areas)
    
# num_physical_competitors_circle[0] = num_physical_competitors_R[0]
# num_delivery_competitors_circle[0] = num_physical_delivery_competitors_R[0]

# den_physical_competitors_circle[0] = num_physical_competitors_circle[0]/areas
# den_delivery_competitors_circle[0] = num_delivery_competitors_circle[0]/areas

# #num_delivery_competitors_tot = reshape_(dummies_del_sparse.sum(0))
# num_delivery_competitors_tot = num_delivery_competitors_circle[R[-1]]
    
# #%%


# data = {'BoroCT2020': CT_columns,
#         'num_delivery_competitors_tot': num_delivery_competitors_tot,
#         'num_delivery_competitors_tot_norm': num_delivery_competitors_tot/100,
#         'log_num_delivery_competitors_tot': np.log(num_delivery_competitors_tot)}#
#         # 'disparity_rel_0': h_rel,
#         # 'balance_rel_0': p_rel,
#         # 'diversity_rel_0': hp_rel,
#         # 'log_disparity_rel_0': np.log(h_rel),
#         # 'log_balance_rel_0': np.log(p_rel),
#         # 'log_diversity_rel_0': np.log(hp_rel)}



# if dists_method == 'euclidean_polygon':
#     R = np.array([0,5,500,1000,1500,2000,2500,3000,3500,4000,4500,5000])
# elif (dists_method == 'hausdorff_polygon'):
#     R = np.array([0,500,1000,1500,2000,2500,3000,3500,4000,4500,5000])
# elif (dists_method == 'grubhub_distances'):
#     R = np.array([0,500,1000,1500,2000,2500,3000,3500,4000,4500,5000,7500,10000,15000,20000])
# else:
#     R = np.array([0, 10, 20, 30, 40, 50, 60, 90, 120,240,480])

# for r in R:
#     data = data | {'num_physical_competitors_'+str(r): num_physical_competitors_R[r],
#                    'num_physical_competitors_norm_'+str(r): num_physical_competitors_R[r]/100,
#                    'den_physical_competitors_'+str(r): den_physical_competitors_R[r],
#                    'num_delivery_competitors_'+str(r): num_delivery_competitors_R[r],
#                    'num_delivery_competitors_norm_'+str(r): num_delivery_competitors_R[r]/100,
#                    'den_delivery_competitors_'+str(r): den_delivery_competitors_R[r],
#                    'num_physical_delivery_competitors_'+str(r): num_physical_delivery_competitors_R[r],
#                    'ratio_physical_delivery_competitors_'+str(r): num_physical_delivery_competitors_R[r]/num_physical_competitors_R[r]}
#     data = data | {'num_physical_competitors_tot_'+str(r): num_physical_competitors_circle[r],
#                    'num_physical_competitors_tot_norm_'+str(r): num_physical_competitors_circle[r]/100,
#                    'den_physical_competitors_tot_'+str(r): den_physical_competitors_circle[r],
#                    'num_delivery_competitors_tot_'+str(r): num_delivery_competitors_circle[r],
#                    'num_delivery_competitors_tot_norm_'+str(r): num_delivery_competitors_circle[r]/100,
#                    'den_delivery_competitors_tot_'+str(r): den_delivery_competitors_circle[r],}
#     data = data | {'disparity_'+str(r): h_CT_R[r], 'balance_'+str(r): p_CT_R[r], 'diversity_'+str(r): hp_CT_R[r]}
#     #data = data | {'disparity_del_'+str(r): h_del_CT_R[r], 'balance_del_'+str(r): p_del_CT_R[r], 'diversity_del_'+str(r): hp_del_CT_R[r]}
#     data['difference_delivery_competitors_' + str(r)] = data['num_delivery_competitors_tot']-data['num_delivery_competitors_tot_'+str(r)]
#     data['difference_delivery_competitors_norm' + str(r)] = data['num_delivery_competitors_tot_norm']-data['num_delivery_competitors_tot_norm_'+str(r)]
#     data['log_num_physical_competitors_'+str(r)] = np.log(data['num_physical_competitors_'+str(r)]+1e-6)
#     data['log_num_physical_competitors_tot_'+str(r)] = np.log(data['num_physical_competitors_tot_'+str(r)]+1e-6)
#     data['log_num_delivery_competitors_'+str(r)] = np.log(data['num_delivery_competitors_'+str(r)]+1e-6)
#     ratios = data['num_delivery_competitors_tot_'+str(r)]/data['num_physical_competitors_tot_'+str(r)]
#     ratios_tot = data['num_delivery_competitors_tot']/data['num_physical_competitors_tot_'+str(r)]
#     data['ratio_competitors_'+str(r)] = np.where(ratios!=np.float('inf'), ratios,0)
#     data['ratio_competitors_tot_'+str(r)] = np.where(ratios_tot!=np.float('inf'), ratios_tot,0)
#     logs=np.log(data['num_delivery_competitors_tot']/data['num_physical_competitors_tot_'+str(r)]+1e-6)
#     data['log_ratio_competitors_'+str(r)] = np.where(logs!=np.float('inf'), logs,0)
#     data['log_disparity_'+str(r)] = -np.log(data['disparity_'+str(r)])
#     data['log_balance_'+str(r)] = -np.log(data['balance_'+str(r)])
#     data['log_diversity_'+str(r)] = -np.log(data['diversity_'+str(r)])
#     #data['log_disparity_del_'+str(r)] = np.log(data['disparity_del_'+str(r)])
#     #data['log_balance_del_'+str(r)] = np.log(data['balance_del_'+str(r)])
#     #data['log_diversity_del_'+str(r)] = np.log(data['diversity_del_'+str(r)]+1e-6)
# # necessary for 
# df = pd.DataFrame(data)
# #df.pop('BoroCT2020')
# #df=df.reset_index()
# df['BoroCT2020'] = df['BoroCT2020'].astype(int)
# df=df.set_index('BoroCT2020')

# df =census_df[['GEOID','BoroName', 'NTA2020', 'Shape_Area', 'geometry']].merge(df, on = 'BoroCT2020', how = 'left')
# for r in R:
#     df['density_physical_competitors_'+str(r)] = df['num_physical_competitors_'+str(r)]/df['Shape_Area']*10e6
#     df['interaction_density_difference_'+str(r)] = df['difference_delivery_competitors_' + str(r)]*df['density_physical_competitors_'+str(0)] 
    

# df['ratio_delivery_physical'] = df['num_delivery_competitors_0']/df['num_physical_competitors_0']
# df['area'] = df['Shape_Area']/1e6
# #df['interaction_density_ratio'] =  df['density_physical_competitors']*df['ratio_delivery_physical']


# #%%

# #del delivering_matrix,  filtering_matrix_CT_sparse, h_matrix, h_del_matrix, hp_matrix, hp_del_matrix, p_del_matrix, pi_matrix

# #%%
# missing_kwds = dict(color='lightgrey', hatch = '///')

# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "sans-serif",
#     "font.sans-serif": "Helvetica",
# })

# fig, ax = plt.subplots(figsize = (25,25))
# df.plot('den_physical_competitors_0', ax=ax, cmap ='inferno_r', missing_kwds=missing_kwds, legend=True, legend_kwds={'shrink': 0.4})
# fig.savefig('figures/num_physical_competitors_0.png', bbox_inches='tight')

# fig, ax = plt.subplots(figsize = (25,25))
# df.plot('num_physical_delivery_competitors_0', ax=ax, cmap ='inferno_r', missing_kwds=missing_kwds, legend=True, legend_kwds={'shrink': 0.4})
# fig.savefig('figures/num_physical_delivery_competitors_0.png', bbox_inches='tight')

# fig, ax = plt.subplots(figsize =(25,25))
# df.plot('num_delivery_competitors_tot', ax=ax, cmap ='RdYlBu_r', missing_kwds=missing_kwds, legend=True, legend_kwds={'shrink': 0.4})
# fig.savefig('figures/num_delivery_competitors_tot.png', bbox_inches='tight')

# #%%

# #%%
# df_=  df.copy()
# divs = ['log_disparity_0', 'log_balance_0', 'log_diversity_0']
# df_.loc[df.num_physical_competitors_0<=1, divs] = np.nan
# for div in divs:
#     fig, ax = plt.subplots(figsize = (25,25))    
#     df_.plot(div, legend=True, missing_kwds=missing_kwds, cmap ='RdYlBu_r', ax=ax, legend_kwds={'shrink': 0.4})
#     fig.savefig('figures/'+div+'.png', bbox_inches='tight')

# #%%

# df_=  df.copy()

# for r in R:
    
#     df_.loc[df.num_delivery_competitors_0<=0, 'difference_delivery_competitors_'+str(r)] = np.nan
#     df_.loc[df.num_delivery_competitors_0<=0, 'num_delivery_competitors_tot_'+str(r)] = np.nan
#     df_.loc[df.num_delivery_competitors_0<=0, 'num_physical_competitors_tot_'+str(r)] = np.nan

# for r in R:
#     fig, ax = plt.subplots(figsize = (25,25))    
#     df_.plot('difference_delivery_competitors_'+str(r), legend=True, missing_kwds=missing_kwds, cmap = plt.cm.RdYlBu_r, ax=ax, legend_kwds={'shrink': 0.4})
#     fig.savefig('figures/'+dists_method+'_difference_delivery_competitors_'+str(r)+'.png', bbox_inches='tight')

#     fig, ax = plt.subplots(figsize = (25,25))    
#     df_.plot('num_delivery_competitors_tot_'+str(r), legend=True, missing_kwds=missing_kwds, cmap = plt.cm.RdYlBu_r, ax=ax, legend_kwds={'shrink': 0.4})
#     fig.savefig('figures/'+dists_method+'_num_delivery_competitors_tot_'+str(r)+'.png', bbox_inches='tight')
    
#     fig, ax = plt.subplots(figsize = (25,25))    
#     df_.plot('num_physical_competitors_tot_'+str(r), legend=True, missing_kwds=missing_kwds, cmap = plt.cm.RdYlBu_r, ax=ax, legend_kwds={'shrink': 0.4})
#     fig.savefig('figures/'+dists_method+'_num_physical_competitors_tot_'+str(r)+'.png', bbox_inches='tight')
    

# #%%

# plt.figure(0)
# mask1=(np.isnan(df.log_disparity_0)) | (df.num_physical_competitors_0<=1)
# y=df.log_disparity_0[~mask1]
# x=df.log_balance_0[~mask1]
# plt.plot(x, y, '.', color='red')
# m, b = np.polyfit(x, y, 1)
# plt.plot(x, m*x+b, color='blue')
# plt.xlabel('Log-balance')
# plt.ylabel('Log-disparity')
# plt.savefig('figures/corr_balance_disparity.png', dpi=300)

      
# #%%

# threshold=1
# data_cut=df[df.num_physical_competitors_0>threshold]
# data_cut=data_cut[data_cut.BoroName=='Queens']
# divs = ['disparity']#, 'balance', 'diversity']
# for div in divs:
#     #fe = 'BoroName'
#     formula = "log_" + div + "_0 ~ "
    
#     ests = []
    
#     formula = "log_" + div + "_0 ~ "
#     #formula =  "den_physical_competitors_0 ~"# + str(r)
#     #m3 = sm.ols(formula, data_cut).fit()
#     #ests.append(m3)
    
#     for r in R[:-1]:
        
#         formula = "log_" + div + "_0 ~ "
#         #formula +=  " + den_physical_competitors_0"# + str(r)
#         formula +=  " + den_physical_competitors_tot_" + str(r)
#         formula +=  " + num_delivery_competitors_tot_norm_" + str(r)
#         #formula +=  " + interaction_density_difference_" + str(r)
        
#         m3 = sm.ols(formula, data_cut).fit()
#         ests.append(m3)
#         print(m3.summary())
        
# #%%    
#     star = Stargazer(ests)
#     star.dependent_variable_name('Log-' + div)
#     rename_covs = {}
#     rename_covs['num_delivery_competitors_tot_norm'] = 'Num. competitors delivering \\\ to census tract'
#     for r in R[:6]:
#         if r == 0:
#             rename_covs['den_physical_competitors_' + str(r)]= 'Den. competitors \\\ in census tract'
#         elif ((dists_method == 'euclidean_polygon') & (r == 5)):
#             rename_covs['den_physical_competitors_' + str(r)]= 'Num. competitors  \\\ in neighboring tracts'
#         else:
#             rename_covs['den_physical_competitors_' + str(r)] = 'Num. competitors \\\ within ' + str(r/1000) + ' km'
    
#     star.covariate_order(rename_covs.keys())
#     star.rename_covariates(rename_covs)
#     star.show_adj_r2 = False
#     star.show_f_statistic = False
#     star.show_residual_std_err=False
#     star.table_label='tab:OLS_' + div + '_delivering'
#     file_name = 'tables/' + div + '_delivery_physical_competitors.tex'
#     #tex_file = open(file_name, 'w')
#     #tex_file.write(star.render_latex())
#     #tex_file.close()
# #%%

# census_data = pd.read_csv(machine + 'census_tract_data/ACS_data.csv')
# census_data['estimate_population'] /= 1000
# census_data['estimate_median_income'] /= 1000
# non_white_frac = 1-census_data['frac_white']
# races = ['white', 'black', 'american_indian', 'asian', 'pacific', 'other_one', 'other_two']
# hhi_all_races = 0
# hhi_non_white = 0
# for race in races:
#     hhi_all_races += census_data['frac_'+race]**2
#     hhi_non_white += (census_data['frac_'+race]/non_white_frac)**2
# census_data['hhi_all_races'] = hhi_all_races
# census_data['hhi_non_white'] = hhi_non_white
# df = df.reset_index()
# df1 = df.merge(census_data, on ='GEOID', how='left')
# census_zoning = pd.read_csv(machine + 'nyc_geodata/zoning_laws/census_w_districts.csv')
# df1 = df1.merge(census_zoning[['BoroCT2020', 'Major_district']], on = 'BoroCT2020')
# df1 = pd.get_dummies(df1, columns=['Major_district'])
# rwac_CT = pd.read_csv(machine + 'census_tract_data/rwac_CT.csv', index_col=[0])
# df1 = df1.merge(rwac_CT[['BoroCT2020', 'C000_r', 'CE01_r', 'CE02_r', 'CE03_r', 'C000_w', 'CE01_w', 'CE02_w', 'CE03_w']], on = 'BoroCT2020')
# df1[['C000_r', 'CE01_r', 'CE02_r', 'CE03_r', 'C000_w', 'CE01_w', 'CE02_w', 'CE03_w']] = df1[['C000_r', 'CE01_r', 'CE02_r', 'CE03_r', 'C000_w', 'CE01_w', 'CE02_w', 'CE03_w']]/1000
# df1['density_population'] = df1['estimate_population']/df1['Shape_Area']*10e6
# df1['density_workers'] = df1['C000_w']/df1['Shape_Area']*10e6
# for r in R:
#     df1['interaction_income_difference_'+str(r)] = df1['estimate_median_income']*df1['density_physical_competitors_'+str(0)]/1e3
    
# #%%


# rents = pd.read_csv('nyc_geodata/rents/Zip_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv')
# rents=rents.rename({'RegionName':'zip', '2022-10-31': 'ZHVI'}, axis='columns')
# rents = rents.loc[(rents.City == 'New York')][['zip', 'ZHVI']]

# tract_zips = pd.read_excel(machine + 'census_tract_data/TRACT_ZIP_122021.xlsx')
# tract_zips = tract_zips.loc[tract_zips['usps_zip_pref_state']=='NY']
# tract_zips=tract_zips.rename({'tract':'GEOID'}, axis='columns')

# # df1 = df1.reset_index()
# # df3=df1.merge(tract_zips[['GEOID','zip','tot_ratio']].merge(rents, on = 'zip', how='left'), on = 'GEOID', how='left')
# # df3['ZHVI'] = df3['ZHVI']*df3['tot_ratio']
# # df3 = df3.groupby(['BoroCT2020']).sum()
# # df3 = df3.reset_index()
# # df3 = df3.set_index('BoroCT2020')
# # df1= df1.set_index('BoroCT2020')
# # df3=df3.sort_index(0)
# # df1=df1.sort_index(0)

# # df3['log_disparity_0']=df1['log_disparity_0']

# temp=df1[['BoroCT2020','GEOID']].merge(tract_zips[['GEOID','zip','tot_ratio']].merge(rents, on = 'zip', how='left'), on = 'GEOID', how='left')
# temp['ZHVI'] = temp['ZHVI']*temp['tot_ratio']
# temp = temp.groupby(['BoroCT2020','GEOID']).sum()
# temp = temp.reset_index()

# df1 =df1.merge(temp[['ZHVI', 'BoroCT2020']], how = 'left', on = 'BoroCT2020')
# #df1=df1.reset_index()
# df1['ZHVI'] /= 1e6
# for r in R:
#     df1['interaction_ZHVI_difference_'+str(r)] = df1['ZHVI']*df1['density_physical_competitors_'+str(0)] 


# #%%

# threshold=1
# data_cut=df1[df1.num_physical_competitors_0>threshold]

# ests = []

# for dependent_variable in ['den_physical_competitors_0', 'log_disparity_0', 'log_balance_0', 'log_diversity_0']:
    
#     formula = dependent_variable + ' ~'
    
#     formula += '+ density_population'
#     formula += '+ estimate_median_income'
#     formula += '+ frac_white'
#     formula += '+ hhi_all_races'
#     formula += '+ Major_district_C'
#     formula += '+ Major_district_R'
#     formula += '+  density_workers'
#     formula += '+ ZHVI'
#     m3 = sm.ols(formula, data_cut).fit()
#     ests.append(m3)
# star = Stargazer(ests)
# rename_covs = {}
# rename_covs['density_population'] = 'Density of population (1000/$km^2$)'
# rename_covs['estimate_median_income'] = 'Median income (1000 \$)'
# rename_covs['frac_white'] = 'Fraction of white residents'
# rename_covs['hhi_all_races'] = 'Race HHI'
# rename_covs['ZHVI'] = 'Home value index'
# rename_covs['density_workers'] = 'Density of workers (1000/$km^2$)'
# rename_covs['Major_district_C'] = 'Commercial district'
# rename_covs['Major_district_R'] = 'Residential district'
# #star.dependent_variable_name('Density of restaurants in $z$')
# custom_columns = ['Density of restaurants', 'Log-disparity', 'Log-balance', 'Log-diversity']
# star.custom_columns(custom_columns, [1, 1,1,1])
# star.covariate_order(rename_covs.keys())
# star.rename_covariates(rename_covs)
# star.significant_digits(4)
# star.show_adj_r2 = False
# star.show_f_statistic = False
# star.show_residual_std_err=False
# star.table_label='tab:OLS_covariates'
# file_name = 'tables/all_covariates.tex'
# tex_file = open(file_name, 'w')
# tex_file.write(star.render_latex())
# tex_file.close()
# print(m3.summary())

# #%%

# threshold=1
# data_cut=df1[df1.num_physical_competitors_0>threshold]

# ests = []

# formula = 'den_physical_competitors_0 ~'
# formula += '+ density_population'
# formula += '+ estimate_median_income'
# formula += '+ frac_white'
# formula += '+ hhi_all_races'
# formula += '+ Major_district_C'
# formula += '+ Major_district_R'
# formula += '+  density_workers'
# formula += '+ ZHVI'
# m3 = sm.ols(formula, data_cut).fit()
# ests.append(m3)
# star = Stargazer(ests)
# rename_covs = {}
# rename_covs['density_population'] = 'Density of population (1000/$km^2$)'
# rename_covs['estimate_median_income'] = 'Median income (1000 \$)'
# rename_covs['frac_white'] = 'Fraction of white residents'
# rename_covs['hhi_all_races'] = 'Race HHI'
# rename_covs['ZHVI'] = 'Home value index'
# rename_covs['density_workers'] = 'Density of workers (1000/$km^2$)'
# rename_covs['Major_district_C'] = 'Commercial district'
# rename_covs['Major_district_R'] = 'Residential district'
# star.dependent_variable_name('Density of restaurants in $z$')
# star.covariate_order(rename_covs.keys())
# star.rename_covariates(rename_covs)
# star.significant_digits(4)
# star.show_adj_r2 = False
# star.show_f_statistic = False
# star.show_residual_std_err=False
# star.table_label='tab:OLS_covariates'
# file_name = 'tables/number_restaurants_covariates.tex'
# tex_file = open(file_name, 'w')
# tex_file.write(star.render_latex())
# tex_file.close()
# print(m3.summary())

# #%%

# threshold=1
# data_cut=df1[df1.num_physical_competitors_0>threshold]
# div = 'disparity'
# #div = 'disparity_rel'
# ests = []

# formula = "log_" + div + "_0 ~ "

# formula += '+ density_population'
# formula += '+ estimate_median_income'
# formula += '+ Major_district_C'
# formula += '+ Major_district_R'
# formula += '+ frac_white'
# formula += '+ hhi_all_races'
# formula += '+  density_workers'
# formula += '+ ZHVI'
# #formula += '+ densityXZHVI'
# m3 = sm.ols(formula, data_cut).fit()
# ests.append(m3)
# star = Stargazer(ests)
# rename_covs = {}
# rename_covs['density_population'] = 'Density of population (1000/$km^2$)'
# rename_covs['estimate_median_income'] = 'Median income (1000 \$)'
# rename_covs['frac_white'] = 'Fraction of white residents'
# rename_covs['hhi_all_races'] = 'Race HHI'
# rename_covs['ZHVI'] = 'Home value index'
# rename_covs['density_workers'] = 'Density of workers (1000/$km^2$)'
# rename_covs['Major_district_C'] = 'Commercial district'
# rename_covs['Major_district_R'] = 'Residential district'
# star.dependent_variable_name('Log-disparity of $z$')
# star.covariate_order(rename_covs.keys())
# star.rename_covariates(rename_covs)
# star.significant_digits(4)
# star.show_adj_r2 = False
# star.show_f_statistic = False
# star.show_residual_std_err=False
# star.table_label='tab:OLS_disparity_covariates'
# file_name = 'tables/disparity_covariates.tex'
# tex_file = open(file_name, 'w')
# tex_file.write(star.render_latex())
# tex_file.close()
# print(m3.summary())

# #%%

            
# #%%
# neighboring_CTs = dists_CT.copy()
# neighboring_CTs[neighboring_CTs>1000]=np.nan
# neighboring_CTs[np.isnan(neighboring_CTs)==False]=1

# exponential_distances = neighboring_CTs.multiply(dists_CT)
# exponential_distances[exponential_distances==0] = np.nan

# exponential_distances = np.exp(-1/1000*exponential_distances)
# exponential_distances[np.isnan(exponential_distances)] = 0
# exponential_distances =  exponential_distances.div(exponential_distances.sum(axis=1), axis=0)


# # exponential_distances[np.isnan(exponential_distances)] = 0
# # exponential_distances[exponential_distances>0] = 1
# # exponential_distances =  exponential_distances.div(exponential_distances.sum(axis=1), axis=0)

# #%%

# variables = ['num_physical_competitors_tot_norm_0', 'den_physical_competitors_0',
#              'estimate_population', 'estimate_median_income', 'density_population',
#              'frac_white', 'frac_black', 'frac_asian', 'frac_other_one', 'frac_other_two',
#              'hhi_all_races',
#              'Major_district_C', 'Major_district_R', 'Major_district_M',
#              'density_workers',
#              'C000_r', 'CE01_r', 'CE02_r', 'CE03_r', 'C000_w', 'CE01_w', 'CE02_w', 'CE03_w',
#              'ZHVI']
# df1 = df1.set_index('BoroCT2020')
# original_vars = df1.loc[CT_columns,variables].fillna(0)
# original_vars = original_vars.sort_index(axis=0)
# exponential_distances = exponential_distances.sort_index(axis=1)
# lagged_vars = exponential_distances.dot(original_vars)

# df2 = df1.join(lagged_vars, on = 'BoroCT2020', rsuffix='_lagged')


# #%%

# # threshold=1
# # df3 = df2[df2.num_physical_competitors_0>threshold].copy()
# # df3['density_1'] = df3.den_physical_competitors_0<=np.nanquantile(df3.den_physical_competitors_0,0.25)
# # df3['density_2'] = (df3.den_physical_competitors_0>np.nanquantile(df3.den_physical_competitors_0,0.25)) & (df3.den_physical_competitors_0<=np.nanquantile(df3.den_physical_competitors_0,0.50))
# # df3['density_3'] = (df3.den_physical_competitors_0>np.nanquantile(df3.den_physical_competitors_0,0.50)) & (df3.den_physical_competitors_0<=np.nanquantile(df3.den_physical_competitors_0,0.75))
# # df3['density_4'] = df3.den_physical_competitors_0>np.nanquantile(df3.den_physical_competitors_0,0.75)


# # density_dummies=df3[['density_1', 'density_2', 'density_3','density_4']].stack()
# # df3 = df3.merge(pd.DataFrame(density_dummies[density_dummies!=0]).reset_index(), on = 'BoroCT2020')
# # #df3=df3.rename({'level_1':'density'})
# #%%

# threshold=1
# data_cut=df2[df2.num_physical_competitors_0>threshold]
# data_cut=data_cut[data_cut.BoroName=='Brooklyn']
# #data_cut=data_cut[data_cut.density_3 | data_cut.density_4]

# div = 'balance'
# variables = ['den_physical_competitors_0',
#               'density_population',
#               'estimate_median_income',
#               'frac_white',
#               'hhi_all_races',
#               #'frac_black',
#               #'frac_asian',
#               #'frac_other_one', 
#               #'frac_other_two',
#               'Major_district_C',
#               'Major_district_R',
#               #'Major_district_M',
#               'density_workers',
#               'ZHVI',
#               #'density_1',
#               #'density_2',
#               #'density_3'
#               ]

# formula =  'log_' + div + "_0 ~ "

# for r in R:
#     #formula = 'den_physical_competitors_0 ~'
#     formula =  'log_' + div + "_0 ~ "
#     #formula += " + den_physical_competitors_0"
#     formula += " + num_delivery_competitors_tot_norm_" + str(r)
#     #formula += " + interaction_income_difference_" + str(r)
#     #formula += " + interaction_ZHVI_difference_" + str(r)
#     #formula += " + num_delivery_competitors_tot_norm"# + str(r)
#     for var in variables:
#         formula += '+ ' + var
#         #formula += '+ ' + var + '_lagged'
#     #formula += '+ BoroName'
    
#     rename_covs['difference_delivery_competitors_' + str(r)] = 'Num. of delivering restaurants \\\ outside ' + str(r/1000) + ' km'
#     m3 = sm.ols(formula, data_cut).fit()
#     print(m3.summary())

#     #%%

# census_specific_data = pd.read_csv(machine + 'grubhub_data/analysis/census-specific-data.csv')
# census_specific_data.rename({'census_tract':'BoroCT2020'}, axis=1,inplace=True)
# median_receiving = census_specific_data.groupby('BoroCT2020').median()[['distance_from_location', 'delivery_fee', 'delivery_time_estimate']]
# median_departing = census_specific_data[['restaurant_id','distance_from_location', 'delivery_fee', 'delivery_time_estimate']].merge(merged_df[['restaurant_id', 'BoroCT2020']], on = 'restaurant_id').groupby('BoroCT2020').median()[['distance_from_location', 'delivery_fee', 'delivery_time_estimate']]

# cc=census_df.join(median_receiving, on = 'BoroCT2020', rsuffix='_rec')
# median_departing = median_departing.reset_index()
# median_departing.BoroCT2020 = median_departing.BoroCT2020.astype(int)
# cc =cc.merge(median_departing, how='left', on = 'BoroCT2020', suffixes=('_rec','_dep'))

# df2 = df2.merge(cc[['BoroCT2020','distance_from_location_rec', 'delivery_fee_rec', 'delivery_time_estimate_rec']], on = 'BoroCT2020', how='left')

# #%%

# fig, ax = plt.subplots(figsize = (20,20))
# cc.plot('distance_from_location_dep', cmap ='plasma', ax=ax,missing_kwds=missing_kwds, legend=True, legend_kwds={'shrink': 0.4})


# fig, ax = plt.subplots(figsize = (20,20))
# cc.plot('distance_from_location_rec', cmap ='plasma', ax=ax,missing_kwds=missing_kwds, legend=True, legend_kwds={'shrink': 0.4})

# #%%
# threshold=1
# data_short=df2[df2.num_physical_competitors_0>threshold]

# params_boro={}
# for boro in ['ALL', 'Manhattan', 'Brooklyn', 'Queens', 'Bronx', 'Staten Island']:
    
#     print(boro)
#     data_short_ = data_short.copy()
#     if boro != 'ALL':
#         data_short_ = data_short[data_short.BoroName==boro]
    
#     params_div={}
#     for div in ['disparity', 'balance', 'diversity']:
        
#         variables = ['den_physical_competitors_0',
#                      'density_population',
#                       'estimate_median_income',
#                       'frac_white',
#                       'hhi_all_races',
#                       'Major_district_C',
#                       'Major_district_R',
#                       'density_workers',
#                       'ZHVI',
#                      ]
    
#         varrs = ['difference_delivery_competitors_']#, 'interaction_income_difference_']
    
#         params_var = {}
#         for varr in varrs:
            
#             params_r = {}
#             for r in R:
#                 formula =  'log_' + div + "_0 ~ "
#                 formula += " + difference_delivery_competitors_" + str(r)
#                 #formula += ' + den_physical_competitors_' + str(r)
#                 #formula += " + interaction_income_difference_" + str(r)
#                 for var in variables:
#                     formula += '+ ' + var
#                     m3 = sm.ols(formula, data_short_).fit()
#                     with open(machine + 'results/summary_'+boro+'_'+div+'_'+str(r)+'.txt', 'w') as fh:
#                         fh.write(m3.summary().as_text())
#                     params_r[varr+str(r)] = [m3.params[varr + str(r)]] + list(m3.conf_int().loc[varr + str(r),:])          
            
#             params_var[varr] = params_r
#         params_div[div] = params_var
    
#     params_boro[boro] = params_div
    
# #%%


# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "sans-serif",
#     "font.sans-serif": "Helvetica",
# })

# i=0
# for boro in ['ALL', 'Manhattan', 'Brooklyn', 'Queens', 'Bronx', 'Staten Island']:

#     for div in ['disparity', 'balance', 'diversity']:
        
#         varrs = ['difference_delivery_competitors_']#, 'interaction_income_difference_']#'interaction_density_difference_']
        
        
#         plt.figure(i)
            
#         for varr in varrs:
            
#             x=R[:-4]
#             color = "black"
#             if varr == 'interaction_income_difference_':
#                 x=x+200
#                 color = "grey"
               
                
#             params_df = pd.DataFrame(params_boro[boro][div][varr]).T
            
#             plt.errorbar(x, y=params_df.iloc[:-4,0], yerr=params_df.iloc[:-4,2]-params_df.iloc[:-4,1], color=color, capsize=3,
#                          linestyle="None",
#                          marker="s", markersize=5, mfc=color, mec=color)
#             plt.title('Estimation of ' + div.capitalize() + ' in ' + boro)
#             plt.ylabel(r"OLS estimate $\beta_r$")
#             #plt.legend(['a','b'])
#             plt.savefig('figures/OLS_params_'+boro+'_'+div+'.png', dpi=300)
#         i+=1
           
# #%%
# threshold=1
# data_short=df2[df2.num_physical_competitors_0>threshold]

# params_boro={}
# for boro in ['ALL', 'Manhattan', 'Brooklyn', 'Queens', 'Bronx', 'Staten Island']:
    
#     print(boro)
#     data_short_ = data_short.copy()
#     if boro != 'ALL':
#         data_short_ = data_short[data_short.BoroName==boro]
    
#     params_div={}
#     for div in ['disparity', 'balance', 'diversity']:
        
#         variables = ['den_physical_competitors_0',
#                      'density_population',
#                       'estimate_median_income',
#                       'frac_white',
#                       'hhi_all_races',
#                       'Major_district_C',
#                       'Major_district_R',
#                       'density_workers',
#                       'ZHVI',
#                      ]
    
#         varrs = ['difference_delivery_competitors_']#, 'interaction_income_difference_']
    
#         params_var = {}
#         for varr in varrs:
            
#             params_r = {}
#             for r in R:
#                 formula =  'log_' + div + "_0 ~ "
#                 formula += " + difference_delivery_competitors_" + str(r)
#                 #formula += ' + den_physical_competitors_' + str(r)
#                 #formula += " + interaction_income_difference_" + str(r)
#                 for var in variables:
#                     formula += '+ ' + var
#                     m3 = sm.ols(formula, data_short_).fit()
#                     with open(machine + 'results/summary_'+boro+'_'+div+'_'+str(r)+'.txt', 'w') as fh:
#                         fh.write(m3.summary().as_text())
#                     params_r[varr+str(r)] = [m3.params[varr + str(r)]] + list(m3.conf_int().loc[varr + str(r),:])          
            
#             params_var[varr] = params_r
#         params_div[div] = params_var
    
#     params_boro[boro] = params_div
    
    
# #%%

# fig, ax = plt.subplots(figsize = (20,20))
# gpd.GeoDataFrame(df2.reset_index().join(pd.DataFrame(m3.resid, columns=['res']))).plot('res', legend=True, ax=ax)

# #%%

# fig, ax = plt.subplots(figsize = (20,20))
# gpd.GeoDataFrame(df2.reset_index().join(pd.DataFrame(m3.resid, columns=['res']))).plot('res', legend=True, ax=ax)


# #%%

# fig, ax = plt.subplots(figsize = (20,20))
# gpd.GeoDataFrame(df2.reset_index().join(pd.DataFrame(m3.resid, columns=['res']))).plot('res', legend=True, ax=ax)

# #%%

