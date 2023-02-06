#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 11:27:16 2022

@author: michelev
"""

import pandas as pd
import geopandas as gpd
import numpy as np

from datetime import date

from grubhub1 import GrubhubClient
from geo import GeoData

from concurrent.futures import ThreadPoolExecutor

from glob import glob
import os.path

import re

#from yelp import YelpClient

import matplotlib.pyplot as plt

import seaborn as sns

from fuzzywuzzy import fuzz
from Levenshtein import distance as levenshtein_distance

from scipy.spatial.distance import pdist, squareform
from Levenshtein import ratio

import ast
#%%

# In this class I create csv files of:
# - restaurants delivering to each census tract

class RetrieveData:
    
    # folders used for reading the documents
    
    machine = '/Users/michelev/spatial-competition-food/'
    geo_folder = 'nyc_geodata/'
    geo_subfolder = 'census_tracts_boundaries/'
    
    platform_folder = 'grubhub_data/'
    platform_census_subfolder = 'census_tracts/'
    platform_restaurants_subfolder = 'restaurants/'
    locations = None
    
    # read file of census tract centroids, store it in locations (keep only number and point)
    def setCityLocations(self):
        
        # retrieve dataframe with centroids of each census tract
        locations_df = gpd.read_file(self.machine + self.geo_folder + self.geo_subfolder + 'census_tracts_centroids.shp')
        # select just the label of the census tract (boro number + CT number) and the centroid (shapely Point)
        locations_pairs = locations_df.loc[:,['BoroCT2020', 'geometry']]
        
        self.locations = locations_pairs
    
    # find restaurants delivering to centroid corresponding to 'row' of the file
    def retrieveGrubHubByCT(self, row, date):
        
        if not os.path.exists(self.machine + self.platform_folder + self.platform_census_subfolder + date):
            os.makedirs(self.machine + self.platform_folder + self.platform_census_subfolder + date)
        
        # select (label, point)
        locations_pair = self.locations.iloc[row,:]
        # create GH object
        GH = GrubhubClient()
        GH.setPoint(locations_pair['geometry'])
        filename=str(locations_pair['BoroCT2020']) + '_' + date
        if len(glob(self.machine + self.platform_folder + self.platform_census_subfolder + date + '/' + str(locations_pair['BoroCT2020'])+'*'))>0:
            print("File already exists. Going to the next iteration.")
        else:
            GH.setFileName(filename)
            # search and save csv file
            GH.searchByPoint()
            GH.createDataFrame(date)

    
    def matchYelpGH():
        
        # Yelp dataset
        YP = YelpClient()
        filedate = '2022-10-19'
        YP.yelp_data = pd.read_csv(YP.machine + YP.output_folder + 'yelp_data' +  '_' + filedate + '.csv')
        YP.yelp_data = YP.yelp_data.merge(pd.read_csv(YP.machine + YP.output_folder + 'number_rests_in_tracts.csv'), on='BoroCT2020')
        # GH dataset
        GH = GrubhubClient()
        GH.grub_data = pd.read_csv(GH.machine + GH.folder + GH.analysis_folder + 'restaurants_delivering_from_tracts.csv', index_col = [0])
        GH.grub_data['phone_number'] = GH.grub_data['phone_number'].astype(str).apply(lambda s: re.sub("\D","",s))
        GH.grub_data['phone_number'] = '1' + GH.grub_data['phone_number'].astype(str)
        #GH.grub_data = GH.grub_data.dropna(axis=0, subset='restaurant_id')
        GH.grub_data.sort_values(by='rating_count', axis=0, inplace=True)
        GH.grub_data.drop_duplicates(['phone_number', 'latitude', 'longitude'], inplace=True)
        GH.grub_data.drop_duplicates(['phone_number', 'BoroCT2020'], inplace=True)
        GH.grub_data.sort_index(0, inplace=True)
        GH.grub_data = GH.grub_data.reset_index(drop=True)
        # from 24.3k to 19.4k
        GH.grub_data['phone'] = GH.grub_data['phone_number'].astype(np.float64)
        #GH.grub_data.drop_duplicates('phone', inplace=True)
        GH.grub_data['isGH'] = True
        dataset = YP.yelp_data.merge(GH.grub_data, on = ['phone', 'BoroCT2020'], how = 'left')

        phone_cross = pd.merge(YP.yelp_data.phone, GH.grub_data.phone, how ='cross')
        phone_cross['equal'] = phone_cross['phone_x'] == phone_cross['phone_y']
        CT_cross = pd.merge(YP.yelp_data.BoroCT2020, GH.grub_data.BoroCT2020, how ='cross')
        CT_cross['equal'] = CT_cross['BoroCT2020_x'] == CT_cross['BoroCT2020_y']
                
        phone_cross_equal =  phone_cross['equal'].values.reshape(len(YP.yelp_data), len(GH.grub_data), order='C')
        del phone_cross
        CT_cross_equal =  CT_cross['equal'].values.reshape(len(YP.yelp_data), len(GH.grub_data), order='C')
        del CT_cross
        
        CT_phone_equal = np.multiply(phone_cross_equal, CT_cross_equal)
        first_matches = np.where(CT_phone_equal[:,CT_phone_equal.sum(0)<=1]==1)
        
        first_matches_df = pd.concat((YP.yelp_data.loc[first_matches[0]].reset_index(),GH.grub_data.loc[first_matches[1]].reset_index()),axis=1)
        
        YP2 = YP.yelp_data.loc[~YP.yelp_data.index.isin(first_matches[0])]
        GH2 = GH.grub_data.loc[~GH.grub_data.index.isin(first_matches[1])]
        YP2.reset_index(drop=True, inplace=True)
        GH2.reset_index(drop=True, inplace=True)
        
    
        names = np.concatenate((YP2.name.values, GH2.name.values)).reshape(-1,1)
        ratio_matrix = squareform(pdist(names, lambda x,y: ratio(x[0], y[0])))
        ratios = pd.DataFrame(ratio_matrix, index=names.ravel(), columns=names.ravel())
        
        ratios_ = ratios.iloc[:len(YP2), len(YP2):]
        max_ratios=np.argmax(ratios_.values, axis=0)
        ratios_CT = np.multiply(CT_cross_equal,ratios_)
        
        ratios_cross = np.zeros(ratios_.shape)
        for i in range(ratios_.shape[1]):
            ratios_cross[max_ratios[i],i]=1
        
        CT_cross_equal = np.delete(CT_cross_equal, first_matches[0], axis=0)
        CT_cross_equal = np.delete(CT_cross_equal, first_matches[1], axis=1)
        
        
        second_matches = np.where(ratios_CT==1)
        
        second_matches_df = pd.concat((YP2.loc[second_matches[0]].reset_index(),GH2.loc[second_matches[1]].reset_index()),axis=1)
        
        merged_df = pd.concat((first_matches_df, second_matches_df), axis=0)
        merged_df.pop('index')
        merged_df.drop_duplicates(['id'], inplace=True)
        merged_df.to_csv(YP.machine + 'data/' + 'new_merged_dataset.csv')
        
    def findRestaurantsAroundNeighborhood():
        
        num_rests_yelp = pd.read_csv('yelp_data/number_rests_in_tracts.csv', index_col=[0])
        num_rests_yelp.index=num_rests_yelp.index.astype(int)
        
        closed_touching_ct = pd.read_csv('nyc_geodata/census_tracts_boundaries/touching_census_tracts.csv', index_col=[0])
        number_rests_touching_ct = closed_touching_ct.T.dot(num_rests_yelp)
        number_rests_touching_ct.index.name = 'BoroCT2020'
        number_rests_touching_ct=number_rests_touching_ct.astype(int)
        number_rests_touching_ct.index=number_rests_touching_ct.index.astype(int)
        
        ct_within1000 = pd.read_csv('nyc_geodata/distances/ct_dummies_linear_distances_1000.csv', index_col=[0])
        ct_within1000['within'] = ct_within1000['within'].apply(ast.literal_eval)
        ct_within1000 = pd.get_dummies(ct_within1000['within'].apply(pd.Series).stack()).sum(level=0)
        ct_within1000.columns = ct_within1000.columns.astype(int)
        number_rests_ct_within1000 = ct_within1000.T.dot(num_rests_yelp)
        number_rests_ct_within1000.index.name = 'BoroCT2020'
        number_rests_ct_within1000.index=number_rests_ct_within1000.index.astype(int)
        
        ct_within3000 = pd.read_csv('nyc_geodata/distances/ct_dummies_linear_distances_3000.csv', index_col=[0])
        ct_within3000['within'] = ct_within3000['within'].apply(ast.literal_eval)
        ct_within3000 = pd.get_dummies(ct_within3000['within'].apply(pd.Series).stack()).sum(level=0)
        ct_within3000.columns = ct_within3000.columns.astype(int)
        number_rests_ct_within3000 = ct_within3000.T.dot(num_rests_yelp)
        number_rests_ct_within3000.index.name = 'BoroCT2020'
        number_rests_ct_within3000.index=number_rests_ct_within3000.index.astype(int)
        
        same_NTA = pd.read_csv('nyc_geodata/census_tracts_boundaries/same_NTA.csv', index_col=[0])
        number_rests_NTA = same_NTA.T.dot(num_rests_yelp)
        number_rests_NTA.index.name = 'BoroCT2020'
        number_rests_NTA.index=number_rests_NTA.index.astype(int)
        
        num_delivering_stores_GH = pd.read_csv('grubhub_data/analysis/delivering_restaurants_in_tracts.csv', index_col=[0])
        num_delivering_stores_GH.index.name = 'BoroCT2020'
        num_delivering_stores_GH.index=num_delivering_stores_GH.index.astype(int)
        
        restaurants_around = num_rests_yelp.join(number_rests_touching_ct, how = 'right', on = 'BoroCT2020', rsuffix='_touch')
        restaurants_around = restaurants_around.join(number_rests_ct_within1000, how = 'right', on = 'BoroCT2020', rsuffix='_1000')
        restaurants_around = restaurants_around.join(number_rests_ct_within3000, how = 'right', on = 'BoroCT2020', rsuffix='_3000')
        restaurants_around = restaurants_around.join(number_rests_NTA, how = 'right', on = 'BoroCT2020', rsuffix='_NTA')
        restaurants_around = restaurants_around.join(num_delivering_stores_GH, how = 'right', on = 'BoroCT2020', rsuffix='_storesGH')
        
        plt.figure()
        plt.scatter(restaurants_around.NumberGHDeliveringRestaurants, restaurants_around.NumberYelpRestaurants_touch, s=4)
        plt.xlabel('Number of restaurants delivering in census tract $z_j$')
        plt.title('Number of restaurants in neighboring census tracts of tract $z_j$')
        plt.savefig('figures/no_touch.png', dpi=300)
        
        plt.figure()
        plt.scatter(restaurants_around.NumberGHDeliveringRestaurants, restaurants_around.NumberYelpRestaurants_1000, s=4)
        plt.xlabel('Number of restaurants delivering in census tract $z_j$')
        plt.title('Number of restaurants in neighboring census tracts within 1 km of tract $z_j$')
        plt.savefig('figures/no_1000.png', dpi=300)
        
        plt.figure()
        plt.scatter(restaurants_around.NumberGHDeliveringRestaurants, restaurants_around.NumberYelpRestaurants_3000, s=4)
        plt.xlabel('Number of restaurants delivering in census tract $z_j$')
        plt.title('Number of restaurants in neighboring census tracts within 3 km of tract $z_j$')
        plt.savefig('figures/no_3000.png', dpi=300)
        
        plt.figure()
        plt.scatter(restaurants_around.NumberGHDeliveringRestaurants, restaurants_around.NumberYelpRestaurants_NTA, s=4)
        plt.xlabel('Number of restaurants delivering in census tract $z_j$')
        plt.title('Number of restaurants in neighboring census tracts in the same NTA as $z_j$')
        plt.savefig('figures/no_NTA.png', dpi=300)
        
        num_delivering_rests_GH_mat = pd.read_csv('grubhub_data/analysis/where_restaurants_deliver_by_GH_id.csv', index_col=[0])
        num_delivering_rests_GH = num_delivering_rests_GH_mat.sum(0).astype(int)
        num_delivering_rests_GH.index.name = 'BoroCT2020'
        num_delivering_rests_GH.index=num_delivering_rests_GH.index.astype(int)
        num_delivering_rests_GH = pd.DataFrame(num_delivering_rests_GH, columns = ['NumberGHDeliveringRestaurantsMatched'])
        
        restaurants_around = restaurants_around.join(num_delivering_rests_GH, how = 'right', on = 'BoroCT2020', rsuffix='_storesGH')
        
        plt.figure()
        plt.scatter(restaurants_around.NumberGHDeliveringRestaurantsMatched, restaurants_around.NumberYelpRestaurants_touch, s=1)
        plt.xlabel('Number of restaurants delivering in census tract $z_j$')
        plt.title('Number of restaurants in neighboring census tracts of tract $z_j$')
        
        plt.figure()
        plt.scatter(restaurants_around.NumberGHDeliveringRestaurantsMatched, restaurants_around.NumberYelpRestaurants_1000, s=1)
        plt.xlabel('Number of restaurants delivering in census tract $z_j$')
        plt.title('Number of restaurants in neighboring census tracts within 1 km of tract $z_j$')
        
        plt.figure()
        plt.scatter(restaurants_around.NumberGHDeliveringRestaurantsMatched, restaurants_around.NumberYelpRestaurants_3000, s=1)
        plt.xlabel('Number of restaurants delivering in census tract $z_j$')
        plt.title('Number of restaurants in neighboring census tracts within 3 km of tract $z_j$')
        
        plt.figure()
        plt.scatter(restaurants_around.NumberGHDeliveringRestaurantsMatched, restaurants_around.NumberYelpRestaurants_NTA, s=1)
        plt.xlabel('Number of restaurants delivering in census tract $z_j$')
        plt.title('Number of restaurants in neighboring census tracts in the same NTA as $z_j$')
        
        num_delivering_id = num_delivering_rests_GH_mat.sum(1).astype(int)
        num_delivering_id = pd.DataFrame(num_delivering_id, columns = ['NumberCensusTractsVisited'])
        delivering_w_census = pd.read_csv('grubhub_data/analysis/restaurants_delivering_from_tracts.csv', index_col=[0])[['restaurant_id','BoroCT2020']].join(num_delivering_id, on = 'restaurant_id', how = 'right')
        delivering_w_census = delivering_w_census[delivering_w_census.BoroCT2020!=4080301]
        delivering_w_census_min = delivering_w_census.groupby('BoroCT2020').min()
        delivering_w_census_median = delivering_w_census.groupby('BoroCT2020').median()
        delivering_w_census_max = delivering_w_census.groupby('BoroCT2020').min()
        #delivering_w_census_min = gpd.GeoDataFrame(delivering_w_census_min.join(census_df, on = 'BoroCT2020', how = 'right'))
        #delivering_w_census_median = gpd.GeoDataFrame(delivering_w_census_median.join(census_df, on = 'BoroCT2020', how = 'right'))
        #delivering_w_census_max = gpd.GeoDataFrame(delivering_w_census_max.join(census_df, on = 'BoroCT2020', how = 'right'))

       
        missing_kwds = dict(color='grey', label='No Data')
        fig, ax = plt.subplots(figsize = (20,20))
        delivering_w_census_min.plot('NumberCensusTractsVisited', ax=ax, cmap ='YlOrRd', legend=True, legend_kwds={'shrink': 0.4}, missing_kwds=missing_kwds)
        fig.savefig('figures/census_visited_min.png', bbox_inches='tight')
        
        missing_kwds = dict(color='grey', label='No Data')
        fig, ax = plt.subplots(figsize = (20,20))
        delivering_w_census_median.plot('NumberCensusTractsVisited', ax=ax, cmap ='YlOrRd', legend=True, legend_kwds={'shrink': 0.4}, missing_kwds=missing_kwds)
        fig.savefig('figures/census_visited_median.png', bbox_inches='tight')
        
        missing_kwds = dict(color='grey', label='No Data')
        fig, ax = plt.subplots(figsize = (20,20))
        delivering_w_census_max.plot('NumberCensusTractsVisited', ax=ax, cmap ='YlOrRd', legend=True, legend_kwds={'shrink': 0.4}, missing_kwds=missing_kwds)
        fig.savefig('figures/census_visited_max.png', bbox_inches='tight')

        
#%%   

RD = RetrieveData()
RD.setCityLocations()
with ThreadPoolExecutor(max_workers=5) as executor:
    executor.map(lambda row: RD.retrieveGrubHubByCT(row, '01-31-2023'),
                 list(range(2100,2325)),
                 #list(range(1400)),
                 #list(range(1800)),
                timeout = 3600)

#%%

def main():
    RD = RetrieveData()


#%%

# p = sns.displot(dataset, x = 'cosine_all', hue = 'BoroName', kind = 'kde')
# p.fig.set_dpi(200)
# #p.fig.savefig('figures/similarity_boroughs.png')

# #%%
# p =sns.displot(dataset, x = 'cosine_NTA', hue = 'BoroName', kind = 'kde')
# p.fig.set_dpi(200)
# #p.fig.savefig('figures/similarity_NTA_boroughs.png')

# #%%

# p = sns.displot(dataset, x = 'similarity_NTA', hue = 'isGH', col = 'NTAName' , kind = 'kde')
# p.fig.set_dpi(200)
# p.fig.savefig('figures/isGH.png')

# p =sns.displot(dataset, x = 'similarity_NTA', hue = 'delivery_mode', kind = 'kde')
# p.fig.set_dpi(200)
# p.fig.savefig('figures/delivery_mode.png')

# #%%

# p = sns.displot(yelp_rests, x = 'distance', hue = 'BoroName', kind = 'kde')#, col = 'NTAName' , kind = 'kde')
# p.fig.set_dpi(200)
# p.fig.savefig('figures/dist_boro.png')

# p = sns.displot(yelp_rests, x = 'distance_within', hue = 'BoroName', kind = 'kde')#, col = 'NTAName' , kind = 'kde')
# p.fig.set_dpi(200)
# p.fig.savefig('figures/dist_within_boro.png')

# p = sns.displot(dataset, x = 'distance_within', hue = 'isGH', kind = 'kde')#, col = 'NTAName' , kind = 'kde')
# p.fig.set_dpi(500)
# p.fig.savefig('figures/isGH.png')

# p = sns.displot(dataset, x = 'distance', hue = 'isGH', kind = 'kde')#, col = 'NTAName' , kind = 'kde')
# p.fig.set_dpi(500)
# p.fig.savefig('figures/isGH_within.png')

# p = sns.displot(yelp_rests, x = 'distance_within', hue = 'BoroName', kind = 'kde')#, col = 'NTAName' , kind = 'kde')
# p.fig.set_dpi(200)
# p.fig.savefig('figures/dist_within_boro.png')


# p =sns.displot(yelp_rests, x = 'similarity_NTA', hue = 'delivery_mode', kind = 'kde')
# p.fig.set_dpi(200)
# p.fig.savefig('figures/delivery_mode.png')


