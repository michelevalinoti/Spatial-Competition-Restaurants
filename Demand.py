#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 15:59:24 2023

@author: michelev
"""

import numpy as np
import pandas as pd

import geopandas as gpd



#%%

machine = '/Users/michelev/spatial-competition-food/'
foot_traffic_folder = 'advan_data/'

def createDemandDataFrame(filename):
    
    year = 2010
    
    census_tracts = gpd.read_file(machine + 'nyc_geodata/census_tracts_boundaries/nyct' + str(year) + '_23a/nyct' + str(year) + '.shp')
    
    counties = {'Bronx': ['36005'], # Bronx = Bronx County
                'Brooklyn': ['36047'], # Brooklyn = Kings County
                'Manhattan': ['36061'], # Manhattan = New York County
                'Queens': ['36081'], # Queens = Queens County
                'Staten Island': ['36085']  # Staten Island = Richmond County
                }
    
    counties = pd.DataFrame(counties).T
    counties = counties.reset_index()
    #counties = counties.rename_axis("BoroName", axis="columns")
    counties.rename({'index': 'BoroName', 0: 'FIPS'}, axis=1, inplace=True)
    census_blocks = gpd.read_file(machine + 'nyc_geodata/census_blocks/nycb' + str(year) + '_23a/nycb' + str(year) + '.shp')
    #census_block_groups = np.vectorize(lambda s: s[:-1])(census_blocks.BCTCB2010.values).astype(int)
    #census_block_groups = census_blocks.BCTCB2020.values#.astype(int)
    census_tracts = census_tracts.merge(counties, on = 'BoroName')
    census_tracts['CT_full'] = census_tracts['FIPS'] + census_tracts['BoroCT' + str(year)].map(lambda x: str(x)[1:])
    
    distances = pd.read_csv(machine + 'nyc_geodata/distances/euclidean_distances_' + str(year) + '.csv')
    distances.set_index('BoroCT' + str(year), inplace=True)
    
    distances_unstack = distances.unstack().reset_index()
    distances_unstack.rename({'BoroCT' + str(year): 'origin', 'level_0': 'destination', 0: 'distance'}, axis=1, inplace=True)
    distances_unstack['OD'] = distances_unstack['origin'] + distances_unstack['destination']

  

    excluded_columns = ['parent_placekey',
                        'safegraph_brand_ids',
                        'brands',
                        'store_id',
                        'websites',
                        'polygon_wkt',
                        'polygon_class',
                        'related_same_day_brand',
                        'related_same_month_brand',
                        'device_type',
                        ]
    
    categories = {'food_categories':   ['Specialty Food Stores',
                                        'Special Food Services',
                                        'Drinking Places (Alcoholic Beverages)',
                                        'Restaurants and Other Eating Places'
                                        ],
                'less_frequent_categories': ['Furniture Stores',
                                            'Electronics and Appliance Stores',
                                            'Clothing Stores',
                                            'Shoe Stores'],
                'more_frequent_categories': ['Health and Personal Care Stores',
                                            'Grocery Stores',
                                            'Beer, Wine, and Liquor Stores']
                }
                
    
    df = pd.read_csv(machine + foot_traffic_folder + filename)
    df['poi_CT'] = df['poi_cbg'].map(lambda x: str(x)[:-1])
    
    df = df.loc[df.region == 'NY']
    #df['poi_cbg']
    df = df.loc[:, ~df.columns.isin(excluded_columns)]
    df = df.loc[(df.poi_CT.isin(census_tracts['CT_full'].values))]
    
    final_arrays = {}
    for origin in ['home', 'daytime']:
        
        category_dict = {}
        for category in categories.keys():
            
            df_ = df.copy()
            df_ = df_.loc[(df_.top_category.isin(categories[category]))]
            
            trips_df = pd.DataFrame()
            for poi in df_.index:
                
                if origin == 'daytime':
                    trips_visits = df_['visitor_daytime_cbgs'].loc[poi]
                elif origin == 'home':
                    trips_visits = df_['visitor_home_aggregation'].loc[poi]
                if pd.isnull(trips_visits):
                    continue
                trips_visits = ast.literal_eval(trips_visits)
                trips_visits = pd.DataFrame(trips_visits, index = [poi])
                # remove last string
                if origin == 'daytime':
                    for col in trips_visits.columns:
                        trips_visits.rename({col: col[:-1]}, axis=1, inplace=True)
                trips_df = pd.concat((trips_df, pd.DataFrame(trips_visits, index = [poi])), axis=1)
                
            trips_df = trips_df.loc[:, trips_df.columns.isin(census_tracts['CT_full'].values)]
            trips_df = trips_df.groupby(trips_df.columns, axis=1).sum()
            trips_df = pd.DataFrame(df_.poi_CT).merge(trips_df, how = 'left', left_index=True, right_index=True)
            trips_df.set_index('poi_CT', inplace = True)
            
            
            # rewrite indices of distances
            dict_key = census_tracts[['BoroCT'+ str(year), 'CT_full']]
            dict_key = pd.DataFrame(dict_key).set_index('BoroCT'+ str(year)).to_dict()['CT_full']
            
            distances.columns = distances.columns.astype(str)
            distances.index = distances.index.astype(str)
            distances = distances.rename(dict_key, axis=0)
            distances = distances.rename(dict_key, axis=1)
            
            
            trips_unstack = trips_df.unstack().reset_index()
            trips_unstack.rename({'poi_CT': 'destination', 'level_0': 'origin', 0: 'frequency'}, axis=1, inplace=True)
            trips_unstack['OD'] = trips_unstack['origin'] + trips_unstack['destination']
            
            final = pd.merge(distances_unstack, trips_unstack,  how='right', left_on=['origin','destination'], right_on=['origin','destination'])
            final = final.loc[(np.isnan(final.frequency) == False) & (final.frequency != 0)]
            final = final.loc[final.distance<=10000]
            
            category_dict[category] = final.distance.repeat(final.frequency)
            
        final_arrays[origin] = category_dict
    #df = df.merge(pd.DataFrame(trips_visits, index = [poi]), how = 'left', left_index=True, right_index=True)
    
    for origin in ['home', 'daytime']:
        
        for category in categories.keys():
            
                final_arrays[origin][category].plot.kde()
                plt.legend(category)
#%%

class DiscreteChoice:

#%%
    
class DemandEstimation:
    
    