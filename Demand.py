#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 15:59:24 2023

@author: michelev
"""

import numpy as np
import pandas as pd

import geopandas as gpd
import matplotlib.pyplot as plt

from scipy import stats
import statsmodels.nonparametric.kernel_density as snk
import json
import base64
import requests
import ast
import re
import random
from collections import Counter
#%%


machine = '/Users/michelev/spatial-competition-food/'
foot_traffic_folder = 'advan_data/'


un = "mv2164@nyu.edu" # Set username
pw = "Fpdfqt6li2023!" # Set password

credentials = f"{un}:{pw}" # Format credentials according to the API's expectations
print(credentials)

#%%

credentials_bytes = credentials.encode('ascii')
base64_credentials_bytes = base64.b64encode(credentials_bytes)
base64_credentials = base64_credentials_bytes.decode('ascii')
print(base64_credentials)

#%%

headers = {
    'accept': 'application/json',
    'Authorization': f'Basic {base64_credentials}'
}

response = requests.post("https://marketplace.deweydata.io/api/auth/tks/get_token", headers=headers)
print(response.json())
access_token = response.json()['access_token']
print(access_token)

#%%

def createUrlsDataFrame(year, access_token):
    
    headers = {
        'accept': 'application/json',
        'Authorization': f'Bearer {access_token}'
    }
    
    retained_links = []
    
    for month in range(1,13):
        response = requests.get("https://marketplace.deweydata.io/api/data/v2/list/" + str(year) + "/" + str(month) + "/01/ADVAN/MP/", headers=headers)
        links = response.json()
        
        regex = re.compile(f"^patterns_monthly")
        
        for link in links:
            
            if regex.match(link['name']):
                retained_links.append(link)
           
    df = pd.DataFrame(retained_links)
            
    df.to_csv(machine + foot_traffic_folder + 'all_urls_' + str(year) + '.csv')
            
#%%

#for year in [2019, 2020, 2021, 2022]:
#    createUrlsDataFrame(year, access_token)


#%%

def filterDataFrame(year, access_token):
    
    census_year = 2010
    census_tracts = gpd.read_file(machine + 'nyc_geodata/census_tracts_boundaries/nyct' + str(census_year) + '_23a/nyct' + str(census_year) + '.shp')
    
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
    census_blocks = gpd.read_file(machine + 'nyc_geodata/census_blocks/nycb' + str(census_year) + '_23a/nycb' + str(census_year) + '.shp')
    #census_block_groups = np.vectorize(lambda s: s[:-1])(census_blocks.BCTCB2010.values).astype(int)
    #census_block_groups = census_blocks.BCTCB2020.values#.astype(int)
    census_tracts = census_tracts.merge(counties, on = 'BoroName')
    census_tracts['CT_full'] = census_tracts['FIPS'] + census_tracts['BoroCT' + str(census_year)].map(lambda x: str(x)[1:])
     
    census_tracts[['BoroCT' + str(census_year),'CT_full']] = census_tracts[['BoroCT' + str(census_year),'CT_full']].astype(int)
    
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
                        'phone_number',
                        'is_synthetic',
                        'includes_parking_lot',
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
                
    top_categories = sum(list(categories.values()), [])
    urls_df = pd.read_csv(machine + foot_traffic_folder + 'all_urls_' + str(year) + '.csv')
    
    fid_list = list(urls_df.url)
    
    count = 0
    final_df = pd.DataFrame()
    for fid in fid_list[:500]:
        
        if count % 100 == 0:
            print(count)
        
        data_url = f"https://marketplace.deweydata.io{fid}"

        headers = {
            'accept': 'application/json',
            'Authorization': f'Bearer {access_token}'
        }

        try:
            df = pd.read_csv(data_url, storage_options=headers, compression='gzip', usecols= lambda x: x not in excluded_columns, low_memory=False)
        except:
            continue
        df['poi_CT'] = df['poi_cbg'].map(lambda x: str(x)[:-1])
         
        census_tracts['CT_full'] = census_tracts['CT_full'].astype(str)
        df = df.loc[df.region == 'NY']
        #df['poi_cbg']
        #df = df.loc[:, ~df.columns.isin(excluded_columns)]
        df = df.loc[df.poi_CT.isin(census_tracts['CT_full'].astype(str).values)]
        df = df.loc[df.top_category.isin(top_categories)]
        if count == 0:
            
            final_df = df.copy()
            
        else:
            
            final_df = pd.concat((final_df, df), axis=0)
            
        count += 1
        
    final_df.to_csv(machine + foot_traffic_folder + 'advand_data_filtered_' + str(year) + '.csv')
    
#%%

filterDataFrame(2019, access_token)

#%%


def createDemandDataFrame(year):
    
    census_year = 2010
    
    census_tracts = gpd.read_file(machine + 'nyc_geodata/census_tracts_boundaries/nyct' + str(census_year) + '_23a/nyct' + str(census_year) + '.shp')
    
    counties = {#'Bronx': ['36005'], # Bronx = Bronx County
                #'Brooklyn': ['36047'], # Brooklyn = Kings County
                'Manhattan': ['36061'], # Manhattan = New York County
                #'Queens': ['36081'], # Queens = Queens County
                #'Staten Island': ['36085']  # Staten Island = Richmond County
                }
    
    counties = pd.DataFrame(counties).T
    counties = counties.reset_index()
    #counties = counties.rename_axis("BoroName", axis="columns")
    counties.rename({'index': 'BoroName', 0: 'FIPS'}, axis=1, inplace=True)
    census_blocks = gpd.read_file(machine + 'nyc_geodata/census_blocks/nycb' + str(census_year) + '_23a/nycb' + str(census_year) + '.shp')
    #census_block_groups = np.vectorize(lambda s: s[:-1])(census_blocks.BCTCB2010.values).astype(int)
    #census_block_groups = census_blocks.BCTCB2020.values#.astype(int)
    census_tracts = census_tracts.merge(counties, on = 'BoroName')
    census_tracts['CT_full'] = census_tracts['FIPS'] + census_tracts['BoroCT' + str(census_year)].map(lambda x: str(x)[1:])
    
    distances = pd.read_csv(machine + 'nyc_geodata/distances/euclidean_distances_' + str(census_year) + '.csv')
    distances.set_index('BoroCT' + str(census_year), inplace=True)
    
    census_tracts[['BoroCT' + str(census_year),'CT_full']] = census_tracts[['BoroCT' + str(census_year),'CT_full']].astype(int)
    
    distances_unstack = distances.unstack().reset_index()
    distances_unstack.rename({'BoroCT' + str(census_year): 'origin', 'level_0': 'destination', 0: 'distance'}, axis=1, inplace=True)
    distances_unstack['origin'] = distances_unstack['origin'].astype(int)
    distances_unstack['destination'] = distances_unstack['destination'].astype(int)
    
    distances_unstack=pd.merge(distances_unstack, census_tracts[['BoroCT' + str(census_year),'CT_full']], how='left',left_on = 'destination', right_on='BoroCT' + str(census_year))
    distances_unstack.dropna(subset = ['CT_full'],inplace=True)
    distances_unstack.pop('destination')
    distances_unstack.rename({'CT_full':'destination'}, axis=1,inplace=True)
    
    distances_unstack=pd.merge(distances_unstack.astype(int), census_tracts[['BoroCT' + str(census_year),'CT_full']], how='left',left_on = 'origin', right_on='BoroCT' + str(census_year))
    distances_unstack.dropna(subset = ['CT_full'],inplace=True)
    distances_unstack.pop('origin')
    distances_unstack.rename({'CT_full':'origin'}, axis=1,inplace=True)
   
    census_tracts['CT_full'] = census_tracts['CT_full'].astype(str)
    #distances_unstack['OD'] = distances_unstack['origin'].astype(str) + distances_unstack['destination'].astype(str)
    categories = {'food_categories':   [#'Specialty Food Stores',
                                        #'Special Food Services',
                                       # 'Drinking Places (Alcoholic Beverages)',
                                        'Restaurants and Other Eating Places'
                                        ],
                'less_frequent_categories': ['Furniture Stores',
                                            'Electronics and Appliance Stores',
                                            'Clothing Stores',
                                            'Shoe Stores'
                                            ],
                'more_frequent_categories': [#'Health and Personal Care Stores',
                                             'Grocery Stores'#,
                                            #'Beer, Wine, and Liquor Stores'
                                            ]
                }

    # categories = {'food_categories':   ['Limited-Service Restaurants',
    #                                     'Full-Service Restaurants'
    #                                     ],
    #             'less_frequent_categories': [#'Furniture Stores',
    #                                         #'Electronics and Appliance Stores',
    #                                         'Clothing Stores',
    #                                         'Shoe Stores'],
    #             'more_frequent_categories': [#'Health and Personal Care Stores',
    #                                          'Grocery Stores'#,
    #                                         #'Beer, Wine, and Liquor Stores'
    #                                         ]
    #             }
    
    df = pd.read_csv(machine + foot_traffic_folder + 'advand_data_filtered_' + str(year) + '.csv')
    
    final_arrays = {}
    for origin in ['home', 'daytime']:
        
        category_dict = {}
        for category in categories.keys():
            
            df_ = df.copy()
            df_ = df_.loc[(df_.top_category.isin(categories[category]))]
            df_ = df_[df_['poi_CT'].astype(str).str.startswith('36061')]
            trips_df = pd.DataFrame()
            #for poi in df_.index:
                
            if origin == 'daytime':
                #trips_visits = df_['visitor_daytime_cbgs'].loc[poi]
                trips_visits = df_['visitor_daytime_cbgs'].fillna('0').apply(ast.literal_eval)
            elif origin == 'home':
                #trips_visits = df_['visitor_home_aggregation'].loc[poi]
                trips_visits = df_['visitor_home_aggregation'].fillna('0').apply(ast.literal_eval)

            #if pd.isnull(trips_visits):
            #    continue
            
            trips_visits = trips_visits[trips_visits!=0]
            
            random.seed(0)
            
            #trips_visits = trips_visits.apply(Counter)
            #trips_visits = trips_visits.sum()
            trips_visits = pd.DataFrame(random.sample(list(trips_visits.values), min(5000, len(trips_visits.values))))
            #trips_visits = pd.DataFrame(list(trips_visits.values))
            # remove last string
            if origin == 'daytime':
                for col in trips_visits.columns:
                    trips_visits.rename({col: col[:-1]}, axis=1, inplace=True)
            
            trips_df = trips_visits.copy()
            del trips_visits
            trips_df = trips_df.loc[:, trips_df.columns.isin(census_tracts['CT_full'].values)]
            trips_df = trips_df.groupby(trips_df.columns, axis=1).sum()
            trips_df = pd.DataFrame(df_.poi_CT).merge(trips_df, how = 'left', left_index=True, right_index=True)
            trips_df.set_index('poi_CT', inplace = True)
            
            
            # rewrite indices of distances
            dict_key = census_tracts[['BoroCT'+ str(census_year), 'CT_full']]
            dict_key = pd.DataFrame(dict_key).set_index('BoroCT'+ str(census_year)).to_dict()['CT_full']
            
            distances.columns = distances.columns.astype(str)
            distances.index = distances.index.astype(str)
            distances = distances.rename(dict_key, axis=0)
            distances = distances.rename(dict_key, axis=1)
            
            
            trips_unstack = trips_df.unstack().reset_index()
            trips_unstack.rename({'poi_CT': 'destination', 'level_0': 'origin', 0: 'frequency'}, axis=1, inplace=True)
            #trips_unstack['OD'] = trips_unstack['origin'] + trips_unstack['destination']
            trips_unstack = trips_unstack.fillna(0)
            
            final = pd.merge(distances_unstack.astype(int), trips_unstack.astype(int),  how='right', left_on=['origin','destination'], right_on=['origin','destination'])
            final = final.loc[(np.isnan(final.frequency) == False) & (final.frequency != 0)]
            final = final.loc[final.distance<=5000]
            
            category_dict[category] = final.distance.repeat(final.frequency)
            
        final_arrays[origin] = category_dict
    #df = df.merge(pd.DataFrame(trips_visits, index = [poi]), how = 'left', left_index=True, right_index=True)
    
    kde = {}
    
    for origin in ['home', 'daytime']:
        
        kde_origin = {}
        for category in categories.keys():
            
                kde_model = snk.KDEMultivariate(final_arrays[origin][category], var_type = 'c', bw=[250])
                kde_origin[category] = kde_model.pdf(np.arange(0,5000))
                #kde = stats.gaussian_kde(final_arrays[origin][category], bw_method='scott')
                #plt.plot(kde(np.arange(0,12000)))
        #plt.legend(labels=['Restaurant-related categories', 'Less local', 'More local'])
        
        kde[origin] = kde_origin
        
    for origin in ['home', 'daytime']:
        
        plt.figure()
        for category in categories.keys():
            
                #final_arrays[origin][category].plot.kde()
                #kde = stats.gaussian_kde(final_arrays[origin][category], bw_method='scott')
                plt.xlabel('$r\,(m)$')
                plt.plot(np.arange(0,5000), kde[origin][category])
                

        plt.legend(labels=['Restaurants', 'Furniture, Electronics, Clothing', 'Grocery Stores'])
        plt.savefig('foot_traffic_all_'+origin+'.png', dpi=300)
#%%
    categories = ['Limited-Service Restaurants',
                  'Full-Service Restaurants'
                  ]
              
    df = pd.read_csv(machine + foot_traffic_folder + 'advand_data_filtered_' + str(year) + '.csv')
    
    final_arrays = {}
    for origin in ['home', 'daytime']:
        
        category_dict = {}
        for category in categories:
            
            df_ = df.copy()
            df_ = df_.loc[df_.top_category=='Restaurants and Other Eating Places']
            df_ = df_.loc[df_.sub_category==category]
            
            df_ = df_[df_['poi_CT'].astype(str).str.startswith('36061')]
            trips_df = pd.DataFrame()
            #for poi in df_.index:
                
            if origin == 'daytime':
                #trips_visits = df_['visitor_daytime_cbgs'].loc[poi]
                trips_visits = df_['visitor_daytime_cbgs'].fillna('0').apply(ast.literal_eval)
            elif origin == 'home':
                #trips_visits = df_['visitor_home_aggregation'].loc[poi]
                trips_visits = df_['visitor_home_aggregation'].fillna('0').apply(ast.literal_eval)

            #if pd.isnull(trips_visits):
            #    continue
            
            trips_visits = trips_visits[trips_visits!=0]
            
            random.seed(0)
            
            #trips_visits = trips_visits.apply(Counter)
            #trips_visits = trips_visits.sum()
            trips_visits = pd.DataFrame(random.sample(list(trips_visits.values), min(5000, len(trips_visits.values))))
            #trips_visits = pd.DataFrame(list(trips_visits.values))
            # remove last string
            if origin == 'daytime':
                for col in trips_visits.columns:
                    trips_visits.rename({col: col[:-1]}, axis=1, inplace=True)
            
            trips_df = trips_visits.copy()
            del trips_visits
            trips_df = trips_df.loc[:, trips_df.columns.isin(census_tracts['CT_full'].values)]
            trips_df = trips_df.groupby(trips_df.columns, axis=1).sum()
            trips_df = pd.DataFrame(df_.poi_CT).merge(trips_df, how = 'left', left_index=True, right_index=True)
            trips_df.set_index('poi_CT', inplace = True)
            
            
            
            # rewrite indices of distances
            dict_key = census_tracts[['BoroCT'+ str(census_year), 'CT_full']]
            dict_key = pd.DataFrame(dict_key).set_index('BoroCT'+ str(census_year)).to_dict()['CT_full']
            
            distances.columns = distances.columns.astype(str)
            distances.index = distances.index.astype(str)
            distances = distances.rename(dict_key, axis=0)
            distances = distances.rename(dict_key, axis=1)
            
            
            trips_unstack = trips_df.unstack().reset_index()
            trips_unstack.rename({'poi_CT': 'destination', 'level_0': 'origin', 0: 'frequency'}, axis=1, inplace=True)
            #trips_unstack['OD'] = trips_unstack['origin'] + trips_unstack['destination']
            trips_unstack = trips_unstack.fillna(0)
            
            final = pd.merge(distances_unstack.astype(int), trips_unstack.astype(int),  how='right', left_on=['origin','destination'], right_on=['origin','destination'])
            final = final.loc[(np.isnan(final.frequency) == False) & (final.frequency != 0)]
            final = final.loc[final.distance<=5000]
            
            category_dict[category] = final.distance.repeat(final.frequency)
            
        final_arrays[origin] = category_dict
    #df = df.merge(pd.DataFrame(trips_visits, index = [poi]), how = 'left', left_index=True, right_index=True)
    
    kde1 = {}
    
    for origin in ['home', 'daytime']:
        
        kde_origin = {}
        for category in categories:
            
                kde_model = snk.KDEMultivariate(final_arrays[origin][category], var_type = 'c', bw=[250])
                kde_origin[category] = kde_model.pdf(np.arange(0,5000))
                #kde = stats.gaussian_kde(final_arrays[origin][category], bw_method='scott')
                #plt.plot(kde(np.arange(0,12000)))
        #plt.legend(labels=['Restaurant-related categories', 'Less local', 'More local'])
        
        kde1[origin] = kde_origin
        
    for origin in ['home', 'daytime']:
        
        plt.figure()
        for category in categories:
            
                #final_arrays[origin][category].plot.kde()
                #kde = stats.gaussian_kde(final_arrays[origin][category], bw_method='scott')
                plt.plot(np.arange(0,5000), kde1[origin][category])
        plt.legend(labels=['Limited-Service Restaurants', 'Full-Service Restaurants'])
        plt.xlabel('$r\,(m)$')
        plt.savefig('foot_traffic_restaurants_'+origin+'.png', dpi=300)