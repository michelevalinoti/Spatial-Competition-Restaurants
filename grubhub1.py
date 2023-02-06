#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 15:07:54 2022

@author: michelev
"""

from bs4 import BeautifulSoup
import requests
import json

import re

import pandas as pd
from flatten_dict import flatten

import numpy as np
import os

import ast

from datetime import date



from scipy.sparse import csr_matrix, save_npz, load_npz
from scipy.spatial.distance import pdist, squareform

from networkx import from_numpy_array as nx_from_numpy_array, from_scipy_sparse_array as nx_from_scipy_sparse_array, connected_components
import networkx.algorithms.community as nx_comm

import dask.dataframe as dd

import Base
from geo import GeoData

#%%

class GrubhubClient:

    session = None
    session_results = None
    #proxies = {'http': 'http://brd-customer-hl_33f39684-zone-zone1:9tzrgl2f2e55@zproxy.lum-superproxy.io:22225',
    #           'https': 'http://brd-customer-hl_33f39684-zone-zone1:9tzrgl2f2e55@zproxy.lum-superproxy.io:22225'}
    proxies ={'http': 'http://brd-customer-hl_33f39684-zone-zone1-country-us:9tzrgl2f2e55@zproxy.lum-superproxy.io:22225',
            'https': 'http://brd-customer-hl_33f39684-zone-zone1-country-us:9tzrgl2f2e55@zproxy.lum-superproxy.io:22225'}
    #proxies = {'http': 'http://brd-customer-hl_33f39684-zone-residential:94evyj5kfwr5@zproxy.lum-superproxy.io:22225',
    #        'https': 'http://brd-customer-hl_33f39684-zone-residential:94evyj5kfwr5@zproxy.lum-superproxy.io:22225'}
    restaurants = None
    file_name = None
    
    machine = '/Users/michelev/spatial-competition-food/'
    folder = 'grubhub_data/'
    output_folder = 'census_tracts/'
    analysis_folder = 'data_analysis/'
    network_folder = 'network_analysis/'
    
    census_folder = 'nyc_geodata/census_tracts_boundaries/'
    census_filename = 'census_tracts.shp'
    centroids_filename = 'census_tracts_centroids.shp'
    
    SORT_MODES = ['default', # (Default)
                  'restaurant_name', # Sort search results by (Restaurant Name)
                  'price', # " (Price (Ascending))
                  'price_descending', # (Price (Descending))
                  'avg_rating', # (Rating)
                  'distance', # (Distance)
                  'delivery_estimate', # (Delivery Estimate)
                  'delivery_minimum'] # (Delivery Minimum)
    
    CUISINES = ['Alcohol',
                'American',
                'Asian',
                'Bagels',
                'Bakery',
                'Bowls',
                'Breakfast',
                'Burritos',
                'Cafe',
                'Cakes',
                'Calzones',
                'Caribbean',
                'Chicken',
                'Chinese',
                'Coffee%20and%20Tea',
                'Convenience',
                'Deli',
                'Dessert',
                'Dominican',
                'Donuts',
                'Fast%20Food',
                'Grill',
                'Grocery%20Items',
                'Halal',
                'Hamburgers',
                'Healthy',
                'Ice&Cream',
                'Italian',
                'Japanese',
                'Kids%20Menu',
                'Latin%20American',
                'Lunch%20Specials',
                'Mexican',
                'National%20Picks',
                'Noodles',
                'Pasta',
                'Pizza',
                'Salads',
                'Sandwiches',
                'Seafood',
                'Shakes',
                'Smoothies%20and%20',
                'Snacks',
                'Soup',
                'Steak',
                'Subs',
                'Sushi',
                'Tacos',
                'Vegetarian',
                'Wings',
                'Wraps'
                ]
    
    def setPoint(self, point):
        
        self.point = point
        
    def setFileName(self, name):
        
        self.file_name = name
        
    # get a set of results from a particular page
    # method based on https://stackoverflow.com/questions/62857914/using-scrapy-to-scrape-food-aggregators-like-grubhub-need-it-for-some-personal-d
    # see also https://github.com/jlumbroso/grubhub

    def newSession(self):
    
        #self.proxies = setProxy()
        session = requests.Session()
        #session.proxies  = self.proxies
        static = 'https://www.grubhub.com/eat/static-content-unauth?contentOnly=1'
        soup = BeautifulSoup(session.get(static).text, 'html.parser')
        client = re.findall("beta_[a-zA-Z0-9]+", soup.find('script', {'type': 'text/javascript'}).text)
    
        headers = {
                'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.116 Safari/537.36',
                'authorization': 'Bearer',
                'content-type': 'application/json;charset=UTF-8'
                  }
        session.headers.update(headers)
    
        # straight from networking tools. Device ID appears to accept any 10-digit value
        data = '{"brand":"grubhub","client_id":"' + client[0] + '","device_id":1234567890,"scope":"anonymous"}'
        resp = session.post('https://api-gtm.grubhub.com/auth', data=data)
        
        refresh = json.loads(resp.text)['session_handle']['refresh_token']
        access = json.loads(resp.text)['session_handle']['access_token']
        
        # update header with new token
        session.headers.update({'authorization': 'Bearer ' + access})
        self.session = session
        
        return self.session

    # write url used to search restaurants in GH's API
    # returns a string
    
    def writeUrl(self, order_method, sort_mode, cuisine, page):
        
        # take latitude and longithde from shapely Point
        latitude = self.point.y
        longitude = self.point.x
        
        # define url string: order method is by default 'delivery' (alternative: 'pickup')
        url = 'https://www.grubhub.com/search?orderMethod=' + order_method
        # In GH if you search for a pickup order you choose from a map rather than a list
        location_mode = order_method.upper()
        url = url + '&locationMode=' + location_mode
        url = url + '&facetSet=umamiV2&pageSize=20&hideHateos=true&searchMetrics=true&'
        url = url + 'latitude=' + str(latitude)
        url = url + '&longitude=' + str(longitude)
        url = url + '&preciseLocation=true&geohash=dr5rswey3kjd'
        # while the order of the results does not matter I will experiment with different sorting methods
        # in order to capture as many results as possible
        url = url + '&sorts=' + sort_mode
        # filtering for different cuisines helps to get as many results as possible
        if cuisine != None:
            url = url + '&facet=cuisine%3A' + cuisine
        url = url + '&includeOffers=true&sortSetId=umamiv3&sponsoredSize=3&countOmittingTimes=true'
        url = url + '&pageNum=' + str(page)
    
        return url
        
    def searchPage(self, order_method = 'delivery', sort_mode = 'default', cuisine = None, page = 1):
        
        search = re.findall('(?<=search\?).*', self.writeUrl(order_method, sort_mode, cuisine, page))[0]
        
        self.session_results = self.session.get('https://api-gtm.grubhub.com/restaurants/search/search_listing?' + search, proxies = self.proxies)
        res_is_none = self.session_results == None
        res_wrong = True
        if res_is_none == False:
            res_wrong = self.session_results.status_code != 200
        while (res_is_none == True) | (res_wrong == True):
            #if res_is_none == False:
            #    print('! Status code: ' + str(session_results.status_code))
            #else:
            #    print('! None')
            # switch IP
            self.session_results = self.session.get('https://api-gtm.grubhub.com/restaurants/search/search_listing?' + search, proxies = self.proxies)
            res_is_none = self.session_results == None
            res_wrong = True
            if res_is_none == False:
                res_wrong = self.session_results.status_code != 200
        results = json.loads(self.session_results.text)
        return results
    
    def searchByCategory(self, page_size, offset, order_method = 'delivery', sort_mode = 'default', cuisine = None):
        
        restaurants = []
        for page in range(offset, page_size+1):
            res = self.searchPage(order_method, sort_mode, cuisine, page)
            restaurants.extend(res['results'])
        
        return restaurants
    
    def searchByPoint(self, order_method = 'delivery'):
        
        # create new session
        print('--- Create session --- ')
        self.newSession()
        
        # find ALL results (restaurants that deliver to point) contained in page 1
        # for dense areas, there could be too many results than the ones one can parse by just changing pages
        res = self.searchPage(order_method)
        total_results = res['stats']['total_results']
        print('* Found ' + str(total_results) + ' results.')
        result_count = res['stats']['result_count']
        page_size = res['stats']['page_size']
        total_hits = res['stats']['total_hits']
        
        self.restaurants = res['results']
        
        # if total_results are less than 500 (maximum that can searched)
        # parse all pages
        
        if total_results < 500:
            print('* Parsing main page...')
            resturants_to_append = self.searchByCategory(page_size, 2)
            self.restaurants.extend(resturants_to_append)
            
        # otherwise filter by each tipe of cuisine
        else:
            for cuisine in self.CUISINES:
                print('** Search for ' + cuisine + ' restaurants...')
                res = self.searchPage(order_method, cuisine = cuisine)
                total_results = res['stats']['total_results']
                print('** Found ' + str(total_results) + ' results.')
                result_count = res['stats']['result_count']
                page_size = res['stats']['page_size']
                total_hits = res['stats']['total_hits']
                
                self.restaurants.extend(res['results'])
                if total_results <= 500:
                    resturants_to_append = self.searchByCategory(page_size, 2, cuisine = cuisine)
                    self.restaurants.extend(resturants_to_append)
                    
                else:
                    for sort_mode in self.SORT_MODES[1:]:
                        resturants_to_append = self.searchByCategory(page_size, 1, order_method, sort_mode = sort_mode, cuisine = cuisine)
                        self.restaurants.append(resturants_to_append)
    
        
    def createDataFrame(self, date):
        
        restaurants = self.restaurants
        
        df = pd.DataFrame.from_records(restaurants, exclude = ['logo', 'description',
            'delivery_fee', 'service_fee', 'delivery_minimum', 'menu_items', 'phone_only',
            'coupons_available', 'coupons_count', 'first_coupon', 'track_your_grub', 'accepts_credit',
            'accepts_cash', 'highlighting_info', 'time_zone',  'real_time_eta', 'delivery_fee_without_discounts',
            'delivery_fee_percent', 'restaurant_cdn_image_url', 'media_image', 'custom_search_tags', 'recommended',
            'rank', 'open', 'next_open_at', 'next_open_at_pickup', 'next_delivery_time',
            'next_pickup_time', 'next_closed_at', 'next_closed_at_pickup', 'inundated', 'soft_blackouted',
            'available_hours', 'override_hours', 'percentage_ad_fee', 'go_to', 'popular_at_your_company',
            'just_in_time_orders', 'sales_force_group', 'pickup_estimate_info',
            'offers_background_color', 'brand_color', 'venue', 'matching_brand_restaurants', 'participants_max',
            'non_supplemental_delivery', 'non_supplemental_open_closed_container', 'supplemental_delivery',
            'supplemental_open_closed_container', 'new_restaurant', 'vendor_location_id', 'curbside_pickup_instructions'])
        
        df = pd.concat((df, pd.DataFrame.from_records(df['address'], exclude = ['address_country'])), axis=1)
        df.pop('address')
        df = pd.concat((df, pd.DataFrame.from_records(df['ratings'], exclude = ['hidden', 'isTooFew'])), axis=1)
        df.pop('ratings')
        df = pd.concat((df, pd.DataFrame.from_records(pd.DataFrame.from_records(df['faceted_rating_data'])['faceted_rating_list'], columns = ['delivery_speed', 'order_accuracy', 'food_quality'])), axis=1)
        df.pop('faceted_rating_data')
        df = pd.concat((df, pd.DataFrame.from_records(pd.DataFrame.from_records(df['price_response'])['delivery_response'])), axis=1)
        df.pop('price_response')
        df['phone_number'] = pd.DataFrame.from_records(df['phone_number'])['phone_number']
        df['routing_number'] = pd.DataFrame.from_records(df['routing_number'])['phone_number']
        
        # df = pd.DataFrame(columns = ['restaurant_id'])
        # for restaurant in restaurants:
        #     data = {}
        #     data = data | restaurant['address']
        #     #data.pop('address_country')
        #     data = data | {'available_offers': restaurant['available_offers']}
        #     data = data | {'available_progress_campaigns': restaurant['available_progress_campaigns']}
        #     data = data | {'available_promo_codes': restaurant['available_promo_codes']}
        #     for badge in restaurant['badge_list']:
        #         data = data | {badge['name']: badge['badge_data'] == 'true'}
        #     data = data | {'brand_id':restaurant['brand_id']}
        #     data = data | {'brand_name':restaurant['brand_name']}
        #     data = data | {'chain_id':restaurant['chain_id']}
        #     data = data | {'chain_name':restaurant['chain_name']}
        #     data = data | {'coupons_count':restaurant['coupons_count']}
        #     data = data | {'cuisines':restaurant['cuisines']}
        #     data = data | {'delivery_mode':restaurant['delivery_mode']}
        #     data = data | {'delivery_time_estimate':restaurant['delivery_time_estimate']}
        #     data = data | {'delivery_time_estimate_lower_bound':restaurant['delivery_time_estimate_lower_bound']}
        #     data = data | {'delivery_time_estimate_upper_bound':restaurant['delivery_time_estimate_upper_bound']}
        #     data = data | {'delivery_type':restaurant['delivery_type']}
        #     data = data | {'distance_from_location':restaurant['distance_from_location']}
        #     for faceted_rating in restaurant['faceted_rating_data']['faceted_rating_list']:
        #         data = data | {faceted_rating['facet_type'] + '_%': faceted_rating['positive_response_percentage']}
        #     data = data | {'merchant_id':restaurant['merchant_id']}
        #     data = data | {'merchant_url_path':restaurant['merchant_url_path']}
        #     data = data | {'new_restaurant':restaurant['new_restaurant']}
        #     data = data | {'open':restaurant['open']}
        #     data = data | {'phone_number':restaurant['phone_number']['phone_number']}
        #     data = data | {'pickup':restaurant['pickup']}
        #     data = data | {'pickup_time_estimate':restaurant['pickup_time_estimate']}
        #     data = data | {'price_rating':restaurant['price_rating']}
        #     data = data | {'pickup':restaurant['pickup']}
        #     data = data | flatten(restaurant['price_response']['delivery_response'], reducer='dot', max_flatten_depth = 2)
        #     #data.pop('pricing_fees')
        #     data = data | {'recommended':restaurant['recommended']}
        #     data = data | flatten(restaurant['ratings'], reducer = 'dot')
        #     data.pop('isTooFew')
        #     data['rating_is_too_few'] = data['too_few']
        #     data.pop('too_few')
        #     data = data | {'total_menu_items': restaurant['total_menu_items']}
        #     data = data | {'restaurant_id': restaurant['restaurant_id']}
        
            # if restaurant['restaurant_id'] not in df['restaurant_id']:
            #     df = df.append(data, ignore_index=True)
        
        df = df.drop_duplicates(subset=['restaurant_id', 'merchant_id'])
        df.to_csv(self.machine + self.folder + self.output_folder + date + '/' + self.file_name + '.csv')

    def getCensusTractsData(self):
        
        path_folder = self.machine +  self.folder + self.output_folder
        
        no_restaurants = {}
        counter=0
        for filename in os.listdir(path_folder):
            
            if Base.getExtension(filename) == 'csv':
                
                census_tract = Base.getCTFromFileName(filename)
                no_restaurants[census_tract] = len(pd.read_csv(path_folder + filename))
                
                counter +=1
                if (counter%100)==0:
                    print(counter)
                    
        census_w_restaurants = pd.DataFrame(no_restaurants.items(), columns=['BoroCT2020', 'NumberGHDeliveringRestaurants'])
        census_w_restaurants.to_csv(self.machine + self.folder + self.analysis_folder + 'delivering_restaurants_in_tracts.csv')
        
    def getUniqueRestaurants(self):
        
        path_folder = self.machine +  self.folder + self.output_folder
        
        cols = ['restaurant_id',
                'merchant_id',
                'name',
                'total_menu_items',
                'cuisines',
                'phone_number',
                'price_rating',
                'delivery_mode',
                'address_locality',
                'address_region',
                'postal_code',
                'street_address',
                'latitude', 
                'longitude',
                'rating_count',
                'rating_value',
                'actual_rating_value',
                'delivery_speed',
                'order_accuracy',
                'food_quality'
                ]
        
        allRestaurants = pd.DataFrame(columns = cols)
        allRestaurants.set_index('restaurant_id', inplace=True)
        counter=0
        for filename in os.listdir(path_folder):
            
            if Base.getExtension(filename) == 'csv':
                
                if len(allRestaurants)==0:
                    allRestaurants = pd.read_csv(path_folder + filename, usecols = cols)
                    
                else:
                    allRestaurants = pd.concat((allRestaurants, pd.read_csv(path_folder + filename, usecols = cols)), axis=0, ignore_index=True)
                
                allRestaurants = allRestaurants.drop_duplicates(subset=['restaurant_id', 'merchant_id'])
                counter +=1
                if (counter%100)==0:
                    print(counter)
    
        
        id_column = 'restaurant_id'
        output_folder = self.machine + self.folder + self.analysis_folder
        
        allRestaurants = Base.findCensusTracts(self, allRestaurants, id_column, output_folder)
        allRestaurants.to_csv(self.machine + self.folder + self.analysis_folder + 'restaurants_delivering_from_tracts.csv')
        
        self.GH_data = allRestaurants
    
    def getNetworkRestaurantData(self):
        
        path_folder = self.machine +  self.folder + self.output_folder
        
        cols = ['restaurant_id', 'merchant_id', 'pickup', 'pickup_time_estimate', 'distance_from_location', 'delivery_time_estimate',
                'delivery_time_estimate_lower_bound', 'delivery_time_estimate_upper_bound', 'delivery_fee', 'service_fee']
        cols_ = cols.copy()
        cols_.append('census_tract')
        allRestaurants = pd.DataFrame(columns = cols_)
        allRestaurants.set_index('restaurant_id', inplace=True)
        counter=0
        for filename in os.listdir(path_folder):
            
            
            if Base.getExtension(filename) == 'csv':
            
                census_tract = Base.getCTFromFileName(filename)
                restaurants_CT = pd.read_csv(path_folder + filename, usecols=cols)
                restaurants_CT['delivery_fee'] = restaurants_CT['delivery_fee'].apply(ast.literal_eval).apply(pd.Series)['flat_cents']
                restaurants_CT['service_fee'] = restaurants_CT['service_fee'].apply(ast.literal_eval).apply(pd.Series)['basis_points']
                restaurants_CT['census_tract'] = census_tract
                allRestaurants = pd.concat((allRestaurants, restaurants_CT), axis=0, ignore_index=True)
            
                #allRestaurants = allRestaurants.drop_duplicates(subset=['restaurant_id', 'merchant_id'])
            counter +=1
            if (counter%100)==0:
                print(counter)

        
        id_column = 'restaurant_id'
        output_folder = self.machine + self.folder + self.analysis_folder
        allRestaurants = Base.findCensusTracts(self, allRestaurants, id_column, output_folder)
        allRestaurants.to_csv(self.machine + self.folder + self.analysis_folder + 'census-specific-data.csv')
        
    def writeSummaryUnique(self):
        
        boros = ['Manhattan',
                 'Brooklyn',
                 'Queens',
                 'Bronx',
                 'Staten Island']
        
        all_rests = pd.read_csv(self.machine + self.folder + self.analysis_folder + 'restaurants_delivering_from_tracts.csv')
        merged_df = pd.read_csv(self.machine + 'data/' + 'new_merged_dataset.csv', index_col=[0])
        merged_df = merged_df[['restaurant_id','BoroName']].merge(all_rests,on='restaurant_id')
        merged_df = merged_df.join(pd.get_dummies(merged_df.BoroName).groupby(level=0).sum())
        varss = cols = boros + ['price_rating',
                                'rating_value',
                                'rating_count',
                                'delivery_speed',
                                'order_accuracy',
                                'food_quality',
                                'total_menu_items',
                                ]
        
        short_df = merged_df[varss]
        
        # There was an error
        
        for idx in short_df.index:
            for facet in ['delivery_speed', 'order_accuracy', 'food_quality']:
                if (pd.isnull(short_df.loc[idx,facet])==False) & (isinstance(short_df.loc[idx,facet], float)==False):
                    dict_facet = ast.literal_eval(short_df.loc[idx,facet])
                    short_df.loc[idx,dict_facet['facet_type']] = dict_facet['positive_response_percentage']
        for idx in short_df.index:
            for facet in ['delivery_speed', 'order_accuracy', 'food_quality']:
                if (isinstance(short_df.loc[idx,facet], float)==False):
                    short_df.loc[idx,facet] = np.nan
        
        short_df = short_df.rename(
                            {'price_rating': 'Price rating',
                            'rating_count': 'Rating count',
                            'rating_value': 'Rating',
                            'delivery_speed': 'Delivery speed',
                            'order_accuracy': 'Order accuracy',
                            'food_quality': 'Food quality',
                            'total_menu_items': 'Total menu items'}, axis=1)
        
        
        short_df=short_df.astype(float)
        summary = {}
        for var in short_df.columns:
            summary[var] = [np.mean(short_df[var]), np.std(short_df[var]), np.min(short_df[var]), np.max(short_df[var])]
            
        summary = pd.DataFrame(summary).T
        summary = summary.rename({0:'Mean',1:'Std. dev.', 2:'Min', 3:'Max'}, axis=1)
        summary = summary.round(2)
        summary['Min'] = summary['Min'].map('{:.0f}'.format)
        summary['Max'] = summary['Max'].map('{:.0f}'.format)
        #summary.loc['Rating count',:] = summary.loc['Rating count',:].map('{:.0f}'.format)
        #summary.loc['Total menu items',:] = summary.loc['Total menu items',:].map('{:.0f}'.format)
        file_name = 'tables/summary_gh.tex'
        tex_file = open(file_name, 'w')
        tex_file.write(summary.to_latex(sparsify = True,
                                        caption = '',
                                        label = 'tab:summary_yelp'))
        tex_file.close()
            
            
    def computeDistanceTimeMatrices(self):
        
        rests_network =  pd.read_csv(self.machine + self.folder + self.analysis_folder + 'census-specific-data.csv', index_col=[0])
        rests_data = pd.read_csv(self.machine + self.folder + self.analysis_folder + 'restaurants_delivering_from_tracts.csv')
        #census_df = GeoData().census_df.copy().reset_index()
        #census_df.rename({'BoroCT2020':'census_tract'},axis=1,inplace=True)
        #rests_network = rests_network.merge(census_df[['BoroCode','census_tract']], on = 'census_tract')
        rests_network['restaurant_id'] = rests_network['restaurant_id'].astype(int)
        rests_network = rests_data[['restaurant_id', 'BoroCT2020']].merge(rests_network, on = 'restaurant_id', how='right')#, rsuffix='_')
        #rests_network.loc[np.isnan(rests_network['BoroCT2020']),'BoroCT2020'] = '000000'+rests_network.loc[np.isnan(rests_network['BoroCT2020']),'BoroCode'].astype(str)
        
        missing_rests=rests_network.loc[np.isnan(rests_network['BoroCT2020'])]
        #missing_rests.to_csv(self.machine + self.folder + self.analysis_folder + 'restaurant_delivering_from_outside_NYC.csv')

        
        rests_network = rests_network.loc[np.isnan(rests_network['BoroCT2020'])==False]
        rests_network['BoroCT2020'] = rests_network['BoroCT2020'].astype(int)
        distancetable=pd.crosstab(rests_network['BoroCT2020'], rests_network['census_tract'], values = rests_network['distance_from_location'], aggfunc=np.nanmedian)
        #rests_network['time_estimate'] = rests_network['delivery_time_estimate_upper_bound']-rests_network['pickup_time_estimate']
        #rests_network = rests_network.loc[rests_network['time_estimate']>0]
        
        timetable=pd.crosstab(rests_network['BoroCT2020'], rests_network['census_tract'], values = rests_network['delivery_time_estimate'], aggfunc=np.nanmedian)


    def findDeliveringRestaurantsInYourArea(self):
        
        merged = pd.read_csv('data/new_merged_dataset.csv', index_col=[0])
        ids = merged.restaurant_id.dropna().astype(int)
        cts = pd.unique(merged.BoroCT2020.dropna().astype(int))
        delivering = pd.DataFrame(np.zeros((len(ids),len(cts))),index=ids, columns=cts)
        
        cols = ['restaurant_id', 'merchant_id']
        path_folder = self.machine +  self.folder + self.output_folder
        counter=0
        for filename in os.listdir(path_folder):
           
           if Base.getExtension(filename) == 'csv':
               
               rests_ct = pd.read_csv(path_folder + filename, usecols = cols)
               common_rests = np.intersect1d(rests_ct.restaurant_id, delivering.index)
               census_tract = Base.getCTFromFileName(filename)
               delivering.loc[common_rests,census_tract] = 1
               
               counter +=1
               if (counter%100)==0:
                   print(counter)
                   
        delivering=delivering.replace(np.nan,0)
        delivering.to_csv(self.machine + self.folder + self.analysis_folder + 'where_restaurants_deliver_by_GH_id.csv')
        delivering = delivering.merge(merged[['id', 'restaurant_id']].set_index('restaurant_id'), on = 'restaurant_id')
        delivering.set_index('id', inplace=True)
        #delivering.drop('restaurant_id', inplace=True)
        delivering.to_csv(self.machine + self.folder + self.analysis_folder + 'where_restaurants_deliver_by_Yelp_id.csv')
        
    
    def identifyDeliveryRestaurants(self):
        
        # dataframe of ALL restaurants on GH and where they deliver TO
        delivery_network_all = pd.read_csv(self.machine + self.folder + self.analysis_folder + 'census-specific-data.csv', usecols = ['merchant_id', 'restaurant_id', 'census_tract'])
        # dataframe of ALL restaurants on GH and where they are LOCATED
        rests_location = pd.read_csv(self.machine + self.folder + self.analysis_folder + 'restaurants_delivering_from_tracts.csv', usecols = ['merchant_id', 'BoroCT2020'])
        # merge the two datasets above
        delivery_network_all = delivery_network_all.merge(rests_location, on = 'merchant_id', how = 'left')
        
        # adjacency matrix of delivery network (=1 if delivers to the column CT)
        dummies_del = pd.get_dummies(delivery_network_all['census_tract'])
        dummies_del.columns = dummies_del.columns.astype(int)
        dummies_del = dummies_del.sort_index(axis=1)
        
        dummies_del = dd.from_pandas(dummies_del, chunksize=50000)
        
        # this matrix says where each rest delivers to
        delivery_network = dummies_del.groupby('merchant_id').sum().compute()
        delivery_network.columns = np.genfromtxt(self.machine + self.folder + self.network_folder + 'CT_names.txt')
        delivery_network.columns = delivery_network.columns.astype(int)

        # join matrix with location census tract
        rests_location = pd.read_csv(self.machine + self.folder + self.analysis_folder + 'restaurants_delivering_from_tracts.csv', usecols = ['merchant_id', 'BoroCT2020'])
        delivery_network = delivery_network.merge(rests_location, on = 'merchant_id')
        delivery_network = delivery_network.set_index('merchant_id')
        
        # save 
        delivery_network.to_csv(self.machine + self.folder + self.network_folder + 'where_firms_deliver_by_restaurant_id.csv')
        
        delivery_mode_est = pd.DataFrame(delivery_network['BoroCT2020'].copy(), columns = ['BoroCT2020'])
        delivery_mode_est.loc[:,'EstimatedPlatformDelivery'] = False
        
        CT_names = np.genfromtxt(self.machine + self.folder + self.network_folder + 'CT_names.txt')
        
        GD = GeoData()
        
        touching_CT = pd.read_csv(GD.machine + GD.census_folder + 'neighboring_census_tracts_1.csv')
        touching_CT_2 = pd.read_csv(GD.machine + GD.census_folder + 'neighboring_census_tracts_2.csv')
        
        for CT in CT_names:
            
            CT = int(CT)
            
            df_CT = delivery_network.loc[delivery_network.BoroCT2020==CT,:]
            
            if len(df_CT) == 0:
                
                continue
            
            if len(df_CT) < 20:
                
                touch_CT = touching_CT.loc[touching_CT.BoroCT2020==CT, :]==1
                touch_CT = touch_CT.T
                touch_CT = list(touch_CT.loc[touch_CT[touch_CT.columns[0]]==1].index)
                touch_CT = np.array(touch_CT).astype(int)
                
                df_CT = delivery_network.loc[delivery_network.BoroCT2020.isin(touch_CT),:]
                
                if len(df_CT) < 20:
                    f
                    touch_CT = touching_CT_2.loc[touching_CT.BoroCT2020==CT, :]==1
                    touch_CT = touch_CT.T
                    touch_CT = list(touch_CT.loc[touch_CT[touch_CT.columns[0]]==1].index)
                    touch_CT = np.array(touch_CT).astype(int)
                    
                    df_CT = delivery_network.loc[delivery_network.BoroCT2020.isin(touch_CT),:]
                   
                    
            df_CT_ = df_CT.copy()
            df_CT_ = df_CT_.loc[:,df_CT_.columns != 'BoroCT2020']
        
            w = pdist(df_CT_.div(df_CT_.sum(1), axis=0))
            ww = squareform(w)
            www = ww[df_CT_.index.isin(df_CT.loc[df_CT.BoroCT2020==CT].index),:]
            
            wwww = squareform(pdist(www))
            G = nx_from_numpy_array(wwww<0.1)
            
            max_cc = max(connected_components(G))
            
            delivery_mode_est.loc[df_CT.loc[df_CT.BoroCT2020==CT].index[list(max_cc)], 'EstimatedPlatformDelivery']=True
            
        delivery_mode_est.to_csv(self.machine + self.folder + self.analysis_folder + 'restaurants_platform_delivery.csv')
        
    
        
    # create and save the following matrices:
    # dummies_CT: where rests. on the platform are located
    # dummies_del: where rests. on the platform deliver
    # dummies_CT_del: how many rests. in one CT deliver to a CT
    def saveDummiesMatrices(self):
        
        # dataframe of ALL restaurants on GH and where they deliver TO
        delivery_network_all = pd.read_csv(self.machine + self.folder + self.analysis_folder + 'census-specific-data.csv', usecols = ['merchant_id', 'restaurant_id', 'census_tract'])
        # dataframe of ALL restaurants on GH and where they are LOCATED
        rests_location = pd.read_csv(self.machine + self.folder + self.analysis_folder + 'restaurants_delivering_from_tracts.csv', usecols = ['merchant_id', 'BoroCT2020'])
        # merge the two datasets above
        delivery_network_all = delivery_network_all.merge(rests_location, on = 'merchant_id', how = 'left')
        del rests_location
        
        # adjacency matrix of delivery network (=1 if delivers to the column CT)
        dummies_del = pd.get_dummies(delivery_network_all['census_tract'])
        dummies_del.columns = dummies_del.columns.astype(int)
        dummies_del = dummies_del.sort_index(axis=1)
        all_cols = dummies_del.columns
        #dummies_del = csr_matrix(dummies_del)
        dummies_del.to_csv(self.machine + self.folder + self.network_folder + 'dummies_pair_delivery_all.csv')

        
        # adjacency matrix of locations (=1 if located in the column CT)
        dummies_CT = pd.get_dummies(delivery_network_all['BoroCT2020'])
        
        # save column of rests id
        delivery_network_all['merchant_id'].to_csv(self.machine + self.folder + self.network_folder + 'rests_merchant_id_all.csv')
        
        
        
        dummies_CT.columns = dummies_CT.columns.astype(int)
        # CTs that receive deliveries but that do not contain any delivering restaurant
        diff_cols = np.setdiff1d(all_cols, dummies_CT.columns)
        
        
        
        
        
        
        dummies_CT = pd.concat((dummies_CT, pd.DataFrame([np.zeros(len(diff_cols))], columns = diff_cols, index = dummies_CT.index)), axis=1)
        dummies_CT = dummies_CT.sort_index(axis=1)
        # save names of CTs (important since sparse matrices lose names)
        np.savetxt(self.machine + self.folder + self.network_folder + 'CT_names.txt', np.array(list(dummies_CT.columns)))
        # sparsify matrices above
        #dummies_CT = csr_matrix(dummies_CT)
        dummies_CT.to_csv(self.machine + self.folder + self.network_folder + 'dummies_pair_CT_all.csv')
  
        
        
        # smaller adjacency matrix where entry jk = number of restaurants in CT j delivering to CT k
        dummies_CT_del = dummies_CT.T.dot(dummies_del)
        dummies_CT_del.to_csv(self.machine + self.folder + self.network_folder + 'dummies_CT_pair_delivery_all.csv')
        
        
        
        
        
        
        delivery_mode_est = pd.read_csv(self.machine + self.folder + self.analysis_folder + 'restaurants_platform_delivery.csv')
        delivery_network_all = pd.read_csv(self.machine + self.folder + self.analysis_folder + 'census-specific-data.csv', usecols = ['merchant_id', 'restaurant_id', 'census_tract'])
        delivery_network_full = delivery_network_all.merge(delivery_mode_est, on = 'merchant_id', how = 'left')
        del delivery_network_all
        
        delivery_network_full = delivery_network_full.loc[delivery_network_full.EstimatedPlatformDelivery==True]
        
        # adjacency matrix of delivery network (=1 if delivers to the column CT)
        dummies_del = pd.get_dummies(delivery_network_full['census_tract'])
        dummies_del.columns = dummies_del.columns.astype(int)
        dummies_del = dummies_del.sort_index(axis=1)
        all_cols = dummies_del.columns
        dummies_del = csr_matrix(dummies_del)
        save_npz(self.machine + self.folder + self.network_folder + 'dummies_pair_delivery_full.csv', dummies_del)

        
        # adjacency matrix of locations (=1 if located in the column CT)
        dummies_CT = pd.get_dummies(delivery_network_full['BoroCT2020'])
        
        # save column of rests id
        delivery_network_full['merchant_id'].to_csv(self.machine + self.folder + self.network_folder + 'rests_merchant_id_full.csv')
        
        
        
        dummies_CT.columns = dummies_CT.columns.astype(int)
        # CTs that receive deliveries but that do not contain any delivering restaurant
        diff_cols = np.setdiff1d(all_cols, dummies_CT.columns)
        
        
        
        
        
        
        dummies_CT = pd.concat((dummies_CT, pd.DataFrame([np.zeros(len(diff_cols))], columns = diff_cols, index = dummies_CT.index)), axis=1)
        dummies_CT = dummies_CT.sort_index(axis=1)
        # save names of CTs (important since sparse matrices lose names)
        np.savetxt(self.machine + self.folder + self.network_folder + 'CT_names.txt', np.array(list(dummies_CT.columns)))
        # sparsify matrices above
        dummies_CT = csr_matrix(dummies_CT)
        
        
        # save sparse matrices
        save_npz(self.machine + self.folder + self.network_folder + 'dummies_pair_delivery_full.npz', dummies_del)
        
        # smaller adjacency matrix where entry jk = number of restaurants in CT j delivering to CT k
        dummies_CT_del = dummies_CT.T.dot(dummies_del)
        save_npz(self.machine + self.folder + self.network_folder + 'dummies_CT_pair_delivery_full.npz', dummies_CT_del)
        
    def findDeliveryAreas(self):
         
         # dataframe mapping indices to census tracts
         CT_names = np.genfromtxt(self.machine + self.folder + self.network_folder + 'CT_names.txt')
         CT_names = pd.DataFrame(CT_names).reset_index()
         CT_names.rename({'index': 'CT_idx', 0: 'BoroCT2020'}, axis=1, inplace=True)
         CT_names = CT_names.astype(int)
         # load sparse matrix where entry jk = number of restaurants in CT j delivering to CT k
         dummies_CT_del = load_npz(self.machine + self.folder + self.network_folder + 'dummies_CT_pair_delivery_full.npz')
         
         # define network from such adjacency matrix
         G_CT_del = nx_from_scipy_sparse_array(dummies_CT_del)
         
         def findCommunities(resolution):
             # find communities by minimizing modularity
             n_clusters = 5
             #communities = nx_comm.louvain_communities(G_CT_del, resolution = resolution)#, weight='weight')
             communities = nx_comm.greedy_modularity_communities(G_CT_del, cutoff = 4, resolution = resolution, weight='weight') 
             modularity = nx_comm.modularity(G_CT_del, communities, weight = 'weight')
             # define dataframe 
             communities_mat = np.zeros([dummies_CT_del.shape[0], 2])
             
             idx=0
             for c in range(len(communities)):
                 communities_mat[idx:idx+len(communities[c]),0] = np.array(list(communities[c]))
                 communities_mat[idx:idx+len(communities[c]),1] = c+1
                 idx = idx+len(communities[c])
                 
             communities_df = pd.DataFrame(communities_mat, columns = ['CT_idx', 'community'])
             # convert dfs values to ints
             communities_df = communities_df.astype(int)
             
             communities_df = communities_df.merge(CT_names, on = 'CT_idx').astype(int)
             communities_df.to_csv(self.machine + self.folder + self.network_folder + 'communities_CT_weighted_full_' + str(resolution) + '.csv')
             
             return modularity
             
         # save modularities
         mods = []
         for r in [0.75, 1, 1.25, 1.5]:
         
            mods.append(findCommunities(r))
            
         np.savetxt(self.machine + self.folder + self.network_folder + 'weighted_modularities.txt', np.array(mods))
    
    def computeCensusSpecificStatistics(self):
        
        # 
        delivery_mode_est = pd.read_csv(self.machine + self.folder + self.analysis_folder + 'restaurants_platform_delivery.csv')
        
        delivery_network_all = pd.read_csv(self.machine + self.folder + self.analysis_folder + 'census-specific-data.csv', usecols = ['merchant_id', 'restaurant_id', 'distance_from_location', 'census_tract'])
        delivery_network_all = delivery_network_all.merge(delivery_mode_est, on = 'merchant_id', how = 'left')
        delivery_network_all = delivery_network_all.merge(GeoData().census_df, on = 'BoroCT2020', how = 'left')
        delivery_network_all['Delivery_area'] = delivery_network_all['BoroName']
        delivery_network_all.loc[(delivery_network_all['Delivery_area'] == 'Brooklyn') | (delivery_network_all['Delivery_area'] == 'Queens'), 'Delivery_area'] = 'BrooklynQueens'
        delivery_network_full = delivery_network_all.loc[delivery_network_all.EstimatedPlatformDelivery==True]

        see = delivery_network_all.groupby('BoroName').median()
        see = delivery_network_full.groupby('BoroName').median()