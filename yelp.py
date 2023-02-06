#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 02:21:13 2022

@author: michelev
"""


# pip install beautifulsoup4
# pip install requests soupsieve lxml

# Modules used for scraping

# from bs4 import BeautifulSoup
# from urllib.request import Request, urlopen
#import json
# import ssl

import os
import Base

from statsmodels.iolib.table import SimpleTable

from bs4 import BeautifulSoup
import requests
import lxml
import re
import json
import urllib
import math

import ast

import numpy as np
import pandas as pd
import geopandas as gpd

import networkx as nx
from shapely.geometry import Point, MultiPoint
import shapely.vectorized
import shapely

from scipy.sparse import csr_matrix, find

import matplotlib.pyplot as plt

import time
#from datetime import date

from scipy.stats import norm
from sklearn.neighbors import KernelDensity

import scipy
import seaborn as sns

import pysal
from pysal.lib import cg as geometry
from pysal.lib import weights

from pyproj import Proj
from sklearn.cluster import KMeans
from concurrent.futures import ThreadPoolExecutor

from Scraper import getResponseProxies


from datetime import datetime

#%%

class YelpClient():
    
    machine = '/Users/michelev/spatial-competition-food/'
    output_folder = 'yelp_data/'
    menu_subfolder = output_folder + 'menus/'
    
    census_folder = 'nyc_geodata/census_tracts_boundaries/'
    census_filename = 'census_tracts.shp'
    centroids_filename = 'census_tracts_centroids.shp'
    census_gpd = gpd.read_file(machine + census_folder + census_filename)
    categories_strings = None
    distances_folder = 'nyc_geodata/distances/'
    
    network_folder = 'network_data/'

    # initialize priority of the use of the API keys and retrieve dataframe of keys
    priority = 1
    api_keys = pd.read_csv(machine + output_folder + '/yelp_keys.csv')
    api_token = None
    
    # keep track of the response to handle errors
    response = None
    
    boroughs = ['Manhattan', 'Brooklyn', 'Queens', 'Bronx', 'Staten Island']
    neighborhoods_dict = {}    

    headers = {'User-Agent':'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/105.0.0.0 Safari/537.3'}

    # URL to pull data from:
    url = 'https://api.yelp.com/v3/businesses/search'
    url_params = None
    location = None
    term = None
    search_limit = None
    offset = None
    categories = None
    sort_by = None
    
    nyc_b_n = None
        
    # file=open(machine + output_folder + 'categories.json')
    # categories_dict = json.loads(file.read())
    # categories_pd = pd.DataFrame.from_dict(categories_dict)
    # parents = ['restaurants', 'african', 'arabian', 'belgian', 'brazilian','breakfast_brunch',
    #            'cafes', 'caribbean', 'chhinese', 'french', 'german',
    #            'italian', 'japanese', 'latin', 'malaysian', 'mediterrean', 'mexican',
    #            'mideastern', 'polish', 'portuguese', 'spanish', 'turkish', 'turkish']
    
    # CATEGORIES = []
    # for j in categories_pd.index:
        
    #     if len(categories_pd.loc[j, 'parents']) > 0:
    #         if categories_pd.loc[j, 'parents'][0] in parents:
    #             CATEGORIES.append(categories_pd.loc[j, 'alias'])
    # CATEGORIES.append('restaurants')
    
    CATEGORIES =   ['latin',
                    'cantonese',
                    'bakeries', # in food
                    'bagels', # in food
                    'turkish',
                    #'foodtrucks',
                    'bbq',
                    #'catering',
                    #'food_court',
                    'hotdogs',
                    #'beer_and_wine',
                    'mideastern',
                    'creperies',
                    #'cocktailbars',
                    #'wine_bars',
                    #'pastashops',
                    'buffets',
                    'irish',
                    'soulfood',
                    #'juicebars',
                    'cajun',
                    'tradamerican',
                    'spanish',
                    'poke', # in food
                    'filipino',
                    'waffles',
                    'sushi',
                    'pakistani',
                    #'fooddeliveryservices',
                    'mediterranean',
                    'szechuan',
                    'burgers',
                    'halal',
                    'japanese',
                    'soup',
                    'ramen',
                    'tapas',
                    'chicken_wings',
                    'cheesesteaks',
                    'salad',
                    'seafood',
                    'african',
                    'empanadas',
                    'indpak',
                    'himalayan',
                    'tacos',
                    'bubbletea', # in food
                    #'cafes',
                    #'sportsbars',
                    'falafel',
                    'tex-mex',
                    'vegetarian',
                    'shopping',
                    'chickenshop',
                    'icecream', # in food
                    'pizza',
                    'french',
                    'sandwiches', # in food
                    'newamerican',
                    'desserts', # in food
                    #'streetvendors',
                    'food', # in food
                    'colombian',
                    'peruvian',
                    #'coffee',
                    'thai',
                    'dimsum',
                    #'convenience',
                    #'nightlife',
                    'chinese',
                    'diners', # in food
                    'breakfast_brunch',
                    'cuban',
                    'southern',
                    'gastropubs',
                    'gluten_free',
                    #'lounges',
                    #'beerbar',
                    'restaurants',
                    'greek',
                    'delis', # in food
                    'noodles',
                    'asianfusion',
                    'hotdog',
                    'mexican',
                    #'eventservices',
                    'caribbean',
                    #'venues',
                    #'grocery',
                    'gourmet',
                    'dominican',
                    #'foodstands',
                    'steak',
                    'wraps',
                    #'bars',
                    'korean',
                    'kosher',
                    'pubs', # in food
                    'italian',
                    'comfortfood',
                    #'arts',
                    'vegan',
                    'tapasmallplates',
                    'vietnamese']

    def findCategories():
        headers = {'User-Agent':'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/105.0.0.0 Safari/537.3'}
        url = 'https://www.yelp.com/search?find_desc=Restaurants&find_loc=New+York%2C+NY'
        response = requests.get(url, headers = headers)
        
        soup = BeautifulSoup(response.content, 'lxml')
        links = soup.find_all('script', type='application/json')
        page = json.loads(re.search('\{.+\}', str(links[0])).group())
        filters = page['legacyProps']['searchAppProps']['searchPageProps']['filterPanelProps']['filterInfoMap']
        
        categories_rests = []
        for key in filters.keys():
          if filters[key]['name'] == 'category':
              categories_rests.append(filters[key]['value'])
              
    def updateDate(self, date):
        
        self.date = date
        
    def findEfficientCategories(self):
        
        # In the piece of code below I aggregate catgories depending on how frequent they are,
        # in order to make less requests as possible in the Yelp API
           
        # Below very rare categories 
            # nyc_totals contained all the restaurants in NYC found through the API
            
            
            # categories_ = categories_rests.copy()
            
            # for k in range(5):
            #     for j in range(len(nyc_totals[k])):
                    
            #         if len(nyc_totals[k][j])>0:
            #             dict_ = nyc_totals[k][j][1]
            #             for i in range(len(categories_rests)):
            #                 if categories_rests[i] in dict_.keys():
            #                     if dict_[categories_rests[i]] > 20:
            #                         if categories_rests[i] in categories_:
            #                             categories_.remove(categories_rests[i])
        
        categories_few = ['afghani', 'african', 'armenian', 'austrian', 'basque', 
                          'belgian', 'bulgarian', 'burmese', 'cajun', 'cambodian',
                          'catalan', 'comfortfood', 'czech', 'eritrean', 'ethiopian',
                          'filipino', 'georgian', 'german', 'greek', 'guamanian',
                          'hawaiian', 'himalayan', 'honduran', 'hungarian', 'iberian',
                          'indonesian', 'international', 'irish', 'kosher', 'laotian',
                          'malaysian', 'modern_european', 'mongolian', 'moroccan', 'newmexican',
                          'nicaraguan', 'persian', 'polish', 'polynesian', 'popuprestaurants',
                           'portuguese', 'poutineries', 'raw_food', 'russian', 'scandinavian',
                           'scottish', 'singaporean', 'slovakian', 'somali', 'soulfood',
                           'southern', 'spanish', 'srilankan', 'supperclubs', 'syrian',
                           'taiwanese', 'tapas', 'tex-mex', 'ukrainian', 'uzbek',
                           'vietnamese', 'wraps']
        
        # Below categories I aggregate because they appear often together
        
            # categories_ = categories_many.copy()
            
            # for k in range(5):
            #     for j in range(len(nyc_totals[k])):
                    
            #         if len(nyc_totals[k][j])>0:
            #             dict_ = nyc_totals[k][j][1]
            #             for i in range(len(categories_many)):
            #                 if categories_many[i] in dict_.keys():
            #                     if dict_[categories_many[i]] > 333:
            #                         if categories_many[i] in categories_:
            #                             categories_.remove(categories_many[i])
            
        american = ['australian', 'british', 'burgers', 'cheesesteaks', 'chicken_wings',
                    'diners', 'dinnertheater', 'fishnchips', 'food_court', 'gamemeat',
                    'steak', 'newamerican' 'tradamerican']
        asian_ch = ['asianfusion', 'chinese', 'chickenshop', 'hkcafe', 'hotpot',
                    'noodles', 'panasian', 'soup', 'wok']
        cafe = ['breakfast_brunch', 'cafes', 'cafeteria', 'gastropubs', 'tapasmallplates', 'waffles']
        deli = ['buffets', 'delis', 'foodstands', 'hotdog', 'hotdogs', 'sandwiches']
        east = ['arabian', 'bangladeshi', 'halal', 'indpak', 'kebab',
                'mideastern', 'pakistani', 'turkish']
        french = ['bistros', 'brasseries', 'creperies', 'fondue', 'french']
        latin = ['argentine', 'bbq', 'brazilian', 'caribbean', 'cuban', 'latin','peruvian']
        veg = ['gluten_free', 'salad', 'vegan', 'vegetarian']
        
        # Remaining categories are the ones with many search results
        categories_many = ['italian', 'japanese', 'sushi', 'korean', 'mediterranean',
                           'mexican', 'pizza', 'seafood', 'thai']

        # Create list of lists of categories
        categories_few_groups = [categories_few[:30], categories_few[30:]]
        categories_few_groups.append(american)
        categories_few_groups.append(asian_ch)
        categories_few_groups.append(cafe)
        categories_few_groups.append(deli)
        categories_few_groups.append(east)
        categories_few_groups.append(french)
        categories_few_groups.append(latin)
        categories_few_groups.append(veg)
        
        # Create list of strings in the format necessary to make a Yelp request
        categories_few_strings = []
        for j in range(len(categories_few_groups)):
            group_string = '(' + categories_few_groups[j][0]
            for k in range(1, len(categories_few_groups[j])):
                group_string = group_string + ',' + categories_few_groups[j][k]
            group_string = group_string + ')'
            categories_few_strings.append(group_string)
            
        categories_strings = [*categories_few_strings, *categories_many]
        
        self.categories_strings = categories_strings
        
        return categories_strings
            
    def updateClient(self):
        
        self.api_token = self.api_keys[(self.api_keys['API'] == 'Yelp') & (self.api_keys['priority'] == self.priority)]['client_secret'].values[0]
        self.headers = {'Authorization': 'Bearer {}'.format(self.api_token)}
            
    def refreshResponse(self):
    
        self.priority += 1
        print("Passing to api key with priority " + str(self.priority))
        if self.priority <= 5:
            api_token = self.api_keys[(self.api_keys['API'] == 'Yelp') & (self.api_keys['priority'] == self.priority)]['client_secret'].values[0]
            headers = {'Authorization': 'Bearer {}'.format(api_token)}
            self.response = requests.get(self.url, headers=self.headers, params=self.url_params)
        else:
            self.priority = 1
    
    def updateURLParams(self):
        
        self.url_params = {
            'location': self.location,
            'term' : self.term,
            'limit': self.search_limit,
            'offset': self.offset,
            'categories': self.categories,
            'sorty_by': self.sort_by,
        }
        
    def findNeighborhoods(self):
        
        
    
    
        bb = 0
        
        # for every borough
        for b in self.boroughs:
            
            print('* Borough: ' + b + ' - ' + str(bb+1) + '/'  + str(len(self.boroughs)))
            url = 'https://www.yelp.com/search?find_desc=Restaurants&find_loc=' + b.replace(' ', '+') + '%2C+NY'
            self.response = requests.get(url, headers = self.headers)
        
            soup = BeautifulSoup(self.response.content, 'lxml')
            links = soup.find_all('script', type='application/json')
            page = json.loads(re.search('\{.+\}', str(links[0])).group())
            filters = page['legacyProps']['searchAppProps']['searchPageProps']['filterPanelProps']['filterInfoMap']
            
            neighborhoods = []
            for key in filters.keys():
                if re.match('^NY:New_York:' + b.replace(' ', '_'), key):
                    neighborhoods.append(filters[key]['text'])
    
            self.neighborhoods_dict[b] = neighborhoods
            
            
            bb+=1
            
    def findTotalRestaurants(self):
        
        if not os.path.exists(self.machine + self.output_folder + self.date):
            os.makedirs(self.machine + self.output_folder + self.date)
        #self.nyc_totals_b = [[] for n in range(len(neighborhoods))]
        
        # index keeping track of boroughs
        self.currentBoroughIndex = 0
        # for every borough
        for b in self.boroughs:
            self.currentBorough = b
            print('* Borough: ' + b + ' - ' + str(self.currentBoroughIndex+1) + '/'  + str(len(self.boroughs)))
            self.updateClient()
            neighborhoods = self.neighborhoods_dict[b]
            
            self.currentNeighborhoodIndex=0
            
            for n in neighborhoods:
                
                self.currentNeighborhood = n
                
                if not os.path.exists(self.machine + self.output_folder + self.date + '/neighborhoods/' + self.currentBorough + '_' + self.currentNeighborhood + '.csv'):
                    
                    self.nyc_b_n = []
                    
                    self.findNeighborhoodRestaurants()
                    print('** Neighborhood: ' + n + ', ' + b + ' - ' + str(self.currentNeighborhoodIndex+1) + '/'  + str(len(neighborhoods)))
                    
                    df_nyc_b_n = pd.DataFrame()
                    for res in self.nyc_b_n:
                        df_nyc_b_n = pd.concat((df_nyc_b_n, pd.DataFrame(json.loads(res.text)['businesses'])), axis=0)
                    df_nyc_b_n.set_index('id', inplace=True)
                    df_nyc_b_n.to_csv(self.machine + self.output_folder + self.date + '/neighborhoods/' + self.currentBorough + '_' + self.currentNeighborhood + '.csv')
                    
                
                
                self.currentNeighborhoodIndex += 1
        
            
            self.currentBoroughIndex += 1
            
            
        
    def findNeighborhoodRestaurants(self):
        
        # update borough and neighborhood string/index
        b = self.currentBoroughIndex
        borough = self.currentBorough
        n = self.currentNeighborhoodIndex
        neighborhood = self.currentNeighborhood
        
        #neighborhoods = self.neighborhoods_dict[b]
        
        self.location = neighborhood + ',' + borough + ',NY'
        self.term = "Restaurants"
        self.search_limit = 50
        self.offset = 0
        self.categories = "(restaurants, All)"
        self.sort_by = 'distance'
        self.updateURLParams()
        
        self.response = requests.get(self.url, headers=self.headers, params=self.url_params)
        
        while self.response.status_code != 200:
            self.refreshResponse()
            
        print('*** All Restaurants in {}: #{} - #{} ... {}'.format(neighborhood + ', ' + borough, self.offset+1, self.offset+self.search_limit, self.response.status_code))
        self.nyc_b_n.append(self.response)
        
        tot_restaurants_n = json.loads(self.response.text)['total']
        
        print('**** Total number of Restaurants: ' + str(tot_restaurants_n))
        
        if tot_restaurants_n > 1000:
            
            for cat in self.categories_strings:
                
                self.location = neighborhood + ',' + borough + ',NY'
                self.location = self.location.replace(' ', '+')
                self.offset = 0
                self.categories = cat
                self.updateURLParams()
                
                self.response = requests.get(self.url, headers=self.headers, params=self.url_params)
                
                while self.response.status_code != 200:
                    self.refreshResponse()
          
                    
                print('*** {} Restaurants: #{} - #{} ... {}'.format(cat, self.offset+1, self.offset + self.search_limit, self.response.status_code))
                self.nyc_b_n.append(self.response)
                
                tot_restaurants_c = json.loads(self.response.text)['total']
                print('**** Total number of Restaurants: ' + str(tot_restaurants_c))
                
                max_y = math.ceil(tot_restaurants_c/self.search_limit)
                for y in range(1, min(max_y, 20)):
                
                    self.offset = 50 * y
                    self.updateURLParams()
                    
                    self.response = requests.get(self.url, headers=self.headers, params=self.url_params)
                                
                    while self.response.status_code != 200:
                        self.refreshResponse()
                    
                    self.nyc_b_n.append(self.response)
            
        else:
            
            max_y = math.ceil(tot_restaurants_n/self.search_limit)
            for y in range(1,min(max_y, 20)):
            
                self.offset = 50 * y
                self.updateURLParams()
                self.response = requests.get(self.url, headers=self.headers, params=self.url_params)
                                
                while self.response.status_code != 200:
                    self.refreshResponse()
                
                    
                print('*** All Restaurants in {}: #{} - #{} ... {}'.format(neighborhood + ', ' + borough, self.offset+1, self.offset+self.search_limit, self.response.status_code))
                 
                self.nyc_b_n.append(self.response)
            
    def aggregateData(self, raw=True):
        
        if raw:
            path_folder = self.machine + self.output_folder + self.date + '/neighborhoods/'
        else:
            path_folder = self.machine + self.output_folder + self.date + '/raw_data_with_dates/'
            
        allRestaurants = pd.DataFrame()
        counter=0
        for filename in os.listdir(path_folder):
        
            if Base.getExtension(filename) == 'csv':
            
                restaurants_b = pd.read_csv(path_folder + filename, index_col = [0])
                allRestaurants = pd.concat((allRestaurants, restaurants_b), axis=0)

        allRestaurants.drop_duplicates(subset='alias', inplace=True)

        allRestaurants['date'] = self.date

        if raw:
            allRestaurants.to_csv(self.machine + self.output_folder + self.date + '/yelp_raw_data_' + self.date + '.csv')
        else:
        
            allRestaurants.to_csv(self.machine + self.output_folder + self.date + '/yelp_raw_data_with_dates_' + self.date + '.csv')
            
    def addEstablishmentYears(self, start):
        
        if os.path.exists(self.machine + self.output_folder + self.date + '/raw_data_with_dates/yelp_raw_data_' + self.date + '_' + str(int(start/100)) + '.csv'):
            print('File already exists. Stop iteration.')
            return None
        
        proxies = {'http': 'http://brd-customer-hl_33f39684-zone-zone1:9tzrgl2f2e55@zproxy.lum-superproxy.io:22225',
                   'https': 'http://brd-customer-hl_33f39684-zone-zone1:9tzrgl2f2e55@zproxy.lum-superproxy.io:22225'}

        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/103.0.5060.114 Safari/537.36'}

        allRestaurants = pd.read_csv(self.machine + 'yelp_data/' + self.date + '/yelp_raw_data_' + self.date + '.csv', index_col = [0])
        
        endpoint = min(start+100, len(allRestaurants))
        allRestaurants = allRestaurants.iloc[start : endpoint]
        
        allRestaurants['firstEstimatedEstablishment'] = np.nan
        allRestaurants['lastEstimatedEstablishment'] = np.nan
        
        print('Parsing restaurant # ' + str(start))
        
        for row in allRestaurants.index:
            
            url_name = allRestaurants.loc[row, 'alias']
            
            url = 'https://www.yelp.com/biz/' + url_name + '?sort_by=date_desc'
            
            # get response
            response = getResponseProxies(url, headers, proxies)
            
            # retrieve from script id and close status
            try:
                soup = BeautifulSoup(response.content, 'lxml')
            except:
                time.wait(10)
                soup = BeautifulSoup(response.content, 'lxml')
                
            links_ = soup.find_all('script', type='application/json')
            if len(links_)>0:
                dict_main_ = json.loads(re.search('\{.+\}', str(links_[0])).group())
            else:
                allRestaurants.loc[row, 'firstEstimatedEstablishment'] = None
                allRestaurants.loc[row, 'lastEstimatedEstablishment'] = None
                continue
            
            
            first_est = re.search("(?<=on yelp since ).*?(?=&)", str(soup.contents))
            if first_est == None:
                first_est = re.search("(?<=stablished in ).*?(?=\.)", str(soup.contents))
            if first_est != None:
                first_est = first_est[0]
                
            try:
                first_est = int(first_est)
            except:
                first_est = None
            
            allRestaurants.loc[row,'firstEstimatedEstablishment'] = first_est
                
            links__ = soup.find_all('script', type='application/ld+json')
            for j in range(len(links__)):
                dict_main = json.loads(re.search('\{.+\}', str(links__[j])).group())
                if 'telephone' in dict_main.keys():
                    break
                
            if 'review' in dict_main.keys():
                lastReview=1900
                for j in range(len(dict_main['review'])):
                    
                        tmp = datetime.strptime(dict_main['review'][j]['datePublished'], '%Y-%m-%d').year
                        if tmp>lastReview:
                            lastReview=tmp
                allRestaurants.loc[row, 'lastEstimatedEstablishment'] = lastReview
                if lastReview == 1900:
                    allRestaurants.loc[row, 'lastEstimatedEstablishment'] = None
            else:
                allRestaurants.loc[row, 'lastEstimatedEstablishment'] = None
                
            
        allRestaurants.to_csv(self.machine + self.output_folder + self.date + '/raw_data_with_dates/yelp_raw_data_' + self.date + '_' + str(int(start/100)) + '.csv')
        
    def findMenu(self, start):
         
         proxies = {'http': 'http://brd-customer-hl_33f39684-zone-zone1:9tzrgl2f2e55@zproxy.lum-superproxy.io:22225',
                    'https': 'http://brd-customer-hl_33f39684-zone-zone1:9tzrgl2f2e55@zproxy.lum-superproxy.io:22225'}

         headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/103.0.5060.114 Safari/537.36'}

         allRestaurants = pd.read_csv(self.machine + self.output_folder + self.date + '/yelp_raw_data_with_dates_' + self.date + '.csv', index_col = [0])
         
         endpoint = min(start+100, len(allRestaurants))
         allRestaurants = allRestaurants.iloc[start : endpoint]
         
         print('Parsing restaurant # ' + str(start))
         
         for row in allRestaurants.index:
             
            
                
            short_url = allRestaurants.loc[row, 'alias']
            url = 'https://www.yelp.com/menu/' + short_url
        
            if os.path.exists(self.machine + self.menu_subfolder + short_url + '.csv'):
                print('Menu already processed.')
                continue
            
            response = getResponseProxies(url, headers, proxies)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            menu_dict={'categories': [], 'section': [], 'item': [], 'ingredients': [], 'price': []}

            categories = allRestaurants.loc[row, 'categories']
            
            categories = categories.replace("'", '"')
            categories = re.findall('(?<=alias": ").*?(?=")', categories)
            categories = ' '.join(categories)
            
             
            menu_=soup.find_all('div', attrs={'class':'menu-sections'})
            sections=str(menu_).split('<div class="section-header')
            
            if len(sections)>0:
                
                sections = sections[1:]
                for section in sections:
                    
                    section = BeautifulSoup(section, 'html.parser')
                    
                    section_name = section.findChildren('h2')[0].contents[0].splitlines()[1].lstrip()
                    
                    menu_=section.find_all('div', attrs={'class':'arrange'})
                    
                    
                    for j in range(len(menu_)):
                        
                        
                        menu_item=None
                        menu_a = menu_[j].findChildren('a')
                        for j_a in range(len(menu_a)):
                            
                            
                            if (len(menu_a[j_a])>1) | (str(menu_a[j_a].contents)[0] == '<'):
                                continue
                            else:
                                menu_item =menu_a[j_a].contents[0]
                                break
                        
                        if menu_item == None:
                            menu_item = menu_[j].findChildren('h4')[0].contents[0].splitlines()[1].lstrip()
                        
                            
                        menu_p = menu_[j].findChildren('p')
                        
                        try:
                            menu_ingredients = menu_p[0].contents[0]
                        except:
                            menu_ingredients = None
                            
                        menu_li = menu_[j].findChildren('li')
                        
                        try:
                            menu_price = re.search("\\d.*\\d", menu_li[0].contents[0])[0]
                        except:
                            menu_price = None
                        
                        menu_dict['categories'].append(categories)
                        menu_dict['section'].append(section_name)
                        menu_dict['item'].append(menu_item)
                        menu_dict['ingredients'].append(menu_ingredients)
                        menu_dict['price'].append(menu_price)
                
            
            else:
            
                menu_=soup.find_all('div', attrs={'class':'arrange'})
                
                # if len(menu_)==0:
                #     continue
                
                for j in range(len(menu_)):
                    
                    menu_item=None
                    menu_a = menu_[j].findChildren('a')
                    for j_a in range(len(menu_a)):
                        
                        
                        if (len(menu_a[j_a])>1) | (str(menu_a[j_a].contents)[0] == '<'):
                            continue
                        else:
                            menu_item =menu_a[j_a].contents[0]
                            break
                    
                    if menu_item == None:
                        menu_item = menu_[j].findChildren('h4')[0].contents[0].splitlines()[1].lstrip()
                    
                        
                    menu_p = menu_[j].findChildren('p')
                    
                    try:
                        menu_ingredients = menu_p[0].contents[0]
                    except:
                        menu_ingredients = None
                        
                    menu_li = menu_[j].findChildren('li')
                    
                    try:
                        menu_price = re.search("\\d.*\\d", menu_li[0].contents[0])[0]
                    except:
                        menu_price = None
                        
                    menu_dict['categories'].append(categories)
                    menu_dict['section'].append(None)
                    menu_dict['item'].append(menu_item)
                    menu_dict['ingredients'].append(menu_ingredients)
                    menu_dict['price'].append(menu_price)
        
            if len(menu_dict['categories'])==0:
                for key in menu_dict.keys():
                    
                    menu_dict[key]=None
                menu_dict['categories'] = categories
            menu_df = pd.DataFrame(menu_dict)
            
            
            menu_df.drop_duplicates(inplace=True, ignore_index=True)
            menu_df.to_csv(self.machine + self.menu_subfolder + short_url + '.csv')
            
        
    def cleanData(self, date):
        
        # in the first sample, I start with 26023 rows
        new_data = pd.read_csv(self.machine + self.output_folder + date + '/yelp_raw_data_with_dates_' + date + '.csv', index_col = [0])
        # set the index equal to the Yelp ID
        new_data.set_index('id', inplace=True)
        
        ### ADJUST LOCATION AND COORDINATES
        # in the original df 'location' (and other columns) are read as strings of dictionaries
        new_data['location'] = new_data['location'].apply(ast.literal_eval)
        # concatenate df with expanded dictionary of locations
        new_data = pd.concat((new_data, pd.DataFrame.from_records(new_data['location'], index=new_data.index)), axis=1)
        # a few points to not belong to the US, so I discard them (6 rows)
        new_data = new_data[new_data['country']=='US']
        
        new_data.drop('location', axis=1, inplace=True)
        # same for coordinates
        new_data['coordinates'] = new_data['coordinates'].apply(ast.literal_eval)
        new_data = pd.concat((new_data, pd.DataFrame.from_records(new_data['coordinates'], index=new_data.index)), axis=1)
        # exclude rows (4) with no coordinates
        new_data = new_data[(new_data['longitude'].apply(np.isnan)==False) | (new_data['latitude'].apply(np.isnan)==False)]
        new_data.drop('coordinates', axis=1, inplace=True)
        
        ### ADJUST CATEGORIES

        # write categories in list of strings format
        new_data['categories'] = new_data['categories'].str.replace("'", '"')
        new_data['category_titles'] = new_data['categories'].apply(lambda s: re.findall('(?<=title": ").*?(?=")', s))
        new_data['categories'] = new_data['categories'].apply(lambda s: re.findall('(?<=alias": ").*?(?=")', s))
        # exclude rows (1 in the first sample) with no category
        new_data = new_data.loc[new_data.categories.apply(len)!=0]
        
        # see if at least one label or all labels belonss to CATEGORIES defined at the top of the code
        any_rest = []
        all_rest = []
        for idx in new_data.index:
            # idx of restaurant appearing in at least one restaurant category
            if any(item in self.CATEGORIES for item in new_data.loc[idx,'categories']) == True:
                any_rest.append(idx)
            # idx of restaurant appearing in all restaurant categories
            if all(item in self.CATEGORIES for item in new_data.loc[idx,'categories']) == True:
                all_rest.append(idx)
                
        # later I filter so that only restaurants with at least one 'restaurant' category are left
        new_data['isOnlyRestaurant'] = False
        new_data['isAtLeastRestaurant'] = False
        new_data.loc[all_rest, 'isOnlyRestaurant'] = True
        new_data.loc[any_rest, 'isAtLeastRestaurant'] = True
        
        self.yelp_data = new_data.copy()
        
        idx_all = self.yelp_data.isOnlyRestaurant==True
        idx_any = self.yelp_data.isAtLeastRestaurant==True
        
        self.yelp_data = self.yelp_data.sort_index(axis=0)
        self.yelp_data.categories = self.yelp_data.categories.apply(lambda x: sorted(x))

        self.yelp_data['BoroCT2020'] = self.yelp_data['BoroCT2020'].astype(int)
        self.yelp_data['concatenated_categories'] = self.yelp_data.categories.apply("+".join)
        self.yelp_data = self.yelp_data.join(pd.get_dummies(self.yelp_data.transactions.apply(ast.literal_eval).explode()).groupby(level=0).sum())
        
        # save matrices: '_original' and '_all' probably won't be used again
        self.data_original = self.yelp_data.copy()
        self.data_original.to_csv(self.machine + self.output_folder + 'yelp_data_original' +  '_' + date + '.csv')
        
        self.data_all = self.yelp_data[idx_all]
        self.data_all.to_csv(self.machine + self.output_folder + 'yelp_data_all' +  '_' + date + '.csv')
        
        # this is the main dataframe I will be working on
        self.yelp_data = self.yelp_data[idx_any]
        self.yelp_data.to_csv(self.machine + self.output_folder + 'yelp_data' +  '_' + date + '.csv')
     
    def matchCensusTract(self):
       
        data = self.yelp_data.copy()
        id_column = 'id'
        
        self.yelp_data = Base.findCensusTracts(self, data, id_column, self.output_folder)
        # CHECK THIS!
        self.yelp_data = self.yelp_data.reset_index().merge(self.census_gpd[['BoroCT2020', 'BoroCode', 'BoroName', 'NTAName', 'NTA2020']], on = 'BoroCT2020', how = 'left')
        self.yelp_data.set_index('id', inplace=True)
        
        self.yelp_data = self.yelp_data.sort_index(axis=0)
        self.yelp_data.categories = self.yelp_data.categories.apply(lambda x: sorted(x))
        self.yelp_data.to_csv(self.machine + self.output_folder + 'yelp_data' +  '_' + self.date + '.csv')
        
    def retrieveData(self):
        
        self.yelp_data = pd.read_csv(self.machine + self.output_folder + 'yelp_data' +  '_' + self.date + '.csv', index_col=[0])
        
    def writeSummaryData(self):
        
        varss=['review_count',
        'rating',
        'price',
        'BoroName',
        'delivery',
        'pickup',
        'restaurant_reservation']
        
        short_df = self.yelp_data[varss]
        
        short_df = short_df.join(pd.get_dummies(short_df.BoroName).groupby(level=0).sum())
        short_df = short_df.join(pd.get_dummies(short_df.price, dummy_na=True).groupby(level=0).sum())
        short_df['N.A.'] = short_df[np.nan]
        short_df = short_df.rename({'delivery': 'Delivery',
                                    'pickup': 'Pickup',
                                    'restaurant_reservation': 'Restaurant reservation'},
                                    axis=1)
        summary = {}
        
        for boro in self.boroughs:
            summary[('Borough', boro)] = [sum(short_df[boro]), sum(short_df[boro])/len(short_df)*100]
        for price in ['$', '$$', '$$$', '$$$$', '$$$$', 'N.A.']:
            summary[('Price', price)] = [sum(short_df[price]), sum(short_df[price])/len(short_df)*100]

        summary[('Rating', 'Rating less or equal than 2')] = [sum(short_df.rating<=2), sum(short_df.rating<=2)/len(short_df)*100]
        summary[('Rating', 'Rating between 2 and 3')] = [sum((short_df.rating>2) & (short_df.rating<=3)), sum((short_df.rating>2) & (short_df.rating<=3))/len(short_df)*100]
        summary[('Rating', 'Rating between 3 and 4')] = [sum((short_df.rating>3) & (short_df.rating<=4)), sum((short_df.rating>3) & (short_df.rating<=4))/len(short_df)*100]
        summary[('Rating', 'Rating between 4 and 5')] = [sum((short_df.rating>4) & (short_df.rating<=5)), sum((short_df.rating>4) & (short_df.rating<=5))/len(short_df)*100]
        
        for transaction in ['Delivery', 'Pickup', 'Restaurant reservation']:
            summary[('Offers', transaction)] = [sum(short_df[transaction]), sum(short_df[transaction])/len(short_df)*100]

        summary = pd.DataFrame(summary).T
        summary = summary.rename({0:'Count',1:'%'}, axis=1)
        summary['Count'] = summary['Count'].map('{:.0f}'.format)
        summary['%'] = summary['%'].map('{:,.2f}'.format)
        
        file_name = 'tables/summary_yelp.tex'
        tex_file = open(file_name, 'w')
        tex_file.write(summary.to_latex(sparsify = True,
                                        caption = '',
                                        label = 'tab:summary_yelp'))
        tex_file.close()
        
       
    
    def findDistancesRestaurants(self):
        
        geo_rests = pd.concat((self.yelp_data.latitude, self.yelp_data.longitude), axis=1)
        
        geo_rests['geometry'] = list(map(Point,list(zip(geo_rests.longitude,geo_rests.latitude))))
        
        CRS_LATLON = 'GCS_WGS_1984'
        CRS_M = 'EPSG:32118'
        
        
        geo_rests = gpd.GeoDataFrame(geo_rests)
        geo_rests.geometry = geo_rests.geometry.set_crs(CRS_LATLON).to_crs(CRS_M)
        
        dist_rests = pd.DataFrame(index=geo_rests.index, columns=geo_rests.index)
        rest_idx=0
        for rest_idx in range(len(geo_rests.index)):
            if rest_idx%100==0:
                print(rest_idx)
            dists = geo_rests.geometry.iloc[rest_idx:].apply(geo_rests.iloc[rest_idx]['geometry'].distance)
            #dists = geo_rests.geometry.apply(geo_rests.loc[rest, 'geometry'].distance)
            dists[dists>5000] = np.nan
            dists[~np.isnan(dists)] = dists[~np.isnan(dists)].astype(int)
            dist_rests.iloc[rest_idx,rest_idx:] = dists#.astype(int)
            rest_idx += 1
       
        dist_rests.to_csv(self.machine + 'yelp_data/restaurant_distances.csv')
        
        
    def saveUndirectedGeographicNetworkRestaurants(self):
        
        R = np.array([100, 200, 300, 400, 500, 750, 1000, 1250, 1500, 1750, 2000, 2500, 3000, 4000, 5000])

        machine = '/Users/michelev/spatial-competition-food/'
        # import dataframe of restaurant (physical) distances:
        rest_dists = pd.read_csv(self.machine + 'yelp_data/restaurant_distances.csv', index_col='id', header=0, float_precision='legacy').astype('Int64')
        rest_dists = rest_dists.fillna(0)
        rest_dists_sparse = csr_matrix(rest_dists.astype(int))
        rest_dists_sparse = rest_dists_sparse + rest_dists_sparse.T
        
        for radius in R:
            
            rests_dists_r = rest_dists_sparse.tolil()
            rests_dists_r[rests_dists_r>radius] = 0
            rests_dists_r = rests_dists_r>0
            rests_dists_r = rests_dists_r.tocsr()
            
            scipy.sparse.save_npz(self.machine + 'yelp_data/geographic_network/restaurant_geographic_network_within_' + str(radius) +'.npz', rests_dists_r)
            
    def saveUndirectedCTRestaurants(self):
        
        dummies_CT = pd.get_dummies(self.yelp_data.BoroCT2020.apply(pd.Series).stack()).groupby(level=0).sum()
        dummies_CT_sparse = csr_matrix(dummies_CT)
        
        filtering_matrix_CT_sparse = dummies_CT_sparse.dot(dummies_CT_sparse.T)
        
        scipy.sparse.save_npz(self.machine + 'yelp_data/geographic_network/census_network.npz', filtering_matrix_CT_sparse)

    def saveDirectedDeliveryCTRestaurants(self):
    
            
        dummies_CT = pd.get_dummies(self.yelp_data.BoroCT2020.apply(pd.Series).stack()).groupby(level=0).sum()
        dummies_CT.columns = dummies_CT.columns.astype(int)
        dummies_CT_sparse = csr_matrix(dummies_CT)
        
        dummies_CT_del = pd.read_csv(self.machine + '/grubhub_data/analysis/where_restaurants_deliver_by_Yelp_id.csv', index_col=[0])
        dummies_CT_del.columns = dummies_CT_del.columns.astype(int)
        dummies_CT_del = dummies_CT_del.loc[:,dummies_CT.columns]
        dummies_CT_del = pd.concat((pd.DataFrame(self.yelp_data.index, index=self.yelp_data.index),dummies_CT_del),axis=1)
        dummies_CT_del.pop('id')
        dummies_CT_del = dummies_CT_del.sort_index(axis=1)
        dummies_CT_del = dummies_CT_del.sort_index(axis=0)
        dummies_CT_del = dummies_CT_del.fillna(0)
        dummies_del_sparse = csr_matrix(dummies_CT_del)
        filtering_matrix_delivering =  dummies_CT_sparse.dot(dummies_del_sparse.T)

        scipy.sparse.save_npz(self.machine + 'grubhub_data/delivery_network/filtering_delivering_matrix.npz', filtering_matrix_delivering)

    def saveUndirectedSimilarityCategories(self):
    
        self.yelp_data['concatenated_categories'] = self.yelp_data.categories.apply("+".join)
        df = self.yelp_data[['categories', 'concatenated_categories']].copy()
        #df = df.groupby('concatenated_categories').sum()
        df.set_index('concatenated_categories', inplace=True)
        category_dummies = pd.get_dummies(df.categories.apply(pd.Series).stack()).groupby(level=0).sum()
        category_dummies[category_dummies>0]=1
        category_dummies.loc['laotian','vietnamese']=1
        #pd.get_dummies(df.categories.apply(pd.Series).stack()).groupby(level=0)
        #category_dummies = category_dummies.drop_duplicates()
        #category_dummies = category_dummies.join(df, how = 'left')
        #category_dummies = category_dummies.groupby('concatenated_categories').sum()
        
        similarity_matrix = 1-scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(category_dummies, 'cosine'))
        # = 1-scipy.spatial.distance.squareform(distance_matrix)
        #del distance_matrix
        angs = 1-pd.DataFrame(2*np.arccos(similarity_matrix)/np.pi,index=category_dummies.index,columns=category_dummies.index)
        #distance_angular_matrix = 1-similarity_angular_matrix
            
        angs.to_csv(self.machine + 'yelp_data/similarity_network/category_similarity_angles.csv')
        
        rests_angs = angs.loc[self.yelp_data.concatenated_categories,self.yelp_data.concatenated_categories]
        rests_angs.index = self.yelp_data.index
        rests_angs.columns = self.yelp_data.index
        
        scipy.sparse.save_npz(self.machine + 'yelp_data/similarity_network/restaurant_similarity_angles.npz', csr_matrix(rests_angs.values))

    def findCategoryDistanceMatrix(self):
        
        self.yelp_data['concatenated_categories'] = self.yelp_data.categories.apply("+".join)
        df = self.yelp_data[['categories', 'concatenated_categories']].copy()
        #df = df.groupby('concatenated_categories').sum()
        df.set_index('concatenated_categories', inplace=True)
        category_dummies = pd.get_dummies(df.categories.apply(pd.Series).stack()).groupby(level=0).sum()
        category_dummies[category_dummies>0]=1
        category_dummies.loc['laotian','vietnamese']=1
        #pd.get_dummies(df.categories.apply(pd.Series).stack()).groupby(level=0)
        #category_dummies = category_dummies.drop_duplicates()
        #category_dummies = category_dummies.join(df, how = 'left')
        #category_dummies = category_dummies.groupby('concatenated_categories').sum()
        
        similarity_matrix = 1-scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(category_dummies, 'cosine'))
        # = 1-scipy.spatial.distance.squareform(distance_matrix)
        #del distance_matrix
        angs = pd.DataFrame(2*np.arccos(similarity_matrix)/np.pi,index=category_dummies.index,columns=category_dummies.index)
        #distance_angular_matrix = 1-similarity_angular_matrix

        similarity_matrix = 1-scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(category_dummies, 'cosine'))
        # = 1-scipy.spatial.distance.squareform(distance_matrix)
        #del distance_matrix
        distance_angular_matrix = 2*np.arccos(similarity_matrix)/np.pi
        #distance_angular_matrix = 1-similarity_angular_matrix
        
        distance_angular_matrix[distance_angular_matrix==1] = np.nan
        np.fill_diagonal(distance_angular_matrix, np.nan)
        
        edge_df =pd.DataFrame(distance_angular_matrix).stack().reset_index()
        edge_df.columns = ['source', 'target', 'weight']
        
        G = nx.from_pandas_edgelist(edge_df, source='source', target='target', edge_attr='weight')
        
        bwc  = nx.betweenness_centrality(G,weight = 'weight')
        clust = nx.clustering(G, weight='weight')
        shortest_paths = networkx.floyd_warshall_numpy(G, weight='weight')
        
        shortest_paths_df=pd.DataFrame(shortest_paths, index = category_dummies.index, columns = category_dummies.index)
        shortest_paths_df.to_csv('shortest_paths.csv')
        #shortest_paths = networkx.all_pairs_bellman_ford_path_length(G, weight = 'weight')
        
        #shortest_paths_df = pd.DataFrame(index = range(len(category_dummies)), columns = range(len(category_dummies)))
        
        # shortest_paths_df.loc[shortest_paths_df.index!=1774,shortest_paths_df.index!=1774]=shortest_paths
        # shortest_paths_df.loc[1774,1774] = 0
        # G1 = networkx.from_numpy_array(shortest_paths_df.to_numpy(dtype = float))
        # shortest_paths_df.loc[shortest_paths_df.index==1774,:] =
        # shortest_paths_df.loc[:,shortest_paths_df.index==1774] =
        
        # similarity_binary = similarity_matrix.copy()
        # similarity_binary[similarity_binary>0]=1
        # similarity_binary = similarity_binary.astype(bool)
        # G1 = nx.from_numpy_array(similarity_binary)
        # bwc = nx.betweenness_centrality(G1)
        
        #freqs = -np.log(category_dummies.sum(0)/category_dummies.shape[0])
        freqs = category_dummies.sum(0)/category_dummies.shape[0]
        category_dummies_weighted = category_dummies*(-np.log(freqs))
        similarity_matrix_weighted = 1-scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(category_dummies_weighted, 'cosine'))
        # = 1-scipy.spatial.distance.squareform(distance_matrix)
        #del distance_matrix
        distance_angular_matrix_weighted = 2*np.arccos(similarity_matrix_weighted)/np.pi
        #distance_angular_matrix = 1-similarity_angular_matrix
        
        distance_angular_matrix_weighted[distance_angular_matrix_weighted==1] = np.nan
        np.fill_diagonal(distance_angular_matrix_weighted, np.nan)
        
        edge_df =pd.DataFrame(distance_angular_matrix_weighted).stack().reset_index()
        edge_df.columns = ['source', 'target', 'weight']
        
        G_weighted = nx.from_pandas_edgelist(edge_df, source='source', target='target', edge_attr='weight')
        
        bwc  = nx.betweenness_centrality(G,weight = 'weight')
        clust = nx.clustering(G, weight='weight')
        shortest_paths = networkx.floyd_warshall_numpy(G_weighted, weight='weight')
        
        shortest_paths_df=pd.DataFrame(shortest_paths, index = category_dummies.index, columns = category_dummies.index)
        shortest_paths_df.to_csv('shortest_paths_w.csv')
        return G, shortest_paths_df
        
    def fillShortestPaths(shortest_paths_df, G, idx):
        
        print(idx)
        shortest_paths_df.loc[idx,idx:] = networkx.shortest_path_length(G, source=idx, weight='weight')
        shortest_paths_df.loc[idx:,idx] = networkx.shortest_path_length(G, source=idx, weight='weight')
        # if path length is higher than 1e6
        
    with ThreadPoolExecutor(max_workers=10) as executor:
        executor.map(lambda idx: fillShortestPaths(shortest_paths_df, G, idx),    
                      list(range(20)),
                      timeout = 3600)


    def writeDistanceMatrices(self):
               
        print('Write cosine matrix...')
        self.cosine_matrix = Base.computeDistanceMatrix(self.yelp_data, 'cosine', self.machine + self.output_folder, '_arccos', save_npz=False, save_csv=True)
    
        #cosine_matrix, categories_names = Base.computeSimilarityMatrix(self.yelp_data_all, 'cosine', self.machine + self.output_folder, 'all')
        #cosine_matrix, categories_names = Base.computeSimilarityMatrix(self.yelp_data_original, 'cosine', self.machine + self.output_folder, 'original')
        print('Write euclidean matrix...')
        self.euclidean_matrix = Base.computeDistanceMatrix(self.yelp_data, 'euclidean', self.machine + self.output_folder, '', save_npz=False, save_csv=True)
     
    def findDeliveringMatrix(self):
        
        dummies_CT =  pd.get_dummies(self.yelp_data.BoroCT2020.apply(pd.Series).stack()).sum(level=0)
        dummies_CT.columns = dummies_CT.columns.astype(int)
        dummies_CT_sparse = csr_matrix(dummies_CT)
        
        filtering_matrix_CT_sparse = dummies_CT_sparse.dot(dummies_CT_sparse.T)
        
        dummies_CT_del = pd.read_csv(self.machine + '/grubhub_data/analysis/where_restaurants_deliver_by_Yelp_id.csv', index_col=[0])
        dummies_CT_del.columns = dummies_CT_del.columns.astype(int)
        dummies_CT_del = dummies_CT_del.loc[:,dummies_CT.columns]
        dummies_CT_del = pd.concat((pd.DataFrame(self.yelp_data.index, index=self.yelp_data.index),dummies_CT_del),axis=1)
        dummies_CT_del.pop('id')
        dummies_CT_del = dummies_CT_del.sort_index(axis=1)
        dummies_CT_del = dummies_CT_del.sort_index(axis=0)
        dummies_CT_del =dummies_CT_del.fillna(0)
        dummies_del_sparse = csr_matrix(dummies_CT_del)
        filtering_matrix_delivering =  dummies_CT_sparse.dot(dummies_del_sparse.T)
        
    def findCommunities(self):
        
        # To circumvent the fact that the network is too large, I compute similarities among unique lists of categories
        self.yelp_data.categories = self.yelp_data.categories.apply(lambda x: sorted(x))
        self.yelp_data['concatenated_categories'] = self.yelp_data.categories.apply("+".join)
        df = self.yelp_data[['categories', 'concatenated_categories']].copy()
        #df = df.groupby('concatenated_categories').sum()
        df.set_index('concatenated_categories', inplace=True)
        category_dummies = pd.get_dummies(df.categories.apply(pd.Series).stack()).sum(level=0)
        category_dummies[category_dummies>0]=1
        category_dummies.loc['laotian','vietnamese']=1
        #category_dummies = category_dummies.drop_duplicates()
        #category_dummies = category_dummies.join(df, how = 'left')
        #category_dummies = category_dummies.groupby('concatenated_categories').sum()
        
        similarity_matrix = 1-scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(category_dummies, 'cosine'))
        # = 1-scipy.spatial.distance.squareform(distance_matrix)
        #del distance_matrix
        similary_angular_matrix = 1-2*np.arccos(similarity_matrix)/np.pi
        similary_angular_matrix = csr_matrix(similary_angular_matrix)
        # distance_matrix = scipy.spatial.distance.pdist(category_dummies, 'cosine')
        # similarity_matrix = 1-scipy.spatial.distance.squareform(distance_matrix)
        # similarity_matrix = csr_matrix(similarity_matrix)
        G = nx.from_scipy_sparse_array(similary_angular_matrix)
        communities = {}
        
        
        for resolution in [1, 1.5, 2]:
            communities[str(resolution)] = nx.algorithms.community.louvain_communities(G, resolution=resolution, seed=123)
            
            category_dummies['community_' + str(resolution)] = 0
        
            community_index = 1
            for community in communities[str(resolution)]:
                category_dummies.iloc[list(community), -1] = community_index
                community_index +=1
        
        clusters = KMeans(n_clusters=14, random_state=0).fit(category_dummies)
        
        communities_df = category_dummies[['community_1', 'community_1.5', 'community_2']]
        communities_df['clusters_1.5'] = KMeans(n_clusters=14, random_state=0).fit(category_dummies).labels_
        communities_df.to_csv(self.machine + self.output_folder + 'communities.csv')
        
        
        return communities
        
    def findNumberCompetitors(self):
        
        communities = pd.read_csv(self.machine + self.output_folder + 'communities.csv')
        for rel in [1,1.5,2]:
            communities.loc[communities['concatenated_categories']=='vietnames','community_'+str(rel)] = communities.loc[communities['concatenated_categories']=='vietnamese','community_'+str(rel)]
        df = self.yelp_data.reset_index().merge(communities, on ='concatenated_categories')
        df.set_index('id',inplace=True)
        df =df.sort_index(axis=0)
        
       
        dummies_comm = pd.get_dummies(df['community_1.5'].apply(pd.Series).stack()).sum(level=0)
        dummies_comm_sparse = csr_matrix(dummies_comm)
        del dummies_comm
        filtering_same_comm = dummies_comm_sparse.dot(dummies_comm_sparse.T)
        
        merged_dataset = pd.read_csv(self.machine +  'data/' + 'merged_dataset.csv', index_col = [0])
        merged_dataset.set_index('id', inplace=True)
        
        df = df.join(merged_dataset.isGH, how='left', lsuffix='_')        
        del merged_dataset

       
        
        # self.yelp_data.loc[self.yelp_data.isGH!=True,'isGH'] = False
        # dummies_GH = pd.get_dummies(self.yelp_data.isGH.apply(pd.Series).stack()).sum(level=0)
        # dummies_GH_sparse = csr_matrix(dummies_GH)
        # del dummies_GH
        # filtering_GH = dummies_GH_sparse.dot(dummies_GH_sparse.T)
        
        dummies_CT =  pd.get_dummies(self.yelp_data.BoroCT2020.apply(pd.Series).stack()).sum(level=0)
        dummies_CT.columns = dummies_CT.columns.astype(int)
        dummies_CT_sparse = csr_matrix(dummies_CT)
        
        filtering_matrix_CT_sparse = dummies_CT_sparse.dot(dummies_CT_sparse.T)
        
        dummies_CT_del = pd.read_csv(self.machine + '/grubhub_data/analysis/where_restaurants_deliver_by_Yelp_id.csv', index_col=[0])
        dummies_CT_del.columns = dummies_CT_del.columns.astype(int)
        dummies_CT_del = dummies_CT_del.loc[:,dummies_CT.columns]
        dummies_CT_del = pd.concat((pd.DataFrame(self.yelp_data.index, index=self.yelp_data.index),dummies_CT_del),axis=1)
        dummies_CT_del.pop('id')
        dummies_CT_del = dummies_CT_del.sort_index(axis=1)
        dummies_CT_del = dummies_CT_del.sort_index(axis=0)
        dummies_CT_del =dummies_CT_del.fillna(0)
        dummies_del_sparse = csr_matrix(dummies_CT_del)
        filtering_matrix_delivering =  dummies_CT_sparse.dot(dummies_del_sparse.T)
       
        del dummies_CT_del
        
       
        final_dataset = {}
        # WITHIN DISTANCES
        
        final_dataset['total_CT'] =  np.array(filtering_matrix_CT_sparse.sum(1).reshape(-1))[0]
        final_dataset['community_CT']=  np.array(filtering_matrix_CT_sparse.multiply(filtering_same_comm).sum(1).reshape(-1))[0]
        final_dataset['total_delivery_CT'] =  np.array(filtering_matrix_CT_sparse.multiply(filtering_matrix_delivering).sum(1).reshape(-1))[0]
        final_dataset['community_delivery_CT']=  np.array(filtering_matrix_CT_sparse.multiply(filtering_same_comm).multiply(filtering_matrix_delivering).sum(1).reshape(-1))[0]
        
        # cutoffs = [500,1000,2000,5000]
        # dists_within_dict = {}
        
        # for cut_idx in range(4):
            
        #     cutoff = cutoffs[cut_idx]
        #     dists_within = pd.read_csv(self.distances_folder + 'ct_dummies_linear_distances_' + str(cutoff) + '.csv')
        #     dists_within['within_'+str(cutoff)] = dists_within['within'].apply(ast.literal_eval)
        #     dists_within_dict[cutoff] = dists_within['within_'+str(cutoff)]
            
        #     if cut_idx==0:
        #         for idx in dists_within.index:
        #             dists_within.at[idx,'within_'+str(cutoff)] =  np.setdiff1d(dists_within_dict[cutoffs[0]][idx], dists_within.loc[idx,'BoroCT2020'])
        #     else:
        #         for idx in dists_within.index:
        #             dists_within.at[idx,'within_'+str(cutoff)] =  np.setdiff1d(dists_within_dict[cutoffs[cut_idx]][idx],dists_within_dict[cutoffs[cut_idx-1]][idx])
            
        #     df = self.yelp_data.reset_index().merge(dists_within, on = 'BoroCT2020', how = 'left')
        #     df.set_index('id', inplace=True)
           
        #     #dummies_dist = pd.get_dummies(df['within_'+str(cutoff)].apply(pd.Series).stack()).sum(level=0)
        #     dummies_dist = df['within_'+str(cutoff)].explode()
        #     dummies_dist= pd.crosstab(dummies_dist.index, dummies_dist)
            
        #     #dummies_dist.columns = dummies_dist.columns.astype(int)
        #     try:
        #         dummies_dist = dummies_dist.loc[:,dummies_CT.columns]
        #     except:
        #         new_cols = np.setdiff1d(dummies_CT.columns,dummies_dist.columns)
        #         for col in new_cols:
        #             dummies_dist[col] = 0
        #         dummies_dist = dummies_dist.loc[:,dummies_CT.columns]
                
        #     new_index = np.setdiff1d(df.index,dummies_dist.index)
            
        
        #     if len(new_index)>0:
        #         dummies_dist = pd.concat((dummies_dist,pd.DataFrame(np.zeros((len(new_index),dummies_dist.shape[1])), index=new_index, columns = dummies_dist.columns)), axis=0)
        #     dummies_dist = dummies_dist.sort_index(axis=0)
        #     dummies_dist_sparse = csr_matrix(dummies_dist)
            
        #     del dummies_dist
            
        #     filtering_matrix_within = dummies_CT_sparse.dot(dummies_dist_sparse.T)
            
        #     del dummies_dist_sparse
        #     # if cut_idx > 0:
        #     #     filtering_matrix_within[cutoff] = filtering_matrix_within[cutoffs[cut_idx]]-filtering_matrix_within[cutoffs[cut_idx-1]]
        #     # if cut_idx >= 1:
        #     #     del filtering_matrix_within[cutoffs[cut_idx-1]]
            
        #     final_dataset['total_' + str(cutoff)] = np.array(filtering_matrix_within.sum(1).reshape(-1))[0]
        #     final_dataset['community_' + str(cutoff)] = np.array(filtering_matrix_within.multiply(filtering_same_comm).sum(1).reshape(-1))[0]
        #     final_dataset['total_delivery_' + str(cutoff)] = np.array(filtering_matrix_within.multiply(filtering_matrix_delivering).sum(1).reshape(-1))[0]
        #     final_dataset['community_delivery_' + str(cutoff)] = np.array(filtering_matrix_within.multiply(filtering_same_comm).multiply(filtering_matrix_delivering).sum(1).reshape(-1))[0]
        
        #     del filtering_matrix_within
        
        # del dummies_CT
        
        final_dataset = pd.DataFrame(final_dataset)
        df['id'] = df.index
        dataset = pd.concat((df.set_index(final_dataset.index),final_dataset), axis=1).set_index('id')
        dataset.BoroCT2020 = dataset.BoroCT2020.astype(int)
        dataset.to_csv('communi.csv')
        
    def findDiversityBalance(self):
        
        self.yelp_data = self.yelp_data.sort_index(axis=0)
        self.yelp_data.categories = self.yelp_data.categories.apply(lambda x: sorted(x))
        self.yelp_data['concatenated_categories'] = self.yelp_data.categories.apply("+".join)
        self.yelp_data.BoroCT2020 = self.yelp_data.BoroCT2020.astype(int)
        dummies = pd.get_dummies(self.yelp_data.concatenated_categories, sparse=True)
        self.cosine_matrix = Base.computeDistanceMatrix(self.yelp_data, 'cosine', self.machine + self.output_folder, '', save_npz=False, save_csv=False)

        #dummies = dummies.join(self.yelp_data[['BoroCT2020', 'NTA2020']])

        #dummies_by_CT = dummies.groupby('BoroCT2020').sum(1)
        
        #dummies_ = dummies_CT.copy()
        dummies_ = neighboring_rests.copy()
        #dummies_ = dummies_CT_del.copy()
        
        label = 'neigh'
        #filtering_matrix = filtering_matrix_CT
        filtering_matrix = filtering_matrix_neighbors
        #filtering_matrix = filtering_matrix_delivering_CT
        # cats_by_CT = dummies_CT.T.dot(dummies)
        cats_neighbors =  neighboring_rests.T.dot(dummies)
        # cats_by_del_CT =  dummies_CT_del.T.dot(dummies)
        # #cats_del = delivering.T.dot(dummies)
       
        
        cats = dummies_.T.dot(dummies)
        
        species = (cats>0).sum(1)
        pi = cats/cats.sum(1).reshape(-1,1)
        pi[np.isnan(pi)]=0
        shannon = np.multiply(pi, np.log(pi))
        #shannon = -shannon.fillna(0).sum(1)
        shannon[np.isnan(shannon)] = 0
        shannon = -shannon.sum(1)
        simpson = 1-np.square(pi).sum(1)
        simpson[np.isnan(simpson)] = 0
        
       
        # dummies_CT =  pd.get_dummies(self.yelp_data.BoroCT2020.apply(pd.Series).stack()).sum(level=0)
        # dummies_CT_sparse = csr_matrix(dummies_CT)
        # filtering_matrix_CT = dummies_CT_sparse.dot(dummies_CT_sparse.T)

        self.yelp_data['Disparity_' + label] = Base.computeSimilarityArray(self.cosine_matrix, 'cosine', filtering_matrix)

        #dummies = pd.get_dummies(self.yelp_data.categories.apply(pd.Series).stack()).sum(level=0)
        #dummies = dummies.join(self.yelp_data[['BoroCT2020', 'NTA2020']])

       
        
        self.yelp_data.drop('pi_'+label, axis=1, inplace=True)
        self.yelp_data['pi_'+label]=0
        for idx in range(len(self.yelp_data)):
            boro = np.where(boros==self.yelp_data.iloc[idx]['BoroCT2020'])[0][0]
            cat = np.where(dummies.columns==self.yelp_data.iloc[idx]['concatenated_categories'])[0][0]
            #print(cat)
            #self.yelp_data.loc[idx, 'pi_'+'CT'] = pi[self.yelp_data.loc[idx,'BoroCT2020'], self.yelp_data.iloc[idx,'concatenated_categories']]
            self.yelp_data.iloc[idx,-1] = pi[boro,cat]
        
        pipj_matrix = self.yelp_data['pi_'+label].to_frame().dot(self.yelp_data['pi_'+label].to_frame().T)
        
        self.yelp_data['Multi_' + label] = Base.computeSimilarityArray((pipj_matrix.multiply(self.cosine_matrix)).to_numpy(), 'cosine', filtering_matrix)
        #self.yelp_data['Multi_neighbors'] = Base.computeSimilarityArray((pipj_matrix.multiply(self.cosine_matrix)).to_numpy(), 'cosine', filtering_matrix_neighbors)
        #self.yelp_data['Multi_delivering'] = Base.computeSimilarityArray((pipj_matrix.multiply(self.cosine_matrix)).to_numpy(), 'cosine', filtering_matrix_delivering)
        #self.yelp_data['Multi_delivering_CT'] = Base.computeSimilarityArray((pipj_matrix.multiply(self.cosine_matrix)).to_numpy(), 'cosine', filtering_matrix_delivering_CT)
        
       
        sum1 = dummies_.T.dot(self.yelp_data['Disparity_' + label])
        sum2 = dummies_.T.dot(self.yelp_data['Multi_' + label])
        
        diversity_indices=pd.DataFrame({'Species_' + label: species, 'Shannon_' + label: shannon, 'Simpson_' + label: simpson, 'Disparity_' + label: sum1, 'Multi_' + label: sum2})
        diversity_indices.index = boros
      
        diversity_indices.to_csv('data/diversity_indices_' + label +'.csv')
       
        #--

        dummies = pd.get_dummies(self.yelp_data.categories.apply(pd.Series).stack()).sum(level=0)
        dummies['length'] = dummies.sum(1)
        dummies = dummies.iloc[:,:-1].div(dummies.length,axis=0)
        dummies = dummies.join(self.yelp_data[['BoroCT2020', 'NTA2020']])
        
        
        dummies_by_CT = dummies.loc[:,dummies.columns!='length'].groupby('BoroCT2020').sum(1)
        species = (dummies_by_CT>0).sum(1)
        #species = dummies_by_CT.sum(1)
        #species.fillna(0, inplace=True)
        #dummies = pd.get_dummies(self.yelp_data.categories.apply(pd.Series).stack()).sum(level=1)
        pi = dummies_by_CT.div(dummies['BoroCT2020'].value_counts(), axis=0)
        shannon = np.multiply(pi, np.log(pi))
        shannon = -shannon.fillna(0).sum(1)
        simpson = 1-np.square(pi).sum(1)
        
        self.cosine_matrix = Base.computeDistanceMatrix(self.yelp_data, 'cosine', self.machine + self.output_folder, '', save_npz=False, save_csv=False)
        
        dummies_CT =  pd.get_dummies(self.yelp_data.BoroCT2020.apply(pd.Series).stack()).sum(level=0)
        dummies_CT_sparse = csr_matrix(dummies_CT)
        filtering_matrix_CT = dummies_CT_sparse.dot(dummies_CT_sparse.T)

        self.yelp_data['Disparity_CT'] = Base.computeSimilarityArray(self.cosine_matrix, 'cosine',filtering_matrix_CT)
        
        #species_ = pd.DataFrame(species, columns=['cuisines']).reset_index()
        
        dummies_pi = dummies.reset_index().merge(species_, on = 'BoroCT2020', how='left')
        dummies_pi.set_index('id', inplace=True)
        dummies_pi = dummies_pi.iloc[:,:-3].div(dummies_pi.cuisines,axis=0)
        dummies_pi = dummies_pi.dot(dummies_pi.T)
        filtering_matrix_CT_multi = filtering_matrix_CT.multiply(dummies_pi)
        self.yelp_data['Multi_CT'] = Base.computeSimilarityArray(self.cosine_matrix, 'cosine',filtering_matrix_CT_multi)
        
        averages = self.yelp_data[['BoroCT2020', 'Disparity_CT', 'Multi_CT']].groupby('BoroCT2020').sum()
        
        diversity_indices=pd.DataFrame({#'Species': species, 'Shannon': shannon, 'Simpson': simpson, 'Disparity': averages['Disparity_CT']
                                        'Multicriteria': averages['Multi_CT']})
        diversity_indices.index = diversity_indices.index.astype(int)
      
        diversity_indices.to_csv('data/diversity_indices2.csv')
        
    def findDifferentiationScore(self):

        self.cosine_matrix = Base.computeDistanceMatrix(self.yelp_data, 'cosine', self.machine + self.output_folder, '', save_npz=False, save_csv=False)
        self.euclidean_matrix = Base.computeDistanceMatrix(self.yelp_data, 'euclidean', self.machine + self.output_folder, '', save_npz=False, save_csv=False)
        
        self.yelp_data.reset_index(inplace=True)
        # WITHIN DISTANCES
        cutoff=500
        dists_within = pd.read_csv(self.distances_folder + 'ct_dummies_linear_distances_' + str(cutoff) + '.csv')
        dists_within['within_'+str(cutoff)] = dists_within['within'].apply(ast.literal_eval)
        df = self.yelp_data.merge(dists_within, on = 'BoroCT2020', how = 'left')
        df.set_index('id', inplace=True)
        dummies = pd.get_dummies(df['within_'+str(cutoff)].apply(pd.Series).stack()).sum(level=0)
        dummies.columns = dummies.columns.astype(int)
        dummies2 =  pd.get_dummies(df.BoroCT2020.apply(pd.Series).stack()).sum(level=0)
        dummies = dummies.loc[:,dummies2.columns]
        dummies_sparse = csr_matrix(dummies)
        dummies2_sparse = csr_matrix(dummies2)
        filtering_matrix_within = dummies2_sparse.dot(dummies_sparse.T)
        
        
        
        ###
        dummies_CT = pd.get_dummies(self.yelp_data.BoroCT2020.astype(int).apply(pd.Series).stack()).sum(level=0)
        boros = list(dummies_CT.columns)
        boros.sort()
        #remaining_cols = np.setdiff1d(boros,dummies_CT.columns) #this is 0
        #for col in remaining_cols:
        #    dummies_CT[col] = 0
        dummies_CT.sort_index(axis=1, inplace=True)
        dummies_CT = csr_matrix(dummies_CT)
        filtering_matrix_CT = dummies_CT.dot(dummies_CT.T)
        
        # NEIGHBORS
        neighbors_1 = pd.read_csv(self.machine  + self.census_folder + 'neighboring_census_tracts_1' + '.csv', index_col = [0])
        neighboring_rests = self.yelp_data[['BoroCT2020']].join(neighbors_1, on = 'BoroCT2020', how = 'left')
        neighboring_rests.drop('BoroCT2020', axis=1,inplace=True)
        neighboring_rests.columns = neighboring_rests.columns.astype(int)
        neighboring_rests = neighboring_rests.loc[:,dummies_CT.columns]
        #remaining_cols = np.setdiff1d(boros,neighboring_rests.columns) #this is 0
        neighboring_rests.sort_index(axis=1, inplace=True)
        neighboring_rests.to_csv(self.machine  + self.census_folder + 'rests_neighboring_census_tracts_1' + '.csv')
        
        neighboring_rests = csr_matrix(neighboring_rests)
        
        filtering_matrix_neighbors = dummies_CT.dot(neighboring_rests.T)
     
        
        #dummies_CT = csr_matrix(dummies_CT)
        # dummies = df.iloc[:,2:]
        # dummies.columns = dummies.columns.astype(int)
        # dummies = dummies.loc[:,dummies2.columns]
        # dummies = csr_matrix(dummies)
        
        
        # NEIGHBORS + DELIVERY
        delivering = pd.read_csv(self.machine + '/grubhub_data/analysis/where_restaurants_deliver_by_Yelp_id.csv', index_col=[0])
        delivering.columns = delivering.columns.astype(int)
        
        #delivering = csr_matrix(delivering)
        
        #filtering_matrix_delivering =  dummies_CT.dot(delivering.T)
        dummies_CT = pd.get_dummies(self.yelp_data.BoroCT2020.astype(int).apply(pd.Series).stack()).sum(level=0)
        dummies_CT_del = dummies_CT.add(delivering, fill_value=0)
        dummies_CT_del[dummies_CT_del>1] = 1
        dummies_CT_del[dummies_CT_del!=1] = 0
        dummies_CT_del = dummies_CT_del.loc[:,dummies_CT.columns]
        dummies_CT_del.sort_index(axis=1, inplace=True)
        
        dummies_CT_del = csr_matrix(dummies_CT_del)
        
        dummies_CT.sort_index(axis=1, inplace=True)
        dummies_CT = csr_matrix(dummies_CT)
        
        filtering_matrix_delivering_CT =  dummies_CT.dot(dummies_CT_del.T)
        ###
        
        # self.yelp_data.set_index('id', inplace=True)
        # delivering = delivering.join(self.yelp_data.BoroCT2020, how = 'right')
        # delivering.drop('BoroCT2020', axis=1, inplace=True)
        # #dummies2 = pd.get_dummies(df['BoroCT2020'].apply(pd.Series).stack()).sum(level=0)
        # delivering = delivering.reindex(sorted(delivering.columns), axis=1)
        # delivering.columns = delivering.columns.astype(int)
        # delivering = delivering.loc[:,dummies2.columns.astype(int)]
        # delivering=delivering.fillna(0)
        # delivering = csr_matrix(delivering)
        # filtering_matrix_delivering =  dummies2_sparse.dot(delivering.T)
        
        # SAME CENSUS TRACT
        
        filtering_matrix_CT = dummies2_sparse.dot(dummies2_sparse.T)
        
        filtering_matrix_OP_CT = filtering_matrix_delivering + filtering_matrix_CT
        filtering_matrix_OP_CT[filtering_matrix_OP_CT>1] = 1
       
        filtering_matrix_OP_within = filtering_matrix_delivering + filtering_matrix_within
        filtering_matrix_OP_within[filtering_matrix_OP_within>1] = 1
        
        filtering_matrix_OP_neighbors = filtering_matrix_delivering + filtering_matrix_neighbors
        filtering_matrix_OP_neighbors[filtering_matrix_OP_neighbors>1] = 1
        
        # SAME NTA
        dummiesNTA = csr_matrix(pd.get_dummies(self.yelp_data.NTA2020.apply(pd.Series).stack()).sum(level=0))
        filtering_matrix_NTA = dummiesNTA.dot(dummiesNTA.T)
        
        # cosine_sparse = csr_matrix(self.cosine_matrix)
        # euclidean_sprse = csr_matrix(self.euclidean_matrix)
        
        
        #self.yelp_data['cosine_all'] = Base.computeSimilarityArray(self.cosine_matrix, 'cosine')
        self.yelp_data['euclidean_all'] = Base.computeSimilarityArray(self.euclidean_matrix, 'euclidean')
        
        
        #self.yelp_data['cosine_within500'] = Base.computeSimilarityArray(self.cosine_matrix, 'cosine', filtering_matrix_within)
        self.yelp_data['euclidean_within500'] = Base.computeSimilarityArray(self.euclidean_matrix, 'euclidean',filtering_matrix_within)
    
        #self.yelp_data['cosine_neighbors'] = Base.computeSimilarityArray(self.cosine_matrix, 'cosine', filtering_matrix_neighbors)
        self.yelp_data['euclidean_neighbors'] = Base.computeSimilarityArray(self.euclidean_matrix, 'euclidean', filtering_matrix_neighbors)
        
        #self.yelp_data['cosine_CT'] = Base.computeSimilarityArray(self.cosine_matrix, 'cosine',filtering_matrix_CT)
        self.yelp_data['euclidean_CT'] = Base.computeSimilarityArray(self.euclidean_matrix, 'euclidean', filtering_matrix_CT)
        
        #self.yelp_data['cosine_NTA'] = Base.computeSimilarityArray(self.cosine_matrix, 'cosine', filtering_matrix_NTA)
        self.yelp_data['euclidean_NTA'] = Base.computeSimilarityArray(self.euclidean_matrix, 'euclidean', filtering_matrix_NTA)
    
        #self.yelp_data['cosine_only_delivering'] = Base.computeSimilarityArray(self.cosine_matrix, 'cosine', filtering_matrix_delivering)
        self.yelp_data['euclidean_only_delivering'] = Base.computeSimilarityArray(self.euclidean_matrix, 'euclidean', filtering_matrix_delivering)
        
        #self.yelp_data['cosine_OP_neighbor'] = Base.computeSimilarityArray(self.cosine_matrix, 'cosine', filtering_matrix_OP_CT)
        self.yelp_data['euclidean_OP_neighbor'] = Base.computeSimilarityArray(self.euclidean_matrix, 'euclidean', filtering_matrix_OP_CT)
        
        #self.yelp_data['cosine_OP500'] = Base.computeSimilarityArray(self.cosine_matrix, 'cosine', filtering_matrix_OP_within)
        self.yelp_data['euclidean_OP500'] = Base.computeSimilarityArray(self.euclidean_matrix, 'euclidean', filtering_matrix_OP_within)
        
        #self.yelp_data['cosine_OPCT'] = Base.computeSimilarityArray(self.cosine_matrix, 'cosine', filtering_matrix_OP_neighbors)
        self.yelp_data['euclidean_OPCT'] = Base.computeSimilarityArray(self.euclidean_matrix, 'euclidean', filtering_matrix_OP_neighbors)
        
        
       #  self.yelp_data['median_cosine_all'] = Base.computeMedianDistanceArray(self.cosine_matrix, 'cosine')
       #  self.yelp_data['median_euclidean_all'] = Base.computeMedianDistanceArray(self.euclidean_matrix, 'euclidean')
       
       #  self.yelp_data['median_cosine_within500'] = Base.computeMedianDistanceArray(self.cosine_matrix, 'cosine', filtering_matrix_within)
       #  self.yelp_data['median_euclidean_within500'] = Base.computeMedianDistanceArray(self.euclidean_matrix, 'euclidean',filtering_matrix_within)
    
       #  self.yelp_data['median_cosine_neighbors'] = Base.computeMedianDistanceArray(self.cosine_matrix, 'cosine', filtering_matrix_neighbors)
       #  self.yelp_data['median_euclidean_neighbors'] = Base.computeMedianDistanceArray(self.euclidean_matrix, 'euclidean', filtering_matrix_neighbors)
        
       #  self.yelp_data['median_cosine_CT'] = Base.computeMedianDistanceArray(self.cosine_matrix, 'cosine',filtering_matrix_CT)
       #  self.yelp_data['median_euclidean_CT'] = Base.computeMedianDistanceArray(self.euclidean_matrix, 'euclidean', filtering_matrix_CT)
        
       #  self.yelp_data['median_cosine_NTA'] = Base.computeMedianDistanceArray(self.cosine_matrix, 'cosine', filtering_matrix_NTA)
       #  self.yelp_data['median_euclidean_NTA'] = Base.computeMedianDistanceArray(self.euclidean_matrix, 'euclidean', filtering_matrix_NTA)
    
       #  self.yelp_data['median_cosine_only_delivering'] = Base.computeMedianDistanceArray(self.cosine_matrix, 'cosine', filtering_matrix_delivering)
       #  self.yelp_data['median_euclidean_only_delivering'] = Base.computeMedianDistanceArray(self.euclidean_matrix, 'euclidean', filtering_matrix_delivering)
        
       #  self.yelp_data['median_cosine_OP_neighbor'] = Base.computeMedianDistanceArray(self.cosine_matrix, 'cosine', filtering_matrix_OP_CT)
       #  self.yelp_data['median_euclidean_OP_neighbor'] = Base.computeMedianDistanceArray(self.euclidean_matrix, 'euclidean', filtering_matrix_OP_CT)
        
       #  self.yelp_data['median_cosine_OP500'] = Base.computeMedianDistanceArray(self.cosine_matrix, 'cosine', filtering_matrix_OP_within)
       #  self.yelp_data['median_euclidean_OP500'] = Base.computeMedianDistanceArray(self.euclidean_matrix, 'euclidean', filtering_matrix_OP_within)
        
       #  self.yelp_data['median_cosine_OPCT'] = Base.computeMedianDistanceArray(self.cosine_matrix, 'cosine', filtering_matrix_OP_neighbors)
       #  self.yelp_data['median_euclidean_OPCT'] = Base.computeMedianDistanceArray(self.euclidean_matrix, 'euclidean', filtering_matrix_OP_neighbors)
       # # self.yelp_data['log_cosine_within'] = -np.log(self.yelp_data['cosine_within'])
        # self.yelp_data['log_euclidean_within'] = -np.log(self.yelp_data['euclidean_within'])
        
        # self.yelp_data['similarity_diff'] = self.yelp_data['cosine_within']-self.yelp_data['similarity']
        # self.yelp_data['distance_diff'] = self.yelp_data['euclidean_within']-self.yelp_data['distance']
        
        self.yelp_data.set_index('id', inplace=True)
        self.yelp_data.to_csv(self.machine + self.output_folder + 'yelp_data' +  '_' + self.date + '_temp6.csv')

    def addDiskCoordinates(self):
        
        self.coords['r'] = self.coords['r']/np.max(self.coords['r'])
        self.yelp_data['r'] = self.coords['r']
        self.yelp_data['theta'] = self.coords['theta']
        
        self.yelp_data['x'] = self.coords['r']*self.coords['theta'].apply(np.cos)
        self.yelp_data['y'] = self.coords['r']*self.coords['theta'].apply(np.sin)
        
        self.yelp_data.to_csv(self.machine + self.output_folder + 'yelp_data' +  '_' + self.date + '.csv', index=False)
        
#%%  

# YP = YelpClient()
# YP.findDistancesRestaurants()

    
def main():
    
    date = '01-31-2023'
    
    YP = YelpClient()
    YP.updateDate(date)
    #YP.findEfficientCategories()
    #YP.findNeighborhoods()
    
    
    #YP.aggregateData()
    
    # with ThreadPoolExecutor(max_workers=5) as executor:
    #     executor.map(lambda start: YP.addEstablishmentYears(start),
    #                   list(np.arange(30000, 40000, 100)),
    #                   timeout = 3600)
        
    
    with ThreadPoolExecutor(max_workers=5) as executor:
        executor.map(YP.findMenu,
                      list(np.arange(0, 20000, 100)),
                      timeout = 3600)

          
    #YP.aggregateData(False)
    
    #YP.yelp_data = YP.yelp_data.sort_index(axis=0)
    #YP.yelp_data.categories = YP.yelp_data.categories.apply(lambda x: sorted(x))
    #YP.yelp_data['concatenated_categories'] = YP.yelp_data.categories.apply("+".join)
  
    #YP.raw_data = pd.read_csv(YP.machine + 'yelp_data.csv', index_col=[0])
    #YP.cleanData()
    #YP.matchCensusTract()
    #YP.writeDistanceMatrices()
    # for start in list(np.arange(0, 5000, 100)):
    #     try:
    #         allRestaurants = pd.read_csv(self.machine + 'yelp_data/' + date + '/raw_data_with_dates/yelp_raw_data_' + date + '_' + str(int(start/100)) + '.csv', index_col=[0])
    #     except:
    #         continue
    #     allRestaurants.loc[allRestaurants.lastEstimatedEstablishment==1900,'lastEstimatedEstablishment']=None
    #     allRestaurants.to_csv(self.machine + 'yelp_data/' + date + '/raw_data_with_dates/yelp_raw_data_' + date + '_' + str(int(start/100)) + '.csv')

    
    #YP.findDifferentiationScore()
    #YP.addDiskCoordinates()
    
    #filedate = '2022-10-14'
    #YP.yelp_data = pd.to_csv(YP.machine + YP.output_folder + 'yelp_data' +  '_' + filedate + '.csv')
    #return YP
#main()