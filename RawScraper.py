#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 11:28:48 2023

@author: michelev
"""
import os

from bs4 import BeautifulSoup
import requests
from lxml import html  
import csv
import re
import argparse
import sys
import pandas as pd

import sys
import numpy as np

import json

from Scraper import getResponseProxies
from concurrent.futures import ThreadPoolExecutor

from datetime import datetime
#%%

#?# copy proxies from text file (should not be public)
proxies = import_proxies()

headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/103.0.5060.114 Safari/537.36'}


machine = '/Users/michelev/spatial-competition-food/'

links = pd.read_csv(machine + 'yelp_data/panel_data/' + 'raw_links.csv', index_col=[0])
links = links['0'].values

#%%

def createSubDataFrame(start, links):
    
    if not os.path.exists(machine + 'yelp_data/panel_data/scraped_old_firms_ ' + str(int(start/100)) + '.csv'):
       
    
        
        print('Building dataframe from row ' + str(start) + '...')
        df_ = {'link': [],
               'id': [],
               'name': [],
               'closed': [],
               'categoryType': [],
               'cuisine': [],
               'streetAddress': [],
               'addressLocality': [],
               'addressCountry': [],
               'addressRegion': [],
               'postalCode': [],
               'priceRange': [],
               'telephone': [],
               'ratingValue': [],
               'reviewCount': [],
               'firstEstablishment': [],
               'lastReview': [],
               'latitude': [],
               'longitude': []
               }
        
        extreme = min(start+100, len(links))
        
        for url_name in links[start:extreme]:
            
            
            
            
            url = 'https://www.yelp.com/biz/' + url_name + '?sort_by=date_desc'
            
            # get response
            response = getResponseProxies(url, headers, proxies)
            
            # retrieve from script id and close status
            soup = BeautifulSoup(response.content, 'lxml')
            
            links_ = soup.find_all('script', type='application/json')
            if len(links_)>0:
                dict_main_ = json.loads(re.search('\{.+\}', str(links_[0])).group())
            else:
                for key in df_.keys():
                    df_[key].append(None)
            
            if len(links_)==0:
                continue
            
            df_['link'].append(url_name)
            
            first_est = re.search("(?<=on yelp since ).*?(?=&)", str(soup.contents))
            if first_est == None:
                first_est = re.search("(?<=stablished in ).*?(?=\.)", str(soup.contents))
            if first_est != None:
                first_est = first_est[0]
                
            try:
                first_est = int(first_est)
            except:
                first_est = None
            
            df_['firstEstablishment'].append(first_est)
                
            try:
                df_['id'].append(dict_main_['legacyProps']['bizDetailsProps']['bizDetailsMetaProps']['businessId'])
            except:
                df_['id'].append(None)
                
            try:
                df_['closed'].append(dict_main_['legacyProps']['bizDetailsProps']['gaDimensions']['global']['biz_closed'][1])
            except:
                df_['closed'].append(None)
            
            try:
                df_['cuisine'].append(dict_main_['legacyProps']['bizDetailsProps']['gaDimensions']['global']['category_paths_to_root'][1])
            except:
                df_['cuisine'].append(None)
                
            links = soup.find_all('script', type='application/ld+json')
            for j in range(len(links)):
                dict_main = json.loads(re.search('\{.+\}', str(links[j])).group())
                if 'telephone' in dict_main.keys():
                    break
            
            try:
                df_['categoryType'].append(dict_main['@type'])
            except:
                df_['categoryType'].append( None)
                
            try:
                df_['name'].append(dict_main['name'])
            except:
                df_['name'].append(None)
                
            try:
                df_['streetAddress'].append(dict_main['address']['streetAddress'])
            except:
                df_['streetAddress'].append(None)
                
            try:
                df_['addressLocality'].append(dict_main['address']['addressLocality'])
            except:
                df_['addressLocality'].append(None)
            
            try:
                df_['addressCountry'].append(dict_main['address']['addressCountry'])
            except:
                df_['addressCountry'].append(None)
            
            try:
                df_['addressRegion'].append(dict_main['address']['addressRegion'])
            except:
                df_['addressRegion'].append(None)
            
            try:
                df_['postalCode'].append(dict_main['address']['postalCode'])
            except:
                df_['postalCode'].append(None)
                
            try:
                df_['telephone'].append(dict_main['telephone'])
            except:
                df_['telephone'].append(None)
            
            try:
                df_['priceRange'].append(dict_main['priceRange'])
            except:
                df_['priceRange'].append(None)
                
            try:
                df_['ratingValue'].append(dict_main['aggregateRating']['ratingValue'])
            except:
                df_['ratingValue'].append(None)
                
            try:
                df_['reviewCount'].append(dict_main['aggregateRating']['reviewCount'])
            except:
                df_['reviewCount'].append(None)
                
            
            if 'review' in dict_main.keys():
                lastReview=1900
                for j in range(len(dict_main['review'])):
                    
                        tmp = datetime.strptime(dict_main['review'][j]['datePublished'], '%Y-%m-%d').year
                        if tmp>lastReview:
                            lastReview=tmp
                df_['lastReview'].append(lastReview)
                if lastReview == 1900:
                    lastReview = None
            else:
                df_['lastReview'].append(None)
                
            try:    
                latlon = re.search("(?<=center=).*?(?=&amp)", str(soup.contents))[0]
                    
                lat = re.search(".*(?=%2C)", latlon)[0]
                long = re.search("(?<=%2C).*", latlon)[0]
            except:
                lat=None
                long=None
                
            
            df_['latitude'].append(lat)
            df_['longitude'].append(long)
        
        pd.DataFrame(df_).to_csv(machine + 'yelp_data/panel_data/scraped_old_firms_ ' + str(int(start/100)) + '.csv')
        

def createDataFrame():
    
    path_folder = machine + 'yelp_data/panel_data/'
    
    df = pd.DataFrame()
    for filename in os.listdir(path_folder):
        
        if filename.startswith('scraped_old_firms_'):
            
            df = pd.concat((df, pd.read_csv(path_folder + filename, index_col=[0])), axis=0)
            
    df.drop_duplicates(inplace=True)
    df = df.dropna(subset = 'firstEstablishment')
    df = df.loc[df.addressRegion=='NY']
    
#%%

# for start in list(np.arange(100, 1000, 100)):
#     createSubDataFrame(start, links)
    

with ThreadPoolExecutor(max_workers=5) as executor:
    executor.map(lambda s: createSubDataFrame(s, links),    
                  list(np.arange(0, 12000, 100)),
                  #list(np.arange(0, len(links), 100)),
                  timeout = 3600)
    
#%%

def retrieveMenus(dataframe):
    
    restaurant_ids = dataframe.restaurant_id
    short_urls = dataframe.short_url
        
    for i in range(len(dataframe)):
        
        short_url = short_urls[i]
        url = 'https://www.yelp.com/menu/' + short_url
        
        response = getResponseProxies(url, headers, proxies)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        menu_=soup.find_all('div', attrs={'class':'arrange'})
        
        menu_dict={'item': [], 'ingredients': [], 'price': []}
        
        
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
            menu_dict['item'].append(menu_item)
            menu_dict['ingredients'].append(menu_ingredients)
            menu_dict['price'].append(menu_price)
            
        menu_df = pd.DataFrame(menu_dict)       
        menu_df.drop_duplicates(inplace=True)
    
    
    