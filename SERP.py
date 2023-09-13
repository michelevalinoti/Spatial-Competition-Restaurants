#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 26 17:47:01 2022

@author: michelev
"""

import json
import re

import numpy as np
import pandas as pd

import os

import Scraper
import Base
#%%

proxies = import_proxies()
#%%

machine = '/Users/michelev/spatial-competition-food/'

BOROUGHS =   ['New+York',
              'Manhattan',
              'Brooklyn',
              'Queens',
              'Bronx',
              'Staten+Island']

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

#%%

def saveSinglePageResult(idx, start, num, borough, cuisine):
    
    opener = Scraper.getOpenerSERP(proxies)
    
    url = 'https://www.google.com/search?q=site%3Awww.yelp.com%2Fbiz%2F+%22' + borough + '%2C+NY%22+%22CLOSED%22+' + cuisine + '+Restaurants+OR+%22' + borough + '%2C+NY+%22CLOSED%22+' + cuisine  + '+Food&start=' + str(start) + '&num=' + str(num) 
    
    try:
        res = opener.open(url + '&lum_json=1').read()
    except:
        raw_df = saveSinglePageResult(idx, start, num, borough, cuisine)
        return raw_df
    
    res = json.loads(res)
    
    if ('organic' in res.keys()) == False:
        return 0
    
    items = res['organic']
    list_of_lists = []
    for j in range(len(items)):
        list_of_lists.append([items[j]['title'], items[j]['link'], items[j]['display_link']])
        
    raw_df = pd.DataFrame(list_of_lists, columns=['title', 'link', 'display_link'])

    return raw_df

def saveRawSearches():
    
    for borough in BOROUGHS[1:]:
        
        print('* Borough: ' + borough, flush = True)
        
        for cuisine in CATEGORIES:
            
            total = 1e6
            
            num = 100
            start = 0
            
            idx = 0
            
            print('** Category: ' + cuisine, flush = True)
            
            while start + num < total:
                
                print('*** Page n. ' + str(idx))
                
                raw_df = saveSinglePageResult(idx, start, num, borough, cuisine)
                if type(raw_df) == int:
                    break
                raw_df.to_csv(machine + 'yelp_data/panel_data/raw_data/' + 'raw_table_' + borough + '_' + cuisine + '_' + str(idx) + '.csv')
                
                if len(raw_df)<num:
                    break
                
                idx += 1
                start += num
                
#%%

saveRawSearches()


#%%

path_folder = machine + 'yelp_data/panel_data/raw_data/'

links = pd.Series()
for filename in os.listdir(path_folder):
    
    def returnString(s):
        
        ss = re.search("(?<=biz/).*?(?=\?|$)", s)
        return ss[0]
    
    if Base.getExtension(filename) == 'csv':
       
        df = pd.read_csv(path_folder + filename, index_col=[0])
        links = pd.concat((links, df.link.apply(returnString)))
        
    
links.drop_duplicates(inplace=True)
links.to_csv(machine + 'yelp_data/panel_data/' + 'raw_links.csv')




