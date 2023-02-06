#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  4 13:52:13 2023

@author: michelev
"""
#see this
import pandas as pd
import numpy as np
import os
from collections import Counter, defaultdict
from nltk.util import ngrams
from itertools import chain
from scipy.spatial.distance import pdist, squareform

import Base
#%%

class TextManipulation:
    
    machine = '/Users/michelev/spatial-competition-food/'
    yelp_folder = 'yelp_data/'
    menu_subfolder = 'menus/'
    
    def loadMenus(self):
        
        menus = {}
        
        path_folder = self.machine + self.yelp_folder + self.menu_subfolder
        
        for filename in os.listdir(path_folder):
            
            if Base.getExtension(filename) == 'csv':
                
                menu = pd.read_csv(path_folder + filename, index_col=[0])
                
                menus[filename[:-4]] = menu
                
        self.menus = menus
        
    def createCorpus(self, date, end):
        
        corpusExists = os.path.exists(self.machine + self.yelp_folder + 'categories_' + 'dataframe_' + date + '.csv')
        
        if not corpusExists:
            categories_df = pd.DataFrame()
            sections_df = pd.DataFrame()
            items_df = pd.DataFrame()
            ingredients_df = pd.DataFrame()
        else:
            categories_df = pd.read_csv(self.machine + self.yelp_folder + 'categories_' + 'dataframe_' + date + '.csv', index_col=[0])
            sections_df = pd.read_csv(self.machine + self.yelp_folder + 'sections_' + 'dataframe_' + date + '.csv', index_col=[0])
            items_df = pd.read_csv(self.machine + self.yelp_folder + 'items_' + 'dataframe_' + date + '.csv', index_col=[0])
            ingredients_df = pd.read_csv(self.machine + self.yelp_folder + 'ingredients_' + 'dataframe_' + date + '.csv', index_col=[0])
            
        count=0
        for key in list(self.menus.keys())[:end]:
            
            if count%100==0:
                print('Processing rest. no. ' + str(count))
            
            # CATEGORIES
            if not self.menus[key].categories.isnull().sum()==len(self.menus[key]):
                categories_list = pd.unique(self.menus[key].categories)[0].split()
             
                new_counter = Counter(categories_list)
                
                categories_df =  pd.concat((categories_df, pd.DataFrame(new_counter, index=[key])), axis=0)
            
            else:
                
                categories_df.loc[key,:] = np.nan
            # SECTIONS
            
            if not self.menus[key].section.isnull().sum()==len(self.menus[key]):
            
                section_list = []
                
                self.menus[key].section = self.menus[key].section.apply(Base.standardizeTextColumn)
                for section in self.menus[key].section[pd.isna(self.menus[key].section)==False]:
                    
                    if len(section)==1:
                        section_list.append(section[0])
                        
                    else:
                        
                        for j in range(len(section)-1):
                            section_list.append(section[j] + '_' + section[j+1])
                new_counter = Counter(section_list)
                sections_df =  pd.concat((sections_df, pd.DataFrame(new_counter, index=[key])), axis=0)

            else:
                 
                  sections_df.loc[key,:] = np.nan
                 
            # ITEMS
             
            if not self.menus[key].item.isnull().sum()==len(self.menus[key]):
            
                items_list = []
                
                self.menus[key].item = self.menus[key].item.apply(Base.standardizeTextColumn)
                for item in self.menus[key].item[pd.isna(self.menus[key].item)==False]:
                    
                    if len(item)==1:
                        section_list.append(item[0])
                        
                    else:
                        
                        for j in range(len(item)-1):
                            section_list.append(item[j] + '_' + item[j+1])
                new_counter = Counter(items_list)
                items_df =  pd.concat((items_df, pd.DataFrame(new_counter, index=[key])), axis=0)

            else:
                 
                  items_df.loc[key,:] = np.nan
                 
            # INGREDIENTS
            
            if not self.menus[key].ingredients.isnull().sum()==len(self.menus[key]):
            
                ingredients_list = []
                
                self.menus[key].ingredients = self.menus[key].ingredients.apply(Base.standardizeTextColumn)
                for ingredient in self.menus[key].ingredients[pd.isna(self.menus[key].ingredients)==False]:
                    
                    if len(ingredient)==1:
                        ingredients_list.append(ingredient[0])
                        
                    else:
                        
                        for j in range(len(ingredient)-1):
                            ingredients_list.append(ingredient[j] + '_' + ingredient[j+1])
                            
                new_counter = Counter(ingredients_list)
                ingredients_df =  pd.concat((ingredients_df, pd.DataFrame(new_counter, index=[key])), axis=0)
 
            else:
                 
                 ingredients_df.loc[key,:] = np.nan
           
        
            count +=1
             
        categories_df.to_csv(self.machine + self.yelp_folder + 'categories_' + 'dataframe_' + date + '.csv')
        sections_df.to_csv(self.machine + self.yelp_folder + 'sections_' + 'dataframe_' + date + '.csv')
        items_df.to_csv(self.machine + self.yelp_folder + 'items_' + 'dataframe_' + date + '.csv')
        ingredients_df.to_csv(self.machine + self.yelp_folder + 'ingredients_' + 'dataframe_' + date + '.csv')
                
    def computeSimilarities(self, date, weights):
        
        # define datasets
        categories_df = pd.read_csv(self.machine + self.yelp_folder + 'categories_' + 'dataframe_' + date + '.csv', index_col=[0])
        sections_df = pd.read_csv(self.machine + self.yelp_folder + 'sections_' + 'dataframe_' + date + '.csv', index_col=[0])
        items_df = pd.read_csv(self.machine + self.yelp_folder + 'items_' + 'dataframe_' + date + '.csv', index_col=[0])
        ingredients_df = pd.read_csv(self.machine + self.yelp_folder + 'ingredients_' + 'dataframe_' + date + '.csv', index_col=[0])
        
        text_df = {'categories': categories_df, 'sections': sections_df, 'items': items_df, 'ingredients': ingredients_df}
        similarity_matrices = {}
        
        del categories_df, sections_df, items_df, ingredients_df
        
        for text_key in text_df.keys():
            
            similarity_matrices[text_key] = 1-squareform(pdist(text_df[text_key], 'cosine'))
            
            text_df[text_key].index
#%%
def main():
    
    TM = TextManipulation()
    TM.loadMenus()
    TM.createCorpus('01-31-2023', 1000)
    
main()
            #     section_list = list(chain.from_iterable(self.menus[key].section))
            
            
            
            
            
                
            #     item = item.lower()
            # menu.ingredients = menu.ingredients.replace('.','')
            # menu.ingredients = menu.ingredients.replace(',','')
            # menu.ingredients = menu.ingredients.replace(',','')
            
            
            # [''.join(grams) for grams in ngrams(ingredients, ngram_size) for ingredients in menu.ingredients]
            
            # [list(ngrams(item, ngram_size)) for item in menu.ingredients]