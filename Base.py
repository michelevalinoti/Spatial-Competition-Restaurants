#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  9 12:37:43 2022

@author: michelev
"""

import numpy as np
import pandas as pd
import geopandas as gpd

from scipy.sparse import csr_matrix, save_npz

import ast


import math
import scipy
import re

import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

from shapely.geometry import Point, MultiPoint, MultiPolygon, LineString
from shapely.ops import unary_union, polygonize
import shapely.vectorized
from shapely.validation import make_valid

import networkx as nx

import nltk.corpus
#nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer 
from nltk.tokenize import word_tokenize
#%%

def reshape_(matrix):
    
    return np.array(matrix.reshape(-1))[0]

def standardize(column):
    
    return (column-column.min())/(column.max()-column.min())

def readYelpDataFrame(yelp_df):
    
    yelp_df['category_titles'] = yelp_df['category_titles'].astype(str).apply(ast.literal_eval)
    yelp_df['categories'] = yelp_df['categories'].astype(str).apply(ast.literal_eval)
    
    return yelp_df

def getCTFromFileName(filename):
    
    strings = filename.split('_')
    return int(strings[0])

def getExtension(filename):
    
    return filename.split(".")[-1]
    
#C
# Find census tract of every resturant of the dataframe
def findCensusTracts(self, data, id_column, output_folder):
    
    # create a dataframe of all restaurants and all census tracts
    restaurant_census = pd.DataFrame(index=data.index, columns=self.census_gpd['BoroCT2020'])
    
    #! parallelize this process
    for col in restaurant_census.columns:
        # retrieve polygon corresponding to the considered census tract
        polygon = self.census_gpd.loc[self.census_gpd['BoroCT2020']==col,'geometry'].iloc[0]
        restaurant_census[col] = shapely.vectorized.contains(polygon, data['longitude'], data['latitude'])
        if (np.where(restaurant_census.columns==col)[0][0])%100==0:
            print('Evaluating census tract #: ' +str(np.where(restaurant_census.columns==col)[0][0]))
            
            
    missing_CT = data[restaurant_census.sum(1)==0]
    
    city = MultiPolygon((self.census_gpd.geometry).values)
    city = make_valid(city)
    
    for row in missing_CT.index:
        latlon = Point(missing_CT.loc[row,['longitude','latitude']].values)
        closest_point = shapely.ops.nearest_points(city, latlon)
        if (len(closest_point)>0):
            closest_point=closest_point[0]
        circle = closest_point.buffer(1e-6)
        missing_CT.loc[row,'longitude_'] = closest_point.x
        missing_CT.loc[row,'latitude_'] = closest_point.y
        #missing_CT.loc[row,'circle_'] = circle
        
    restaurant_census_missing = pd.DataFrame(index=missing_CT.index, columns=self.census_gpd['BoroCT2020'])
    
    count = 0
    
    for col in restaurant_census_missing.columns:
        # retrieve polygon corresponding to the considered census tract
        polygon = self.census_gpd.loc[self.census_gpd['BoroCT2020']==col,'geometry'].iloc[0]
        restaurant_census_missing[col] = shapely.vectorized.contains(polygon, missing_CT['longitude_'], missing_CT['latitude_'])
        
        if (np.where(restaurant_census_missing.columns==col)[0][0])%100==0:
            print('Evaluating census tract #: ' +str(np.where(restaurant_census_missing.columns==col)[0][0]))
       
    for col in restaurant_census_missing.columns:
        # retrieve polygon corresponding to the considered census tract
        polygon = self.census_gpd.loc[self.census_gpd['BoroCT2020']==col,'geometry'].iloc[0]
        restaurant_census_missing.loc[restaurant_census_missing.sum(1)==0,col] = shapely.vectorized.touches(polygon, missing_CT.loc[restaurant_census_missing.sum(1)==0,'longitude_'], missing_CT.loc[restaurant_census_missing.sum(1)==0,'latitude_'])
        
        if (np.where(restaurant_census_missing.columns==col)[0][0])%100==0:
            print('Evaluating census tract #: ' +str(np.where(restaurant_census_missing.columns==col)[0][0]))
           
    restaurant_census.loc[restaurant_census_missing.index,:] = restaurant_census_missing
    # check whether all the restaurants have been matched with only one census tract
    data = data[restaurant_census.sum(1)>0] # 
    data = data[restaurant_census.sum(1)<2] #
    
    # create (and save) dataframe of restaurants in each census tract from the initial matrix
    #census_w_restaurants = pd.Series(restaurant_census.sum(0), name=column_name)
    #census_w_restaurants.to_csv(output_folder + 'number_rests_in_tracts.csv')
    
    # create (and save) dataframe of where each restaurant is
    sparse_rc = csr_matrix(restaurant_census)
    census_of_rests_idx = sparse_rc.nonzero()[1]
    corresponding_rests = self.census_gpd['BoroCT2020'].iloc[census_of_rests_idx]
    census_of_rests = pd.DataFrame({'restaurant_id': data.restaurant_id.values, 'BoroCT2020': corresponding_rests.values})
    census_of_rests.to_csv(output_folder + 'census_tract_of_restaurants.csv')
  
    # join the tables above
    data = data.merge(census_of_rests,on='restaurant_id')
    data.set_index(id_column, inplace=True)
    return data

def standardizeTextColumn(item, remove_stopwords = True, stemming = True):
    
    if pd.isna(item):
        return item
    # remove everything except literals
    item = re.sub(r"(@[A-Za-z]+)|([^A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", item)
    item = item.lower()
    item = item.lstrip()
    item = item.rstrip()
    
    if remove_stopwords == True:
        stop = stopwords.words('english')
        item = ' '.join([word for word in item.split() if word not in (stop)])
    
    if stemming == True:
        ps = PorterStemmer()
        item = [ps.stem(word) for word in item.split()]
        
    return item
    
# Define distance matrices and save them, together with the list of categories
def computeDistanceMatrix(df, metric, output_folder, label, save_npz=True, save_csv=False):
            
    # dataframe containing rests over rows and dummies of categories over columns
    dummies = pd.get_dummies(df.categories.apply(pd.Series).stack()).sum(level=0)
    
    if metric=='euclidean':
        dummies = dummies.div(dummies.sum(1),axis=0)
    # retrieve the labels of categories and save them
    categories_names = pd.DataFrame(dummies.columns, columns=['categories_names'])
    categories_names.to_csv('yelp_data/categories_names' + label + '.csv', index=False)

    distance_matrix = scipy.spatial.distance.pdist(dummies, metric)
    distance_matrix = scipy.spatial.distance.squareform(distance_matrix)
    
    # if metric=='cosine':
    #     #distance_matrix = (1-distance_matrix)
    #     distance_matrix = np.arccos(1-distance_matrix)
    #     distance_matrix = 1-2*distance_matrix/np.pi
        
    if save_npz:
        np.savez(output_folder + 'sparse_matrix' + label + '_' + metric  + '.npz', distance_matrix)
    if save_csv:
        np.savetxt(output_folder + 'distance_matrix' + label + '_' + metric + '.csv', distance_matrix)
    
    return distance_matrix

def computeCentrality(distance_matrix, metric, filtering_matrix=None):
    
    if filtering_matrix != None:
        distance_matrix = np.multiply(distance_matrix, filtering_matrix)
  
    G = nx.from_numpy_matrix(distance_matrix)
    BC = nx.betweenness_centrality(G)
    CC = nx.closeness_centrality(G)
    shortest_paths = dict(nx.all_pairs_shortest_path(G))
    
def computeSimilarityArray(distance_matrix, metric, filtering_matrix=None):

    max_euclidean = np.max(distance_matrix)
    min_euclidean = np.min(distance_matrix)
    
    if filtering_matrix != None:
        distance_matrix = np.multiply(distance_matrix, filtering_matrix.todense().astype(bool))
        
    similarity_array = distance_matrix.sum(1)
    
    # if filtering_matrix==None:
    #     similarity_array = similarity_array / len(similarity_array)
    # else:
    #     similarity_array = similarity_array / filtering_matrix.sum(1)
    
    # normalize if distances are euclidean
    if metric == 'euclidean':
        similarity_array = (similarity_array - min_euclidean)/(max_euclidean-min_euclidean)
    
        #similarity_array = np.array(similarity_array.reshape(-1))[0]#np.array(similarity_array[0])[0]
    return similarity_array

def computeMedianDistanceArray(distance_matrix, metric, filtering_matrix=None):

    max_euclidean = np.max(distance_matrix)
    min_euclidean = np.min(distance_matrix)
    
    if filtering_matrix != None:
        distance_matrix = np.multiply(distance_matrix,filtering_matrix.todense())
        
    similarity_array = np.median(distance_matrix, axis=1)
    
    # if filtering_matrix==None:
    #     similarity_array = similarity_array / len(similarity_array)
    # else:
    #     similarity_array = similarity_array / filtering_matrix.sum(0)
    
    # normalize if distances are euclidean
    if metric == 'euclidean':
        similarity_array = (similarity_array - min_euclidean)/(max_euclidean-min_euclidean)
    
    if filtering_matrix!=None:
        similarity_array = np.array(similarity_array.reshape(-1))[0]#np.array(similarity_array[0])[0]
    return similarity_array

def defineGridOverPolygon(polygon, resolution):
    # credit to: https://stackoverflow.com/questions/66010964/fastest-way-to-produce-a-grid-of-points-that-fall-within-a-polygon-or-shape
    
    # determine polygon edges
    latmin, lonmin, latmax, lonmax = polygon.bounds
    # construct rectangle of points
    y, x = np.round(np.meshgrid(np.arange(lonmin, lonmax, resolution), np.arange(latmin, latmax, resolution)), 4)
    points = MultiPoint(list(zip(x.flatten(),y.flatten())))
    
    # validate each point falls inside shapes
    valid_points = []
    valid_points.extend(list(points.intersection(polygon)))
    
    return valid_points
    
def createCirclesGrid(polygon, resolution, radius):

    points = defineGridOverPolygon(polygon, resolution)
    
    circles = []
    for point in points:
        circles.append(point.buffer(radius))
    
    rings = [LineString(list(pol.exterior.coords)) for pol in circles]
    union = unary_union(rings)
    union = union.intersection(polygon)
    result = [geom for geom in polygonize(union)]
    circle_df = gpd.GeoDataFrame(result, columns = ['geometry'])
    circle_df['intersections'] = 0
    for circle in circles:
        circle_df['intersections'] += circle_df['geometry'].intersects(circle)
        
    return circle_df
    
    # plot resulting polygons
    
    
    # colors = cm.rainbow(np.linspace(0, 1, len(result)))
    # fig, ax = plt.subplots(figsize = (25,25))
    # plot_polys(result,  colors)
    # #plt.ylim([min(y),max(y)])
    # plt.show()
    
# def findLargestConnectedComponent(array):
#    
#     abs_dists = np.abs(array.reshape(-1,1)-array.reshape(1,-1))
#     max_cc = max(nx.connected_components(nx.Graph(abs_dists<20)))
#     argmax_cc = array[list(max_cc)]
#     return [min(argmax_cc), max(argmax_cc)]


# def transformInPolarCoordinates(cartesian_coordinate):
#    
#     x = cartesian_coordinate[0]
#     y = cartesian_coordinate[1]
#    
#     r = np.sqrt(x**2+y**2)
#     # if x!=0:
#     theta = np.arctan2(y,x)
#     # elif x==0 and y>0:
#     #     theta = math.pi/2
#     # elif x==0 and y<0:
#     #     theta = -math.pi/2
#        
#     return (r,theta)
    
    