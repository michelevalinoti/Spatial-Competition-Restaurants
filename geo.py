#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 23:12:41 2022

@author: michelev
"""
# pip install shapely
# pip install geopandas
# pip install mapclassify

import Base

import numpy as np
import pandas as pd

from shapely.geometry import MultiPoint
import shapely.wkt
import geopandas as gpd

import geojson
import fiona

import networkx

import matplotlib.pyplot as plt
import mapclassify
import libpysal as ps

from itertools import combinations

import contextily
import cenpy

#%%

class GeoData:
    
    machine = '/Users/michelev/spatial-competition-food/'
    yelp_folder = 'yelp_data/'
    
    platform_folder = 'grubhub_data/'
    platform_census_subfolder = 'census_tracts/'
    platform_analysis_subfolder = 'analysis/'
    
    census_folder = 'nyc_geodata/census_tracts_boundaries/'
    census_filename = 'census_tracts.shp'
    centroids_filename = 'census_tracts_centroids.shp'
    
    distances_folder = 'nyc_geodata/distances/'
    
    census_df = gpd.read_file(machine + census_folder + census_filename)
    census_df = census_df.set_index('BoroCT2020')
    centroids_df = gpd.read_file(machine + census_folder + centroids_filename)
    centroids_df = centroids_df.set_index('BoroCT2020')
    
    #'EPSG:4326'
    CRS_LATLON = 'GCS_WGS_1984'
    CRS_M = 'EPSG:32118'
    
    
    def computeCentroidDistances(self):
        
        centroids_m = self.centroids_df.geometry.set_crs(self.CRS_LATLON).to_crs(self.CRS_M)
        distances = pd.DataFrame(columns = self.centroids_df.index, index = self.centroids_df.index)
        
        ct_idx=0
        for ct in centroids_m.index:
            if ct_idx%100==0:
                print(ct_idx)
            distances.loc[ct] = centroids_m.apply(centroids_m[ct].distance)
            
            ct_idx += 1
            
        distances.to_csv(self.machine + self.distances_folder + 'centroids_distances.csv')
     
    def computePolygonfDistances(self, year = 2020):
        
        census_df = gpd.read_file(self.machine + 'nyc_geodata/census_tracts_boundaries/nyct' + str(year) + '_23a/nyct' + str(year) + '.shp')
        census_df = census_df.set_index('BoroCT' + str(year))
        self.census_df = census_df
        
        census_df = census_df.geometry.set_crs(self.census_df.crs, allow_override=True).to_crs(self.CRS_M)
        euclidean_distances = pd.DataFrame(columns = self.census_df.index, index = self.census_df.index)
        hausdorff_distances = pd.DataFrame(columns = self.census_df.index, index = self.census_df.index)
        
        ct_idx=0
        for ct in census_df.index:
            if ct_idx%100==0:
                print(ct_idx)
            euclidean_distances.loc[ct] = census_df.apply(census_df[ct].distance)
            hausdorff_distances.loc[ct] = census_df.apply(census_df[ct].hausdorff_distance)
            
            ct_idx += 1
            
        
        euclidean_distances.to_csv(self.machine + self.distances_folder + 'euclidean_distances_' + str(year) + '.csv')
        hausdorff_distances.to_csv(self.machine + self.distances_folder + 'hausdorff_distances_' + str(year) + '.csv')
        
        #cross_CT=pd.DataFrame(self.census_df.BoroName).merge(pd.DataFrame(self.census_df.BoroName).T, how='cross').set_index(self.census_df.index)
        
        Manhattan_correction=np.outer(self.census_df.BoroName=='Manhattan',self.census_df.BoroName!='Manhattan')
        Manhattan_correction=Manhattan_correction+Manhattan_correction.T
        QueensBronx_correction=np.outer(self.census_df.BoroName=='Bronx',self.census_df.BoroName=='Queens')
        QueensBronx_correction=QueensBronx_correction+QueensBronx_correction.T
        corrections = np.logical_or(Manhattan_correction,QueensBronx_correction)
        euclidean_distances_corrected = euclidean_distances.copy()
        hausdorff_distances_corrected = hausdorff_distances.copy()
        
        euclidean_distances_corrected[corrections] = np.nan
        hausdorff_distances_corrected[corrections] = np.nan
        
        euclidean_distances_corrected[euclidean_distances_corrected==0] = 1
        
        euclidean_distances_corrected.to_csv(self.machine + self.distances_folder + 'euclidean_distances_corrected_' + str(year) + '.csv')
        hausdorff_distances_corrected.to_csv(self.machine + self.distances_folder + 'hausdorff_distances_corrected_' + str(year) + '.csv')
        
        
    def shortestPaths(self):
        
        dists = pd.read_csv(self.machine + self.distances_folder + 'linear_distances.csv', index_col=0)
        G = networkx.from_numpy_matrix(dists.values)
        shortest_paths = dict(networkx.eigenvector_centrality(G))
        
    def findCensusTractsWithinDistance(self, filename, cutoff=1000):
        
        dists = pd.read_csv(self.machine + self.distances_folder + filename + '.csv', index_col=0)
        ct_within = dists<cutoff
        ct_within.to_csv(self.machine + self.distances_folder + 'ct_' + filename + '_' + str(cutoff)  + '.csv')
        ct_dummies = pd.DataFrame(index=ct_within.index, columns=['within'])
        for idx in ct_within.index:
            ct_dummies.loc[idx,'within'] = ct_within.index[np.where(ct_within.loc[idx])[0]].tolist()
        ct_dummies.to_csv(self.machine + self.distances_folder + 'ct_dummies_' + filename + '_' + str(cutoff)  + '.csv')

    def enlargeWithYelp(self):
        
        census_w_restaurants = pd.read_csv(self.machine + self.yelp_folder + 'number_rests_in_tracts.csv')
        census_w_restaurants = census_w_restaurants.set_index('BoroCT2020')
        self.census_df = self.census_df.join(census_w_restaurants, on = 'BoroCT2020', how = 'right', lsuffix = '_geo', rsuffix = '_yelp')
        
        fig, ax = plt.subplots(figsize = (20,20))
        self.census_df.plot('NumberYelpRestaurants', ax=ax,  cmap ='inferno_r', legend=True, legend_kwds={'shrink': 0.4})
        fig.savefig('figures/yelp_restaurants.png',bbox_inches='tight')
        
        # census_w_restaurants = pd.read_csv(self.machine + self.platform_folder + self.platform_analysis_subfolder + 'restaurants_in_tracts.csv')
        # census_w_restaurants = census_w_restaurants.set_index('BoroCT2020')
        # self.census_df = self.census_df.join(census_w_restaurants, on = 'BoroCT2020', how = 'right', lsuffix = '_geo', rsuffix = '_yelp')
        # fig, ax = plt.subplots(figsize = (20,20))
        # self.census_df.plot('NumberGHRestaurants', ax=ax,  cmap ='inferno_r', legend=True, legend_kwds={'shrink': 0.4})
        # fig.savefig('figures/GH_restaurants.png',bbox_inches='tight')
        
    def enlargeWithGrubhub(self):
        
        census_w_restaurants = pd.read_csv(self.machine + self.platform_folder + self.platform_analysis_subfolder + 'delivering_restaurants_in_tracts.csv')
        census_w_restaurants = census_w_restaurants.set_index('BoroCT2020')
        self.census_df = self.census_df.join(census_w_restaurants, on = 'BoroCT2020', how = 'right', lsuffix = '_geo', rsuffix = '_gh')
    
        fig, ax = plt.subplots(figsize = (20,20))
        self.census_df.plot('NumberGHDeliveringRestaurants', ax=ax, cmap ='RdYlBu_r', legend=True, legend_kwds={'shrink': 0.4})
        fig.savefig('figures/gh_to.png', bbox_inches='tight')
        
        restaurants_w_census = pd.read_csv(self.machine + self.platform_folder + self.platform_analysis_subfolder + 'restaurants_in_tracts.csv')
        restaurants_w_census = restaurants_w_census.set_index('BoroCT2020')
        self.census_df = self.census_df.join(restaurants_w_census, on = 'BoroCT2020', how = 'right', lsuffix = '_geo', rsuffix = '_gh')
        
        fig, ax = plt.subplots(figsize = (20,20))
        self.census_df.plot('NumberGHRestaurants_gh', ax=ax, cmap ='inferno_r', legend=True, legend_kwds={'shrink': 0.4})
        fig.savefig('figures/gh_from.png', bbox_inches='tight')
        # restaurants_in_census = pd.read_csv(self.machine + self.platform_folder + self.platform_analysis_subfolder + 'restaurants_delivering_from_tracts.csv')
        # self.census_df = self.census_df.merge(restaurants_in_census.groupby(['BoroCT2020'], as_index=False).size(), on = 'BoroCT2020')
        # self.census_df = self.census_df.rename({'size':'NumberGHRestaurants'}, axis=1)
        # fig, ax = plt.subplots(figsize = (50,50))
        # self.census_df.plot('NumberGHRestaurants', ax=ax, cmap ='RdYlBu', legend=True, legend_kwds={'shrink': 0.3})
        
        # fig, ax = plt.subplots(figsize = (50,50))
        # self.census_df.plot('NumberGHDeliveringRestaurants', scheme='quantiles', k=10, cmap ='RdYlBu', legend=True, legend_kwds={'fontsize': 50, 'loc': 'lower right'}, ax=ax)
    def findNeighborsNTAs(self):
        
        dummiesNTA = pd.get_dummies(self.census_df.NTA2020.apply(pd.Series).stack()).sum(level=0)
        sameNTA =  dummiesNTA.dot(dummiesNTA.T)
        
        sameNTA.to_csv(self.census_folder + '/same_NTA.csv')
        

    def findNeighboringTracts(self):
        
        ct_names = self.census_df.index
        touching_ct = pd.DataFrame(index = ct_names, columns= ct_names)
        
        for i in range(len(ct_names)):
            
            if i%100==0:
                print(i)
            polygon_i = self.census_df.geometry[ct_names[i]]
            
            for j in range(i, len(ct_names)):
                
                if j==i:
                    
                    touching_ct.iloc[i,j] = 1
                    
                elif polygon_i.touches(self.census_df.geometry[ct_names[j]]):
                    
                    touching_ct.iloc[i,j] = 1
                    touching_ct.iloc[j,i] = 1
                     
                elif polygon_i.overlaps(self.census_df.geometry[ct_names[j]]):
                    
                    touching_ct.iloc[i,j] = 1
                    touching_ct.iloc[j,i] = 1
                    
                else:
                    
                    touching_ct.iloc[i,j] = 0
                    touching_ct.iloc[j,i] = 0
                    
        touching_ct.to_csv(self.machine + self.census_folder + 'touching_census_tracts.csv')
        
        # same matrix but without a 1 if i==j
        touching_ct_0 = touching_ct-np.eye(len(ct_names))
        
        closed_touching_ct=touching_ct_0.copy()
        # close the triads: check for any tract the neighbors, and for each pair of neighbors, find the common (it's a product)
        for i in range(len(ct_names)):
            
            if i%100==0:
                print(i)
            neighbors_i = np.nonzero(touching_ct_0.iloc[i,:].values)[0]
            neighbors_pairs_i =  list(combinations(neighbors_i, 2))
            for (j,k) in neighbors_pairs_i:
                common_neighbors_pairs_jk = np.multiply(touching_ct_0.iloc[j,:].values,touching_ct_0.iloc[k,:].values)
                common_neighbors_pairs_jk = np.nonzero(common_neighbors_pairs_jk)[0]
                closed_touching_ct.iloc[i,common_neighbors_pairs_jk]=1
        
            closed_touching_ct.iloc[i,i]=1

        closed_touching_ct.to_csv(self.machine + self.census_folder + 'neighboring_census_tracts_1.csv')
    
    def findNeighborsAtDistance(self, dist=2):
        
        common_neighbors_1 = pd.read_csv(self.machine + self.census_folder + 'neighboring_census_tracts_1.csv', index_col=0)
        
        #! Adjust parks!
        common_neighbors_2 = common_neighbors_1.T.dot(common_neighbors_1)
        common_neighbors_2 = common_neighbors_2>0
        common_neighbors_2.to_csv(self.machine + self.census_folder + 'neighboring_census_tracts_2.csv')
        
    
    def main(self):
        
        GD = GeoData()
        #GD.computeDistancesMatrix()
        GD.findCensusTractsWithinDistance('linear_distances', 5000)
        #GD.findNeighboringTracts()
        #GD.enlargeWithYelp()
        #GD.enlargeWithGrubhub()
    
        input_folder = 'nyc_geodata/boroughs_boundaries/'
        destination_folder = 'nyc_geodata/locations_grid/'
        
        boroughs_shape = gpd.read_file(machine + input_folder + 'geo_export_819798c2-2e2b-492d-b9b7-23d6b5f9d35a.shp')
        boroughs_df = pd.DataFrame(columns = boroughs_shape.columns)
        for bb in range(len(boroughs_shape)):
            borough_data = [boroughs_shape.iloc[bb, col] for col in range(4)]
            borough_shape = boroughs_shape.iloc[bb]['geometry']
            borough_grid = []
            for polygon in borough_shape:
                xmin, ymin, xmax, ymax = polygon.bounds
                n = 250
                x = np.arange(np.floor(xmin*n)/n, np.ceil(xmax*n)/n, 1/n)  
                y = np.arange(np.floor(ymin*n)/n, np.ceil(ymax*n)/n, 1/n)
                points = MultiPoint(np.transpose([np.tile(x, len(y)), np.repeat(y, len(x))]))
                result = points.intersection(polygon)
                if (result.is_empty == False) & (result.geom_type == 'MultiPoint'):
                    borough_grid.append(result)
            borough_data.append(borough_grid)
            boroughs_df.loc[bb] = borough_data
        boroughs_df = boroughs_df.explode('geometry')
        boroughs_df = boroughs_df.explode('geometry')
        gpd.GeoDataFrame(boroughs_df).to_file(machine + destination_folder + 'locations_grid_' + str(n) + '.shp')
        
        input_folder = 'nyc_geodata/census_tracts_boundaries/'
        
        # Import geo data of census tracts
        census_shape = pd.read_csv(machine + 'nyc_geodata/census_tracts_boundaries/2020_Census_Tracts_-_Tabular.csv')
        census_shape = gpd.GeoDataFrame(census_shape)
        census_shape['geometry'] = census_shape['the_geom'].apply(shapely.wkt.loads)
        census_shape = census_shape.set_geometry('geometry')
        census_shape.pop('the_geom')
        census_shape.to_file(machine + input_folder + 'census_tracts.shp')
        
        
        # The file (2020) should contain 2325 census tracts
        # Some of them (44) are multipolygons because they typically include some offshore land
        # I add another column in which I retrieve the polygon with the largest area
        # I checked manually that no important land is lost
        # This is important when computing centroids of the areas
        # Original multipolygons may still matter to compute areas, etc.
        
        census_shape['geometry_main'] = census_shape['geometry'].copy()
        census_shape = census_shape.set_geometry('geometry_main')
        centroids = []
        for idx in census_shape.index:
            
            if len(census_shape.loc[idx, 'geometry'])>1:
                multi_polygon = census_shape.loc[idx,'geometry']
                largest_polygon = multi_polygon[0]
                for polygon in multi_polygon:
                    area = polygon.area
                    if largest_polygon.area < area:
                        largest_polygon = polygon
                census_shape.loc[idx,'geometry_main'] = largest_polygon
            centroids.append(census_shape.loc[idx,'geometry_main'].centroid)
        
        census_shape.pop('geometry')
        census_shape = census_shape.set_geometry('geometry_main')
        census_shape.to_file(self.machine + input_folder + 'census_tracts_largest.shp')
        
        census_shape['centroid'] = gpd.GeoSeries(centroids)
        census_shape = census_shape.set_geometry('centroid')
        census_shape.pop('geometry_main')
        census_shape.to_file(machine + input_folder + 'census_tracts_centroids.shp')

def findZoningDistricts(self):
    
        zoning = gpd.read_file(self.machine + 'nyc_geodata/zoning_laws/nycgiszoningfeatures_202210shp/' + 'nyzd.shp')
        
        zoning['District'] = np.nan
        for distr in ['C', 'R', 'M']:
            zoning[distr] =0
            zoning.loc[zoning.ZONEDIST.apply(lambda s: s.startswith(distr)),distr] = 1
            zoning.loc[zoning.ZONEDIST.apply(lambda s: s.startswith(distr)),'District'] = distr
            
        zoning['District'].fillna('P',inplace=True)
        
        census_zoning = gpd.read_file(self.machine + 'nyc_geodata/census_tracts_boundaries/census_tracts_largest.shp').copy()
        census_zoning.geometry = census_zoning.geometry.set_crs(self.CRS_LATLON).to_crs(zoning.crs)
        census_zoning.geometry = census_zoning.geometry.buffer(0)
        zoning.geometry = zoning.geometry.buffer(0)
        
        for district in ['C', 'R', 'M', 'P']:
            census_zoning[district]=0
            
        i=0
        for census in census_zoning.index:
            
            if i%100==0:
                print(i)
            census_area = census_zoning.loc[census, 'geometry'].area
            for poly in zoning.index:
                
                try:
                    intersection_area = census_zoning.loc[census, 'geometry'].intersection(zoning.loc[poly,'geometry']).area
                except:
                    print('Error')
                    intersection_area = 0
                    
                census_zoning.loc[census, zoning.loc[poly,'District']] += intersection_area/census_area
            i+=1
            
        census_zoning.to_csv(self.machine + 'nyc_geodata/zoning_laws/census_w_districts.csv')
        
    
def LEHD():
    
    census_blocks = gpd.read_file(machine + 'census_tract_data/nycb2020_22c/nycb2020.shp')
    census_blocks = pd.DataFrame(census_blocks.loc[:,census_blocks.columns!='geometry'])
    census_blocks['BoroCT2020'] = census_blocks.BoroCode.astype(str) + census_blocks.CT2020.astype(str)
    census_blocks['BoroCT2020'] = census_blocks['BoroCT2020'].astype(int)
    census_blocks.GEOID = census_blocks.GEOID.astype(int)
    rac = pd.read_csv(machine + 'census_tract_data/ny_rac_S000_JT00_2019.csv')
    wac = pd.read_csv(machine + 'census_tract_data/ny_wac_S000_JT00_2019.csv')
    
    rac = rac.rename(columns = {'h_geocode':'GEOID'})
    wac = wac.rename(columns = {'w_geocode':'GEOID'})
    
    rwac = rac.merge(wac, on = 'GEOID', how='outer', suffixes=('_r', '_w'))
    rwac_CB= census_blocks.merge(rwac, on = 'GEOID', how = 'left')
    rwac_CT = rwac_CB.groupby('BoroCT2020').sum().reset_index()
    
    rwac_CT.to_csv(machine + 'census_tract_data/rwac_CT.csv')
   
def constructCensusData(self):
    
    ACS_data = pd.read_csv(self.machine + 'census_tract_data/ACS_data.csv')
    ACS_data['estimate_population'] /= 1000
    ACS_data['estimate_median_income'] /= 1000
    non_white_frac = 1-ACS_data['frac_white']
    races = ['white', 'black', 'american_indian', 'asian', 'pacific', 'other_one', 'other_two']
    hhi_all_races = 0
    hhi_non_white = 0
    for race in races:
        hhi_all_races += ACS_data['frac_'+race]**2
        hhi_non_white += (ACS_data['frac_'+race]/non_white_frac)**2
        ACS_data['hhi_all_races'] = hhi_all_races
    ACS_data['hhi_non_white'] = hhi_non_white
    
    census_zoning = pd.read_csv(self.machine + 'nyc_geodata/zoning_laws/census_w_districts.csv')
    
    rwac_CT = pd.read_csv(self.machine + 'census_tract_data/rwac_CT.csv', index_col=[0])
    rwac_CT['C000_w'] = rwac_CT['C000_w']/1000
    
    dataset = census_zoning.merge(ACS_data, on = 'GEOID', how = 'left')
    dataset = dataset.merge(rwac_CT[['BoroCT2020', 'C000_w']], on = 'BoroCT2020')
    
    dataset['density_population'] = dataset['estimate_population']/dataset['Shape_Area']*10e6
    dataset['density_workers'] = dataset['C000_w']/dataset['Shape_Area']*10e6
       
    rents = pd.read_csv('nyc_geodata/rents/Zip_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv')
    rents = rents.rename({'RegionName':'zip', '2022-10-31': 'ZHVI'}, axis='columns')
    rents = rents.loc[(rents.City == 'New York')][['zip', 'ZHVI']]
    
    tract_zips = pd.read_excel(self.machine + 'census_tract_data/TRACT_ZIP_122021.xlsx')
    tract_zips = tract_zips.loc[tract_zips['usps_zip_pref_state']=='NY']
    tract_zips = tract_zips.rename({'tract':'GEOID'}, axis='columns')
    
    temp = dataset[['BoroCT2020','GEOID']].merge(tract_zips[['GEOID','zip','tot_ratio']].merge(rents, on = 'zip', how='left'), on = 'GEOID', how='left')
    temp['ZHVI'] = temp['ZHVI']*temp['tot_ratio']
    temp = temp.groupby(['BoroCT2020','GEOID']).sum()
    temp = temp.reset_index()
    
    dataset = dataset.merge(temp[['ZHVI', 'BoroCT2020']], how = 'left', on = 'BoroCT2020')
    dataset['ZHVI'] /= 1e6
    
    dataset.to_csv(self.machine + 'census_tract_data/merged_dataset.csv')
    
def writeSummaryCensusTractData(self):
    
    dataset = pd.read_csv(self.machine + 'census_tract_data/merged_dataset.csv')
    
    varss = ['BoroName', 'Shape_Area', 'estimate_population', 'C000_w', 'density_population', 'density_workers',
             'frac_white', 'frac_black', 'frac_asian', 'hhi_all_races', 'ZHVI', 'C', 'R', 'M', 'P']
    
    short_df = dataset[varss]
    
    short_df = short_df.join(pd.get_dummies(short_df.BoroName).groupby(level=0).sum())
    short_df['Shape_Area'] /= 1e6
    
    summary = {}
    
    rename_dict1 = {'Shape_Area': 'Area ($km^2$)'}
    rename_dict2 = {'estimate_population': 'Resident population (1000s)',
                    'estimate_population': 'Num. of workers (1000s)',
                    'density_population': 'Density population (1000s/$km^2$)',
                    'density_workers': 'Density workers (1000s/$km^2$)',
                    'frac_white': 'Fraction white',
                    'frac_black': 'Fraction black',
                    'frac_asian': 'Fraction asian',
                    'hhi_all_races': 'Race concentration index'}
    rename_dict3 = {'ZHVI': 'Home value index'}
    rename_dict4 = {'C': 'Commercial district',
                    'R': 'Residential district',
                    'M': 'Manifacture district',
                    'P': 'Other (parks)'
                    }
    
    
    short_df = short_df.rename(rename_dict1, axis=1)
    short_df = short_df.rename(rename_dict2, axis=1)
    short_df = short_df.rename(rename_dict3, axis=1)
    short_df = short_df.rename(rename_dict4, axis=1)
    
    for var in rename_dict1.values():
        
        summary['', var] = [np.mean(short_df[var]), np.std(short_df[var]), np.min(short_df[var]), np.max(short_df[var]), np.nan]
        
    for var in rename_dict2.values():
        
        summary['Borough', var] = [np.mean(short_df[var]), np.std(short_df[var]), np.min(short_df[var]), np.max(short_df[var]), np.nan]
    
    for var in rename_dict3.values():
        
        summary['', var] = [np.mean(short_df[var]), np.std(short_df[var]), np.min(short_df[var]), np.max(short_df[var]), np.nan]
    
    for var in rename_dict4.values():
        
        summary['Zoning', var] = [np.mean(short_df[var]), np.std(short_df[var]), np.min(short_df[var]), np.max(short_df[var]), np.nan]
    
    summary[('Total','')] = [np.nan, np.nan, np.nan, np.nan, len(short_df)]
    summary = pd.DataFrame(summary).T
    summary = summary.rename({0:'Mean',1:'Std. dev.', 2:'Min', 3:'Max', 4:'Count'}, axis=1)
    summary = summary.round(2)
    #summary['Min'] = summary['Min'].map('{:.0f}'.format)
    #summary['Max'] = summary['Max'].map('{:.0f}'.format)
    summary['Count'] = summary['Count'].map('{:.0f}'.format)
    summary[summary=='nan']=np.nan
    #summary.loc['Total menu items',:] = summary.loc['Total menu items',:].map('{:.0f}'.format)
    file_name = 'tables/summary_CT.tex'
    tex_file = open(file_name, 'w')
    tex_file.write(summary.to_latex(sparsify = True,
                                    escape=False,
                                    #column_format = 'lcr',
                                    bold_rows = True,
                                    na_rep='',
                                    caption = '',
                                    label = 'tab:summary_CT'))
    tex_file.close()
    
    #%%
    
    # census_specific_data = pd.read_csv(machine + 'grubhub_data/analysis/census-specific-data.csv')
    # census_specific_data.rename({'census_tract':'BoroCT2020'}, axis=1,inplace=True)
    # median_receiving = census_specific_data.groupby('BoroCT2020').median()[['distance_from_location', 'delivery_fee', 'delivery_time_estimate']]
    # median_departing = census_specific_data[['restaurant_id','distance_from_location', 'delivery_fee', 'delivery_time_estimate']].merge(merged_df[['restaurant_id', 'BoroCT2020']], on = 'restaurant_id').groupby('BoroCT2020').median()[['distance_from_location', 'delivery_fee', 'delivery_time_estimate']]
    
    # cc=census_df.join(median_receiving, on = 'BoroCT2020', rsuffix='_rec')
    # median_departing = median_departing.reset_index()
    # median_departing.BoroCT2020 = median_departing.BoroCT2020.astype(int)
    # cc =cc.merge(median_departing, how='left', on = 'BoroCT2020', suffixes=('_rec','_dep'))
    
    # df2 = df2.merge(cc[['BoroCT2020','distance_from_location_rec', 'delivery_fee_rec', 'delivery_time_estimate_rec']], on = 'BoroCT2020', how='left')
    
    # dataset=dataset_c#[(dataset_c['NumberGHRestaurants']>0)]#[((dataset_c.price=='$$') | (dataset_c.price=='$')) & (dataset_c.isAtLeastRestaurant)]
    # missing_kwds = dict(color='lightgrey', hatch = '///')
    # on = 'NTA2020'
    # here=dataset.groupby(on).median()#quantile(0.5)
    # #here=dataset[(dataset['isGH']==True) & (dataset['delivery_mode'] == 'FULL_GHD')].groupby(on).quantile(0.75)#max()
    # #here=dataset[(dataset['isGH']==True<<)].groupby(on).mean()-dataset[(dataset['isGH']==False)].groupby(on).mean()
    # #here=dataset[(dataset['isGH']==True) & (dataset['delivery_mode'] == 'FULL_GHD')].groupby(on).median()-dataset[(dataset['isGH']==False) & (dataset['delivery_mode'] != 'OFF')].groupby(on).median()
    # here['NTA_2020']=here.index
    # here=census_df.join(here,how='left',on=on, lsuffix='_geo', rsuffix='_df')
    # fig, ax = plt.subplots(figsize = (50,50))
    # here.plot('similarity_NTA_all', ax=ax,legend=True, cmap ='RdYlBu',  legend_kwds={'shrink': 0.3}, missing_kwds=missing_kwds)
    # fig, ax = plt.subplots(figsize = (50,50))
    # here.plot('similarity_all', legend=True, missing_kwds=missing_kwds,  cmap = plt.cm.RdYlBu, ax=ax, legend_kwds={'shrink': 0.3})
    # fig, ax = plt.subplots(figsize = (50,50))
    # here.plot('similarity_diff_all', legend=True, missing_kwds=missing_kwds,  cmap = plt.cm.RdYlBu, ax=ax, legend_kwds={'shrink': 0.3})

        