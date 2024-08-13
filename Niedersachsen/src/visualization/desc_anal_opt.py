# %%
import pickle
import geopandas as gpd
import pandas as pd
import os
import math as m
from functools import reduce # For merging multiple DataFrames

os.chdir('C:/Users/aladesuru/Documents/DataAnalysis/Lab/Niedersachsen')
from src.data import dataload as dl

"""
the main purpose of this script is to generate descriptive statistics for the data
"""

gld = dl.load_data(loadExistingData=True)
gld.info()    
gld.head()    

# %%

kulturcode_mastermap = pd.read_csv('reports/Kulturcode/kulturcode_mastermap.csv', encoding='windows-1252')

# merge the kulturcode_mastermap with data dataframe on 'kulturcode' column
gld = pd.merge(gld, kulturcode_mastermap, on='kulturcode', how='left')
gld = gld.drop(columns=['kulturart', 'kulturart_sourceyear'])
gld.info()


def generate_statistics(gld, output_dir='reports/statistics'):
    # General data descriptive statistics
    gen_stats = gld[['year', 'area_ha', 'peri_m', 'par', 'cpar', 'shp_index', 'fract']].describe()
    gen_stats.to_csv(f'{output_dir}/gen_stats08.08.csv')  # Save to CSV

    # Stats per year
    yearly_genstats = gld.groupby('year')[['area_ha', 'peri_m', 'par', 'cpar', 'shp_index', 'fract']].describe()
    yearly_genstats.to_csv(f'{output_dir}/yearly_genstats08.08.csv')  # Save to CSV

    # Calculate the sum of each column
    column_sums = gld.groupby('year')[['area_ha', 'peri_m', 'par', 'cpar', 'shp_index', 'fract']].sum()
    column_sums['stat'] = 'sum'
    column_sums.to_csv(f'{output_dir}/sums08.08.csv')  # Save to CSV

    # Calculate the median of each column
    column_medians = gld.groupby('year')[['area_ha', 'peri_m', 'par', 'cpar', 'shp_index', 'fract']].median()
    column_medians['stat'] = 'median'
    column_medians.to_csv(f'{output_dir}/medians08.08.csv')  # Save to CSV

    # Combine all statistics into a single DataFrame
    yearly_stats = pd.concat([yearly_genstats, column_sums, column_medians])
    yearly_stats.to_csv(f'{output_dir}/yearly_stats08.08.csv')  # Save to CSV

# Example usage
# generate_statistics(gld)


def generate_statistics(gld, output_dir='reports/statistics'):
    # General data descriptive statistics for all float columns
    float_columns = 
    gen_stats = gld[gld.select_dtypes(include=['float64', 'int32']).columns].describe()
    gen_stats.to_csv(f'{output_dir}/gen_stats08.08.csv')  # Save to CSV






import pandas as pd

    # Stats per year for all float columns
    yearly_genstats = gld.groupby('year')[float_columns].describe()
    yearly_genstats.to_csv(f'{output_dir}/yearly_genstats08.08.csv')  # Save to CSV

    # Calculate the sum of each float column
    column_sums = gld.groupby('year')[float_columns].sum()
    column_sums['stat'] = 'sum'
    column_sums.to_csv(f'{output_dir}/sums08.08.csv')  # Save to CSV

    # Calculate the median of each float column
    column_medians = gld.groupby('year')[float_columns].median()
    column_medians['stat'] = 'median'
    column_medians.to_csv(f'{output_dir}/medians08.08.csv')  # Save to CSV

    # Combine all statistics into a single DataFrame
    yearly_stats = pd.concat([yearly_genstats, column_sums, column_medians])
    yearly_stats.to_csv(f'{output_dir}/yearly_stats08.08.csv')  # Save to CSV

# Example usage
# generate_statistics(gld)



import pandas as pd

def generate_statistics(gld, output_dir='reports/statistics'):
    # Include 'year' column but exclude 'kulturcode' column
    columns_to_include = gld.select_dtypes(include=['float64']).columns.tolist()
    columns_to_include.append('year')
    
    if 'kulturcode' in columns_to_include:
        columns_to_include.remove('kulturcode')
    
    # General data descriptive statistics for the selected columns
    gen_stats = gld[columns_to_include].describe()
    gen_stats.to_csv(f'{output_dir}/gen_stats08.08.csv')  # Save to CSV

    # Stats per year for the selected columns
    yearly_genstats = gld.groupby('year')[columns_to_include].describe()
    yearly_genstats.to_csv(f'{output_dir}/yearly_genstats08.08.csv')  # Save to CSV

    # Calculate the sum of each selected column
    column_sums = gld.groupby('year')[columns_to_include].sum()
    column_sums['stat'] = 'sum'
    column_sums.to_csv(f'{output_dir}/sums08.08.csv')  # Save to CSV

    # Calculate the median of each selected column
    column_medians = gld.groupby('year')[columns_to_include].median()
    column_medians['stat'] = 'median'
    column_medians.to_csv(f'{output_dir}/medians08.08.csv')  # Save to CSV

    # Combine all statistics into a single DataFrame
    yearly_stats = pd.concat([yearly_genstats, column_sums, column_medians])
    yearly_stats.to_csv(f'{output_dir}/yearly_stats08.08.csv')  # Save to CSV

# Example usage
# generate_statistics(gld)