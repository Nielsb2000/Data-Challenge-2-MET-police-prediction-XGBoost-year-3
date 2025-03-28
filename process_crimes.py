#####################
#
# this file contains everything needed to process the data, assuming it is already downloaded on the machine.
# also assumes df_wards.csv is already present on the machine
#
#####################

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
import os
from datetime import datetime
import tqdm
# import modules
from process_wards import getAll
from extract_zip import extract_zip


def merge(path='data/raw/'):
    """assuming dowloaded data zip file is extracted in path, returns merged df
    """
    path = Path(path)
    dfs = []
    for dirpath, dirnames, filenames in os.walk(path):  # walk through data directory
        for dirname in dirnames:                        # walk through subdirectories
            subdir = os.path.join(dirpath, dirname)
            for f in os.listdir(subdir):                # walk through files in subdirectories
                if not (f.endswith('stop-and-search.csv') or f.endswith('outcomes.csv') or f.endswith('.zip') or f == '.gitignore'):
                    dfs.append(pd.read_csv(os.path.join(subdir, f), usecols=lambda column: column != 'Context', dtype={'Latitude': np.float64, 'Longitude': np.float64},encoding='latin-1'))
    df = pd.concat(dfs, ignore_index=False)
    return df

def find_polygon_name_fast(point, polygons_gdf, spatial_index, name_col):
    """fast find in which polygon is the point contained
    """
    try:
        possible_matches_index = list(spatial_index.intersection(point.bounds))
        possible_matches = polygons_gdf.iloc[possible_matches_index]
    except Exception:
        return None
    for idx, row in possible_matches.iterrows():
        if row['geometry'].contains(point):
            return row[name_col]
    return None

def loadWards():
    _, gdf_wards = getAll()
    #gdf_wards = gpd.read_file('data/df_wards.shp')
    gdf_wards = gdf_wards.set_crs(epsg=4326)  # WGS84
    return gdf_wards

def findWards(df):
    geometry = [Point(xy) for xy in zip(df['Latitude'], df['Longitude'])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry)
    gdf = gdf.set_crs(epsg=4326)  # WGS84
    gdf_wards = loadWards()
    print(f'Matching crimes to ward, this will take a while\nstarted at {datetime.now().strftime("%H:%M")}')

    tqdm.tqdm.pandas()
    gdf['wards'] = gdf['geometry'].progress_apply(lambda point: find_polygon_name_fast(point, gdf_wards, gdf_wards.sindex, 'name'))

    #gdf['wards'] = gdf['geometry'].apply(lambda point: find_polygon_name_fast(point, gdf_wards, gdf_wards.sindex, 'name'))
    return gdf


def process(df, agg_level):
    if agg_level == 'wards':
        df.dropna(how='any', subset=['Longitude', 'Latitude', 'Crime type', 'wards'], inplace=True)
        df_tiny = df.drop(columns=['Reported by', 'geometry', 'Falls within', 'Location', 'LSOA name'])
    else:
        df.dropna(how='any', subset=['Longitude', 'Latitude', 'Crime type', 'LSOA code'], inplace=True)
        df_tiny = df.drop(columns=['Reported by', 'Falls within', 'Location', 'LSOA name'])
    # Group by LSOA code, Crime type, and Month, then count the number of crimes
    crime_count = df_tiny.groupby([agg_level, 'Crime type', 'Month']).size().reset_index(name='Count')
    crime_pivot = crime_count.pivot_table(index=[agg_level, 'Month'], columns='Crime type', values='Count', fill_value=0).reset_index()
    
    if agg_level == 'LSOA code':
        crime_pivot['Year'] = pd.to_datetime(crime_pivot['Month']).dt.year
        density_df = pd.read_csv("data/df_density.csv")
        df_age = pd.read_csv("data/df_age.csv")
        crime_pivot = pd.merge(crime_pivot, density_df, on=['LSOA code', 'Year'], how='left')
        crime_pivot = pd.merge(crime_pivot, df_age, on=['LSOA code', 'Year'], how='left')
    return crime_pivot


def clean(crime_pivot):
    wards_encoder = LabelEncoder()
    wards_encoder.fit(crime_pivot['wards'])
    crime_pivot['Encoded wards'] = wards_encoder.transform(crime_pivot['wards'])
    crime_pivot['Month'] = pd.to_datetime(crime_pivot['Month'])
    reference_point = crime_pivot['Month'].min()
    crime_pivot['Encoded Month'] = (crime_pivot['Month'].dt.year - reference_point.year) * 12 + (crime_pivot['Month'].dt.month - reference_point.month)
    scaler = MinMaxScaler()
    crime_pivot['Encoded Month'] = scaler.fit_transform(crime_pivot[['Encoded Month']])

    crime_count_columns = crime_pivot.columns[2:-2]
    scalers = {}
    for column in crime_count_columns:
        scaler = MinMaxScaler()
        crime_pivot[f'Normalized {column}'] = scaler.fit_transform(crime_pivot[[column]])
        scalers[column] = scaler
    filtered_df = crime_pivot.filter(regex='^(Encoded|Normalized)')
    return filtered_df


def main(sample=True, agg_level='w'):
    """main that saves gdf of the crimes with corresponding ward they happened in.
    """
    if not any(os.path.isdir(os.path.join('data/raw/', name)) for name in os.listdir('data/raw/')):
        extract_zip()
    print('only first 10,000 rows are processed') if sample else print('processing all data')
    agg_level = 'wards' if agg_level == 'w' else 'LSOA code' if agg_level == 'l' else None
    print(f'aggregating data on {agg_level} level')
    print('Merging data files...')
    df_full = merge()
    if sample:
        df = df_full.iloc[:10000].copy()
    else:
        df = df_full
    if agg_level == 'wards':
        gdf = findWards(df)
        df = pd.DataFrame(gdf)
    df.to_csv(f'data/df_crimes_non_agg.csv')
    df = process(df, agg_level)
    print('Saving df to file...')
    df = df.sort_values(by=['Month', agg_level])
    print('\n'+'-'*20 + 'final dataset' + '-'*20)
    print(df.info())
    df.to_csv(f'data/df_crimes_{agg_level.replace(" ", "_")}.csv')
    #print('Normalizing, Encoding and storing...')
    #df_norm = clean(df)
    #df_norm.to_csv('data/df_crimes_normalized.csv')


if __name__ == '__main__':
    sample = lambda question: next((x == 'y' for x in iter(lambda: input(question + ' (y/n): ').lower().strip(), '') if x in ['y', 'n']), True)
    agg_level = lambda question: next((x for x in iter(lambda: input(question + ' (l/w): ').lower().strip(), '') if x in ['l', 'w']), None)
    main(sample=sample('Do you want to preprocess just a sample of the data? (10 000)'), agg_level=agg_level('Choose aggregation level: (l)soa, (w)ards'))
