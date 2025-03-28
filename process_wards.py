######################
#
# this file contains the code necessary to obtain the distances.csv part of the dataset
#
######################

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon
import requests
from concurrent.futures import ThreadPoolExecutor
import time
from tqdm import tqdm
import warnings
import os

warnings.filterwarnings("ignore")


def fetch_boundary_data(ward: str):
    # Function to fetch the boundary data for a given ward

    url = f"https://data.police.uk/api/metropolitan/{ward}/boundary"
    response = requests.get(url)
    # if ward is not metropolitan, use hertfordshire api
    if response.status_code != 200:

        url = f"https://data.police.uk/api/hertfordshire/{ward}/boundary"
        response = requests.get(url)

    if response.status_code != 200:
        print(response.status_code)

    return pd.DataFrame(response.json())

def getDistances(df_wards):
    """function that takes geodataframe of wards with 'centroid' column and returns df of relative distances between centroids
    """
    # Create an empty DataFrame for the distances
    distances_df = pd.DataFrame(index=df_wards['name'], columns=df_wards['name'])

    # Calculate the distances between all pairs of centroids
    for i, centroid_i in zip(df_wards['name'], df_wards['centroids']):
        for j, centroid_j in zip(df_wards['name'], df_wards['centroids']):
            distances_df.at[i, j] = centroid_i.distance(centroid_j)
    return distances_df


def getNeigh():
    """calls to API to get all neighbourhoods and returns df
    """
    # API endpoint for all neighbourhoods in metropolitan and hertfortshire area
    boroughs = ["metropolitan", "hertfordshire"]
    df_wards = pd.DataFrame(columns=["id", "name", "borough"])
    for borough in boroughs:

        url = f"https://data.police.uk/api/{borough}/neighbourhoods"
        # make a get request to the url
        response = requests.get(url)
        if response.status_code != 200:
            raise Exception(f'Request to {url} failed with status code: {response.status_code}')
        else:
            # convert the response to a pandas dataframe, add borough column and append to df_wards
            df_wards = df_wards.append(pd.DataFrame(response.json()).assign(borough=borough))

    # set index to id
    df_wards.set_index("id", inplace=True)

    return df_wards

def getBoundaries(df_wards, max_workers=5, delay=.01):
    """gets df of all neighbourhoods and calls to API to get boundaries, returns gdf with 'geometry', 'centroid' columns.
    max_workers: n of concurrent api calls
    delay: seconds of delay inbetween batches of api calls
    """
    # Fetch the boundary data in parallel using ThreadPoolExecutor with limited concurrent requests
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        boundary_data = []
        for future in tqdm(executor.map(fetch_boundary_data, df_wards.index.tolist()), total=df_wards.shape[0]):
            time.sleep(delay)
            boundary_data.append(future)
    df_wards["longitude"] = [df["longitude"].to_numpy().astype(float) for df in boundary_data]
    df_wards["latitude"] = [df["latitude"].to_numpy().astype(float) for df in boundary_data]
    # Remove the first row because it was a duplicate
    df_wards = df_wards.iloc[1:]
    # Create a new column 'boundaries' containing a list of tuples of the longitude and latitude
    df_wards["boundaries"] = df_wards.apply(lambda x: list(zip(x["latitude"], x["longitude"])), axis=1)
    # Create a new column 'geometry' containing boundaries as shapely Polygon object
    df_wards['geometry'] = df_wards['boundaries'].apply(lambda x: Polygon(x))
    # Turn pandas df in geopandas gdf
    gdf_wards = gpd.GeoDataFrame(df_wards, geometry='geometry')
    # define function to get centroid of every ward
    get_centroid = lambda polygon: polygon.centroid
    # Use the apply function to create a new column with the centroids
    gdf_wards['centroids'] = df_wards['geometry'].apply(get_centroid)
    return gdf_wards

def getAll():
    """main, gets and stores to csv the datasets
    """
    print('Getting Neighbours...')
    df_wards = getNeigh()
    print('Getting Boundaries...')
    gdf_wards = getBoundaries(df_wards)
    print('Getting Distances...')
    df_distances = getDistances(gdf_wards)
    print('Saving Dataframes...')

    if not os.path.exists('data'):
        os.makedirs('data')

    gdf_wards.to_csv('data/df_wards.csv')
    gdf_wards.drop(columns=['latitude', 'longitude', 'boundaries', 'centroids'], inplace=True)
    gdf_wards.to_file("data/df_wards.shp")
    df_distances.to_csv('data/df_distances.csv')
    return df_distances, gdf_wards

if __name__ == '__main__':
    getAll()
