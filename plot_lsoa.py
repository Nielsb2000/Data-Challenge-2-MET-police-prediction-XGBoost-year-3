# want to plot the LSAO data
# boundaries are in shapely format

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
from shapely.geometry import Polygon
import folium
from branca.colormap import LinearColormap
import webbrowser
import argparse
import geopandas as gpd


# load the data
def load_data():
    path = r"data/lsoa_london.shp"
    lsoa_london = gpd.read_file(path)
    df = pd.read_csv('data/y_true_pred.csv')
    # Generate random values within the range
    random_values = np.random.uniform(0, 40, len(lsoa_london))
    # Add the random values as a new column to the GeoDataFrame
    lsoa_london['y'] = random_values
    return lsoa_london


# function to only have barnet
def get_barnet_area(lsoa_london):
    barnet_neighbourhoods = ["Barnet", "Harrow", "Hertsmere", "Enfield", "Haringey", "Camden", "Brent"]
    lsoa_barnet = lsoa_london[lsoa_london.borough.isin(barnet_neighbourhoods)]
    return lsoa_barnet


def plot_folium(lsoa_df):
    # Get the minimum and maximum values of the numerical column
    min_value = lsoa_df['y'].min()
    max_value = lsoa_df['y'].max()
    # Create a linear colormap
    colormap = LinearColormap(['blue', 'red'], vmin=min_value, vmax=max_value)
    # Get the center of the map
    center = lsoa_df.iloc[50].geometry.centroid
    # Create the map
    m = folium.Map(location=[center.y, center.x], zoom_start=11, tiles='cartodbpositron')
    # Add the polygons with hue based on the 'numerical_column'
    folium.GeoJson(lsoa_df, style_function=lambda feature: {
            'fillColor': colormap(feature['properties']['y']),
            'color': 'black',
            'weight': 0.5,
            'fillOpacity': 0.7}).add_to(m)
    # Add the colormap to the map
    colormap.add_to(m)
    # Save the map and overwrite the old one
    m.save('lsoa_plot.html')
    # Open the map
    #webbrowser.open('lsoa_plot.html')


def main():
    # load the data
    lsoa_london = load_data()
    lsoa_barnet_area = get_barnet_area(lsoa_london)
    plot_folium(lsoa_barnet_area)


if __name__ == "__main__":
    main()
