#######
#
# Visualize model outputs
#
#####

import numpy as np
import pandas as pd
import geopandas as gpd
import folium
from folium.plugins import TimeSliderChoropleth
from branca.colormap import LinearColormap
from matplotlib.cm import magma


def load_data(path=r'data/lsoa_london.shp'):
    """
    from return df with true_MM/pred_MM/mse_MM columns, indexed by LSOA code
    """
    df_lsoa = gpd.read_file(path)
    y_true = pd.read_csv('data/y_true.csv', index_col=0).transpose()
    y_pred = pd.read_csv('data/y_pred.csv', index_col=0).transpose()

    df_lsoa = df_lsoa.set_index('lsoa21cd', drop=True)
    df_lsoa = df_lsoa.sort_index()
    y_true = y_true.sort_index()
    y_pred = y_pred.sort_index()

    df_mse = (y_true.subtract(y_pred))**2

    df_lsoa = df_lsoa.join(y_true, how='left', rsuffix='_true')
    df_lsoa = df_lsoa.join(y_pred, how='left', rsuffix='_pred')
    df_lsoa = df_lsoa.join(df_mse, how='left', rsuffix='_mse')
    df_lsoa.drop(columns=['label'], inplace=True)
    #df_lsoa = df_lsoa.to_crs(3857)
    return df_lsoa


def plot_folium(df_lsoa, what='mse'):
    """
    what: 'true', 'pred', 'mse' what to visualize in the chart
    """
    df_lsoa['geometry'] = df_lsoa.geometry.simplify(tolerance=0.0005)  # Increase tolerance if needed
    df_lsoa = df_lsoa.round(4)  # Reducing precision to 4 decimal places
    # Identify mse columns
    mse_cols = [col for col in df_lsoa.columns if f'_{what}' in col]#[:20]
    df_lsoa = df_lsoa[[*mse_cols, 'geometry']]
    # Create the linear colormap
    min_value = df_lsoa[mse_cols].min().min()
    max_value = df_lsoa[mse_cols].max().max()
    if min_value > max_value:
        min_value, max_value = max_value, min_value
    colormap = LinearColormap(['blue', 'red'], vmin=min_value, vmax=max_value).to_step(n=10)
    #colormap = LinearColormap(colors=[magma(i) for i in range(256)], vmin=min_value, vmax=max_value).to_step(n=10)

    # Get the center of the map
    center = df_lsoa.geometry.centroid.iloc[50]
    center = [center.y, center.x]
    # Generate a monthly time index starting from April 2010 in UNIX timestamp format
    start_date = pd.to_datetime('2010-04-01')
    dates = pd.date_range(start=start_date, periods=len(mse_cols), freq='M')
    dates_unix = [int(date.value // 10**6) for date in dates]
    # Generate style dictionary for TimeSliderChoropleth
    time_dict = dict(zip(mse_cols, dates_unix))
    # Generate style dictionary for TimeSliderChoropleth
    styledict2 = {
        str(idx): {time_dict[mse_col]: {'color': colormap(df_lsoa.loc[idx, mse_col]), 'opacity': 0.7} for mse_col in mse_cols}
       for idx, _ in df_lsoa.iterrows()}

    styledict = {
        str(idx): {mse_col: {'color': colormap(df_lsoa.loc[idx, mse_col]), 'opacity': 0.7} for mse_col in mse_cols}
        for idx, _ in df_lsoa.iterrows()}
    # Create the map
    m = folium.Map(location=center, zoom_start=11, tiles='cartodbpositron')
    # Create TimeSliderChoropleth layer and add to the map
    TimeSliderChoropleth(
        data=df_lsoa.to_json(), styledict=styledict).add_to(m)
    # Add the colormap to the map
    colormap.add_to(m)
    # Save the map and overwrite the old one
    m.save(f'lsoa_plot_{what}.html')


df = load_data()
plot_folium(df)
