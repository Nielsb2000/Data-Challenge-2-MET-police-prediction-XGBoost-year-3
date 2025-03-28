import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,mean_absolute_error
import xgboost as xgb
import matplotlib.pyplot as plt
import geopandas as gpd
from sklearn.metrics import r2_score
from statsmodels.tsa.stattools import grangercausalitytests


df = pd.read_csv("data/df_crimes_LSOA_code.csv")
df = df.drop('House price', axis=1)
df = df.drop('Density', axis=1)

# Extract season info from 'Month' column
df['Month'] = pd.to_datetime(df['Month'], format='%Y-%m')
df['Season'] = pd.cut(df['Month'].dt.month, [0, 3, 6, 9, 12], labels=['Winter', 'Spring', 'Summer', 'Autumn'])
# Create a dictionary comprehension to specify the aggregation functions
sum_columns = df.columns[2:18]
avg_columns = df.columns[19:-2]
aggregation_dict = {column: 'sum' for column in sum_columns}
aggregation_dict.update({column: 'mean' for column in avg_columns})

# Group by season 
df = df.groupby(['LSOA code', 'Season', 'Year']).agg(aggregation_dict).reset_index()

#get boundary for LSOA
path = r"data/LSOA_boundary/LSOA_2011_EW_BFC.shp"
lsoa_london = gpd.read_file(path)
lsoa_london = lsoa_london[['LSOA11CD','geometry']]
lsoa_london = lsoa_london.rename(columns={'LSOA11CD':'LSOA code'})
geometries = lsoa_london['geometry']
df = df.merge(lsoa_london, on=['LSOA code'])

# get boundary for ward
path = r"data/df_wards.shp"
df_ward = gpd.read_file(path)
wards_barnet = ["High Barnet", "Underhill", "Barnet Vale", "East Barnet", "Friern Barnet","Woodhouse", 
                "Whetstone", "Brunswick Park", "Totteridge and Woodside", "Mill Hill", "Cricklewood",
                 "Edgwarebury", "Burnt Oak", "Colindale South", "West Hendon", "Colindale North","Hendon",
                 "West Finchley", "East Finchley", "Garden Suburb", "Finchley Church End", "Golders Green", "Childs Hill"]
df_ward = df_ward[df_ward['name'].isin(wards_barnet)]

## filter corr 
exclude_columns = ["LSOA code", "Season", "geometry", "Year"]
comparison_columns = [column for column in df.columns if column not in exclude_columns]
df[comparison_columns] = df[comparison_columns].fillna(df[comparison_columns].mean())
correlations = df[comparison_columns].corr()["Burglary"]
positive_corr_columns = correlations[correlations > 0].index.tolist()
# filter granger cuasality
for col in positive_corr_columns:
    result = grangercausalitytests(df[[col, 'Burglary']], maxlag=2, verbose=False)
    if (result[1][0]['ssr_ftest'][1] <= 0.05) or (result[2][0]['ssr_ftest'][1] <= 0.05):
        exclude_columns.append(col)
exclude_columns.append('Burglary')
df = df[exclude_columns]

le = preprocessing.LabelEncoder()
df['LSOA code'] = le.fit_transform(df['LSOA code'])
df['Season'] = le.fit_transform(df['Season'])


X = df.drop(['Burglary', 'geometry'], axis=1)
y = df['Burglary']


# Apply scaling
scaler = preprocessing.MinMaxScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Create DMatrices
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)


# Set parameters for XGBoost
param = {
    'max_depth': 6,
    'eta': 0.5,
    'silent': 1,
    'objective': 'reg:squarederror',
    'alpha': 0.1,  
    'lambda': 0.1,
}
num_round = 300

# Train the model
bst = xgb.train(param, dtrain, num_round)   
df_predict = df.drop('Burglary', axis=1)

# Initialize an empty DataFrame to store the predicted values for each LSOA code
predictions_df = pd.DataFrame(columns=['LSOA code', 'prediction', 'geometry'])

# predict each LSOA code
lsoa_codes = df_predict['LSOA code'].unique()
for lsoa in lsoa_codes:
    specific_data = df_predict[df_predict['LSOA code'] == lsoa]
    specific_data_scaled = scaler.transform(specific_data.drop('geometry', axis=1))
    d_specific = xgb.DMatrix(specific_data_scaled)
    preds = bst.predict(d_specific)

    # Store the predicted value and geometry in the DataFrame
    row = {'LSOA code': lsoa, 'prediction': preds[0], 'geometry': specific_data['geometry'].iloc[0]}
    predictions_df = predictions_df.append(row, ignore_index=True)

# print("Predictions for next season:\n", predictions_df)

# Create a GeoDataFrame from predictions_df
predictions_gdf = gpd.GeoDataFrame(predictions_df, geometry='geometry')

#prediction map
fig, ax = plt.subplots(figsize=(10, 10))
# Plot the polygons
lsoa_london.plot(ax=ax, color='lightgray', edgecolor='blue')
predictions_gdf.plot(ax=ax, column='prediction', cmap='YlOrRd', linewidth=0.8, edgecolor='black', legend=True,
                     vmin=0, vmax=3)  
df_ward.plot(ax=ax, color='none', edgecolor='black', linewidth=3)

# Set aspect ratio and axis limits
ax.set_aspect('equal')
ax.set_xlim(lsoa_london.total_bounds[0], lsoa_london.total_bounds[2])
ax.set_ylim(lsoa_london.total_bounds[1], lsoa_london.total_bounds[3])

# Remove the axis labels
ax.set_axis_off()

ax.set_title('Burglary predictions for next season')
plt.show()

# Plot residual
# Split out the last season
last_season = df[((df['Season'] == 3) & (df['Year'] == 2022))]
last_season = last_season.drop(['Burglary', 'geometry'], axis=1)
last_season_scaled = scaler.transform(last_season)
d_last_season = xgb.DMatrix(last_season_scaled)

last_season_preds = bst.predict(d_last_season)

# Get the real last season values
last_season_real = df[(df['Season'] == 3) & (df['Year'] == 2022)]['Burglary'].values
mae = mean_absolute_error(last_season_real, last_season_preds)
mse = mean_squared_error(last_season_real, last_season_preds)
r2 = r2_score(last_season_real, last_season_preds)

# Calculate the residuals
residuals = last_season_real - last_season_preds
# Plot the residuals
plt.scatter(last_season_real, residuals)
plt.axhline(y=0, color='r', linestyle='--')  # Add a horizontal line at y=0 for reference
plt.xlabel('True Values')
plt.ylabel('Residuals')
plt.title(f'Residual plot for last season, MAE: {round(mae,2)}, MSE: {round(mse,2)}, R-squared: {round(r2, 2)}')
plt.show()
