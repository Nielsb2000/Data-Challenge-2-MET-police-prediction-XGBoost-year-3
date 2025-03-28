import numpy as np
from scipy.sparse import csr_matrix
from torch_geometric.utils import from_scipy_sparse_matrix
import torch
import pandas as pd
import os
import json
from torch_geometric.data import InMemoryDataset, Data
from shutil import copyfile
from sklearn.metrics.pairwise import euclidean_distances


def distance_to_weight(W, sigma2=0.1, epsilon=0.5, gat_version=False):
    """"
    Given distances between all nodes, convert into a weight matrix
    :param W distances
    :param sigma2 User configurable parameter to adjust sparsity of matrix
    :param epsilon User configurable parameter to adjust sparsity of matrix
    :param gat_version If true, use 0/1 weights with self loops. Otherwise, use float
    """
    n = W.shape[0]
    W = W / 10000.
    W2, W_mask = W * W, np.ones([n, n]) - np.identity(n)
    # refer to Eq.10
    W = np.exp(-W2 / sigma2) * (np.exp(-W2 / sigma2) >= epsilon) * W_mask
    # If using the gat version of this, round to 0/1 and include self loops
    if gat_version:
        W[W>0] = 1
        W += np.identity(n)
    return W



### DF CRIMES STUFF ###########
# Assuming `df` is your DataFrame
df = pd.read_csv('data/df_crimes_LSOA_code.csv')
df.set_index(['LSOA code', 'Month'], inplace=True)
df.sort_index(inplace=True)
df.sort_values(by=['LSOA code', 'Month'], inplace=True)

# Get unique 'LSOA code' and 'Month' from original DataFrame
unique_LSOA_codes = df.index.get_level_values('LSOA code').unique()
unique_Months = df.index.get_level_values('Month').unique()
# Ensure 'LSOA code' and 'Month' are of the same data type in new MultiIndex and original DataFrame's index
new_index = pd.MultiIndex.from_product([unique_LSOA_codes.astype(df.index.get_level_values('LSOA code').dtype),
                                        unique_Months.astype(df.index.get_level_values('Month').dtype)],
                                       names=['LSOA code', 'Month'])
# Reindex DataFrame with new index and fill missing values with zeros
df = df.reindex(new_index, fill_value=0)
df.drop(columns=['Unnamed: 0'], inplace=True)

print("Number of unique LSOA codes: ", df.index.get_level_values('LSOA code').nunique())
print("Number of unique Months: ", df.index.get_level_values('Month').nunique())
print("Number of features: ", df.shape[1])
print("Are there missing values? ", df.isnull().values.sum().sum())
print("Number of rows in DataFrame: ", df.shape[0])
print('nrows*nmonths=', df.index.get_level_values('Month').nunique()*df.index.get_level_values('LSOA code').nunique())
num_filled_rows = (df != 0).any(axis=1).sum()
print(f'Number of filled-in rows: {num_filled_rows}')
print(f'Total number of rows: {len(df)}')
print(f'{round(num_filled_rows/len(df),4)} % of the data filled in with 0s')

# Number of unique 'LSOA code'
num_nodes = df.index.get_level_values('LSOA code').nunique()
# Number of unique 'Month'
num_months = df.index.get_level_values('Month').nunique()
# Number of features in the original DataFrame
num_features = df.shape[1]
# Convert DataFrame to tensor
df_tensor = torch.tensor(df.values, dtype=torch.float)
# Reshape tensor to have dimensions [num_nodes, num_features, num_months]
df_tensor = df_tensor.view(num_nodes, num_months, num_features)
# Extract 'Burglary' time-series as the target attribute
target = df_tensor[:, :, df.columns.get_loc('Burglary')]


### DF DISTANCES STUFF #########
df_centroids = pd.read_csv('data/LSOA_Dec_2011_PWC_in_England_and_Wales_2022_1923591000694358693.csv')
df_centroids = df_centroids[df_centroids['LSOA11CD'].isin(df.index.get_level_values('LSOA code'))]
# create a new dataframe containing only x and y columns
df_coordinates = df_centroids[['x', 'y']]
# compute the distance matrix
dist_matrix = euclidean_distances(df_coordinates, df_coordinates)
# create DataFrame from distance matrix
df_dist = pd.DataFrame(dist_matrix, index=df_centroids['LSOA11CD'], columns=df_centroids['LSOA11CD'])
# Create edge_index from adjacency matrix
df_distances_tensor = torch.tensor(df_dist.values, dtype=torch.float)


# Assuming df_distances is your distance matrix
distance_matrix = df_dist.values
# Normalize the distance matrix to get the weight matrix
weight_matrix = distance_to_weight(distance_matrix, sigma2=0.1, epsilon=0.5, gat_version=False)
# Convert the weight matrix to a sparse matrix
sparse_weight_matrix = csr_matrix(weight_matrix)
# Convert the sparse matrix to an edge index and edge weight
edge_index, edge_weight = from_scipy_sparse_matrix(sparse_weight_matrix)



#### OTHER ####

# Assuming df_dist is your adjacency matrix DataFrame
index_to_lsoa = {i: lsoa for i, lsoa in enumerate(df_dist.index)}
lsoa_to_index = {lsoa: i for i, lsoa in enumerate(df_dist.index)}
mappings = {'index_to_lsoa': index_to_lsoa, 'lsoa_to_index': lsoa_to_index}
with open('data/mappings.json', 'w') as f:
    json.dump(mappings, f)

# Create Data object
data = Data(x=df_tensor, edge_index=edge_index, edge_attr=edge_weight, y=target)
print('saving data to file...')
torch.save(data, 'data/graph_data.pt')
print(data)
