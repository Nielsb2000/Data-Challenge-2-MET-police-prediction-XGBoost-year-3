################
#
# file that takes processed data and turns into temporal graph representation for GNN
#
################


import numpy as np
import pandas as pd
import networkx as nx
import torch
import torch.nn.functional as F
#from torch_geometric.nn import GCNConv, GConvGRU
from torch.optim import Adam
import matplotlib.pyplot as plt
from torch_geometric_temporal.signal import temporal_signal_split
from torch_geometric_temporal.signal.static_graph_temporal_signal import StaticGraphTemporalSignal
from sklearn.preprocessing import StandardScaler


crimes = pd.read_csv('data/df_crimes.csv')
adj_matrix = pd.read_csv('data/df_distances.csv')


# Display the plot

crimes = crimes.sort_values(by=['wards', 'Month'])

timesteps = crimes['Month'].nunique()
nodes = crimes['wards'].nunique()
feature_dim = len(crimes.columns) - 3 # -2: remove month and wards and trash col
print(f'{"-"*30}\nGraph data info:\nnumber of timesteps: {timesteps}\nnumber of nodes: {nodes}\nfeature dimensions: {feature_dim}\n{"-"*30}')



################ last processing adj matrix

# Convert the adjacency matrix to an edge list
adj_matrix.set_index('name', inplace=True)
np.fill_diagonal(adj_matrix.values, 1)


# Create a graph from the edge_list
G = nx.from_pandas_adjacency(adj_matrix, create_using=nx.Graph())

# Generate edge_index and edge_weight from NetworkX graph
edge_index = list(G.edges())
edge_weight = nx.get_edge_attributes(G, "weight").values()

# Create a mapping of neighborhood names to unique integers
node_mapping = {node: i for i, node in enumerate(G.nodes())}
print(node_mapping['Barnet Vale'])

# Remap the nodes in the edge_index to these integers
edge_index = [(node_mapping[u], node_mapping[v]) for u, v in G.edges()]

# Then convert to tensor
edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

# Ensure that the same node_mapping is used when creating your node features and targets


# Convert edge_index and edge_weight to tensors
edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()  # Transpose to get 2xN shape
edge_weight = torch.tensor(list(edge_weight), dtype=torch.float)

plot = False
if plot:
    H = G.subgraph(list(G.nodes())[:100])
    nx.draw(H, with_labels=True)
    plt.show()

############ last processing crimes ##############

# turn month in an int index
crimes['Month'] = pd.to_datetime(crimes['Month'])
month_mapping = {month: index for index, month in enumerate(crimes['Month'].unique(), start=1)}
crimes['Month'] = crimes['Month'].map(month_mapping)

# turn wards in int mapping
crimes['wards'] = crimes['wards'].map(node_mapping)
print(crimes.info())
print(crimes[['wards', 'Month', 'Burglary', 'Bicycle theft', 'Robbery']].head())

########### turn to tensor #########
print('Number of NaN values in crimes:', crimes.isna().sum().sum())

# Selecting feature columns
feature_cols = [col for col in crimes.columns if col not in ['wards', 'Month', 'Mapped wards', 'Mapped month', 'Burglary']]


# Initialize a new StandardScaler instance
scaler = StandardScaler()

# Pivoting the DataFrame to get a multi-index DataFrame where the first level of the index is 'wards',
# the second level is 'Month', and the columns are the feature columns
features_df = crimes.pivot_table(values=feature_cols, index=['wards', 'Month'])

# Initialize a new StandardScaler instance
scaler = StandardScaler()

# Fit the scaler to your features and transform them
scaled_features = scaler.fit_transform(features_df.values)

# Convert the scaled features back into a DataFrame
scaled_features_df = pd.DataFrame(scaled_features, columns=features_df.columns, index=features_df.index)

# Use the scaled features to create the tensors for the StaticGraphTemporalSignal
# Creating a list of tensors where each tensor corresponds to a month and contains the features for all wards in that month
features = [torch.tensor(scaled_features_df.loc[(slice(None), month), :].values, dtype=torch.float) for month in crimes['Month'].unique()]

# Pivoting the DataFrame to get a DataFrame where the index is 'wards', the columns are 'Month', and the values are 'Burglary'
targets_df = crimes.pivot_table(values='Burglary', index='wards', columns='Month')

# Creating a list of tensors where each tensor corresponds to a month and contains the 'Burglary' counts for all wards in that month
targets = [torch.tensor(targets_df.loc[:, month].values, dtype=torch.float) for month in crimes['Month'].unique()]

dataset = StaticGraphTemporalSignal(edge_index, edge_weight, features, targets)
train_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=0.8)
print(dataset)


############## MODEL STUFF #############

from torch_geometric_temporal.nn.recurrent import DCRNN, GConvGRU

class RecurrentGCN(torch.nn.Module):
    def __init__(self, node_features):
        super(RecurrentGCN, self).__init__()
        self.recurrent = DCRNN(node_features, 32, 1)
        self.linear = torch.nn.Linear(32, 1)

    def forward(self, x, edge_index, edge_weight):
        h = self.recurrent(x, edge_index, edge_weight)
        h = F.relu(h)
        h = self.linear(h)
        return h

from tqdm import tqdm

model = RecurrentGCN(node_features = feature_dim)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

model.train()

losses = []
for epoch in tqdm(range(200)):
    cost = 0
    for time, snapshot in enumerate(train_dataset):
        #print(snapshot.edge_index.shape, snapshot.edge_index.dtype)
        #print(snapshot.edge_attr.shape, snapshot.edge_attr.dtype)
        #print(snapshot.x.shape)


        y_hat = model(snapshot.x, snapshot.edge_index.t(), snapshot.edge_attr)
        cost = cost + torch.mean((y_hat-snapshot.y)**2)
    cost = cost / (time+1)
    cost.backward()
    optimizer.step()
    optimizer.zero_grad()
    losses.append(cost.item())  # save the loss at each epoch


# plot the loss over epochs
plt.figure(figsize=(10,5))
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss during Training')
plt.show()



model.eval()
cost = 0
for time, snapshot in enumerate(test_dataset):
    y_hat = model(snapshot.x, snapshot.edge_index.t(), snapshot.edge_attr)
    cost = cost + torch.mean((y_hat-snapshot.y)**2)
cost = cost / (time+1)
cost = cost.item()
print("MSE: {:.4f}\nacross all the spatial units and time periods".format(cost))


mse_per_category = []
for i in range(y_hat.shape[0]):  # Assuming you have 500 categories
    cost = 0
    count = 0
    for time, snapshot in enumerate(test_dataset):
        y_hat = model(snapshot.x, snapshot.edge_index.t(), snapshot.edge_attr)
        cost += torch.mean((y_hat[i]-snapshot.y[i])**2)
        count += 1
    mse = cost / count
    mse_per_category.append(mse.item())

# Plot MSE per category with y-axis limited to 0.25
plt.figure(figsize=(10,5))
plt.bar(range(y_hat.shape[0]), np.sort(mse_per_category))
plt.ylim(0, 25)
plt.xlabel('Category')
plt.ylabel('MSE')
plt.title('MSE per Category')
plt.show()

# Calculate MSE for each timestep for a given category
category = 279  # Change to your chosen category
mse_per_timestep = []
for time, snapshot in enumerate(test_dataset):
    y_hat = model(snapshot.x, snapshot.edge_index.t(), snapshot.edge_attr)
    mse = torch.mean((y_hat[category]-snapshot.y[category])**2)
    mse_per_timestep.append(mse.item())

# Plot MSE per timestep for the chosen category
plt.figure(figsize=(10,5))
plt.plot(range(len(mse_per_timestep)), mse_per_timestep)
plt.xlabel('Timestep')
plt.ylabel('MSE')
plt.title(f'MSE over Time for Category {category}')
plt.show()
