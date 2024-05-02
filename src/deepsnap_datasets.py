import os
import contextlib

import pickle

import torch
import numpy as np
import pandas as pd
import networkx as nx
from deepsnap.hetero_graph import HeteroGraph

from relsyndgb.metadata import Metadata
from relsyndgb.data import load_tables, remove_sdv_columns

from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from table_to_graph import database_to_graph, DATA_DIR

# TODO: this was meant for masked table modeling
# transform if for normal node classification / regression

def get_num_message_edges(hete):
  message_type_edges = [(message_type, len(edges)) for message_type, edges in hete.edge_type.items()]
  return message_type_edges


def deepsnap_dataset_from_graph(G, metadata, label_encoders_path, feature_dim = 32):
    # create edges in both directions
    G = G.to_undirected()
    G = nx.to_directed(G)

    node_features = {}
    node_types = {}

    data_types = {table: metadata.get_dtypes(table) for table in metadata.get_tables()}

    label_encoders = pickle.load(open(label_encoders_path, 'rb'))

    for node in G.nodes:
        node_data = G.nodes[node]
        node_types[node] = node_data['node_type']
        table = node_data['node_type']
        node_data_dict = node_data.copy()
        features = []
        for column, value in node_data_dict.items():
            del node_data[column]
            if column == 'node_type' or column not in data_types[table]:
                continue
            if data_types[table][column] == 'object':
                features.append(label_encoders[table][column].transform([value])[0])
            elif data_types[table][column] == 'datetime64[ns]' or data_types[table][column] == 'datetime64':
                features.append(pd.to_datetime(value).value)
            else:
                features.append(value)
        features = torch.tensor(features).float()

        node_features[node] = features

    # assign the node attributes
    nx.set_node_attributes(G, node_features, 'node_feature')
    nx.set_node_attributes(G, node_types, 'node_type')

    # deepsnap has a pesky print statement that we need to suppress
    with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
        hete = HeteroGraph(G)

    # sort edge index to ensure message type order is consistent when sampling
    hete['edge_index'] = {k: v for k, v in sorted(hete['edge_index'].items(), key=lambda item: item[0])}
    return hete



def create_deepsnap_dataset(dataset_name, target_table, masked_tables, feature_dim = 32, k = 5, load_stored = True):
    if load_stored and os.path.exists(f'data/hetero_graph/{dataset_name}_{target_table}_{k}.pkl'):
        return pickle.load(open(f'data/hetero_graph/{dataset_name}_{target_table}_{k}.pkl', 'rb'))
    
    metadata = Metadata().load_from_json(f'{DATA_DIR}/downloads/{dataset_name}/metadata.json')
    tables = load_tables(f'{DATA_DIR}/downloads/{dataset_name}/', metadata)
    tables, metadata = remove_sdv_columns(tables, metadata)

    G, _ = database_to_graph(dataset_name)

    # create edges in both directions
    G = G.to_undirected()
    G = nx.to_directed(G)

    node_labels = {}
    node_features = {}
    node_types = {}

    label_encoders = {table: {} for table in metadata.get_tables()}

    # use kmeans labels as node labels
    kmeans = {}

    data_types = {}
    means = {}
    stds = {}
    for table in metadata.get_tables():
        data_types[table] = metadata.get_dtypes(table)
        for column, dtype in data_types[table].items():
            if dtype == 'object':
                label_encoders[table][column] = LabelEncoder()
                label_encoders[table][column].fit(tables[table][column])
                tables[table][column] = label_encoders[table][column].transform(tables[table][column])
            elif dtype == 'datetime64[ns]' or dtype == 'datetime64':
                tables[table][column] = pd.to_numeric(pd.to_datetime(tables[table][column]))
                
        tables[table].pop(metadata.get_primary_key(table))
        for parent in metadata.get_parents(table):
            for fk in metadata.get_foreign_keys(parent, table):
                tables[table].pop(fk)
        means[table] = tables[table].mean().values
        stds[table] = tables[table].std().values
        tables[table] = (tables[table] - tables[table].mean()) / tables[table].std()
        kmeans[table] = KMeans(n_clusters=k, n_init='auto').fit(tables[table].to_numpy())

    for node in G.nodes:
        node_data = G.nodes[node]
        node_types[node] = node_data['node_type']
        table = node_data['node_type']
        node_data_dict = node_data.copy()
        if table in masked_tables:
            features = torch.ones(feature_dim)
            current_node_features = []
            for column, value in node_data_dict.items():
                del node_data[column]
                # skip type and primary and foreign keys which do not have a label encoder
                if column == 'node_type' or column not in data_types[table]:
                    continue
                if data_types[table][column] == 'object':
                    current_node_features.append(label_encoders[table][column].transform([value])[0])
                elif data_types[table][column] == 'datetime64[ns]' or data_types[table][column] == 'datetime64':
                    current_node_features.append(pd.to_datetime(value).value)
                else:
                    current_node_features.append(value)
            current_node_features = np.array(current_node_features)
        else:
            features = []
            for column, value in node_data_dict.items():
                del node_data[column]
                if column == 'node_type' or column not in data_types[table]:
                    continue
                if data_types[table][column] == 'object':
                    features.append(label_encoders[table][column].transform([value])[0])
                elif data_types[table][column] == 'datetime64[ns]' or data_types[table][column] == 'datetime64':
                    features.append(pd.to_datetime(value).value)
                else:
                    features.append(value)
            current_node_features = np.array(features).astype(float)
            current_node_features = (current_node_features - means[table]) / stds[table]
            features = torch.tensor(features).float()
            
        pred = torch.tensor(kmeans[table].predict(current_node_features.reshape(1, -1))[0]).long()
        # convert to one-hot
        node_labels[node] = torch.nn.functional.one_hot(pred, num_classes=k)
        node_features[node] = features

    # assign the node attributes
    nx.set_node_attributes(G, node_labels, 'node_label')
    nx.set_node_attributes(G, node_features, 'node_feature')
    nx.set_node_attributes(G, node_types, 'node_type')
    
    # deepsnap has a pesky print statement that we need to suppress
    with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
        hete = HeteroGraph(G)
        
    # sort edge index to ensure message type order is consistent when sampling
    hete['edge_index'] = {k: v for k, v in sorted(hete['edge_index'].items(), key=lambda item: item[0])}

    # store the hetero dataset
    os.makedirs('data/hetero_graph', exist_ok=True)
    pickle.dump(hete, open(f'data/hetero_graph/{dataset_name}_{target_table}_{k}.pkl', 'wb'))
    # store the label encoders
    pickle.dump(label_encoders, open(f'data/hetero_graph/{dataset_name}_{target_table}_{k}_label_encoders.pkl', 'wb'))
    return hete

def main():
    dataset_name = 'mutagenesis'
    target_table = 'bond'
    masked_tables = ['bond']
    create_deepsnap_dataset(dataset_name, target_table, masked_tables, load_stored = False)

if __name__ == "__main__":
    main()
    