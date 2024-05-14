from torch_geometric.data import HeteroData
import torch_geometric.transforms as T
import torch
import numpy as np
import pandas as pd
from pathlib import Path

from utils.metadata import Metadata
from utils.data import load_tables, remove_sdv_columns
from utils.utils import CustomHyperTransformer


DATA_DIR = "./data"

def csv_to_hetero_splits(database_name, target_table, target_column):
    data_train, ht_dict = csv_to_hetero(database_name, target_table, target_column, split='train')
    data_val = csv_to_hetero(database_name, target_table, target_column, split='val', ht_dict=ht_dict)
    data_test = csv_to_hetero(database_name, target_table, target_column, split='test', ht_dict=ht_dict)

    return data_train, data_val, data_test

def csv_to_hetero(database_name, target_table, target_column, split=None, ht_dict=None):

    data_path = Path(f'{DATA_DIR}/{database_name}')

    if split is not None:
        data_path = data_path / 'split' / split
    
    metadata = Metadata().load_from_json(f'{DATA_DIR}/{database_name}/metadata.json')

    tables = load_tables(data_path, metadata)
    tables, metadata = remove_sdv_columns(tables, metadata)
    y = tables[target_table].pop(target_column)

    data = HeteroData()

    ht_dict = {} if ht_dict is None else ht_dict

    ht_ = None

    # tranform to numerical        
    for key in metadata.get_tables():
        id_cols = metadata.get_column_names(key, sdtype='id')
        temp_table = tables[key].drop(columns=id_cols)

        if key not in ht_dict or split == "train":
            ht_ = CustomHyperTransformer()
            ht_.fit(temp_table)
            ht_dict[key] = ht_
        else:
            ht_ = ht_dict[key]

        temp_table = ht_.transform(temp_table)
        tables[key] = pd.concat([tables[key][id_cols], temp_table], axis=1)

    id_map = {}

    for key in metadata.get_tables():
        id_cols = metadata.get_column_names(key, sdtype='id')
        for col in id_cols:
            if key not in id_map:
                id_map[key] = {}
            
            if col not in id_map[key]:
                id_map[key][col] = {}
                for i, id_ in enumerate(tables[key][col].unique()):
                    id_map[key][col][id_] = i

            tables[key][col] = tables[key][col].map(id_map[key][col])
            

    # set connections
    for relationship in metadata.relationships:
        parent_table = relationship['parent_table_name']
        child_table = relationship['child_table_name']
        parent_column = relationship['parent_primary_key']
        child_column = relationship['child_foreign_key']

        data[parent_table, 'to', child_table].edge_index = torch.tensor(tables[child_table][[parent_column, child_column]].values.T.astype('int64'))
        data[child_table, 'from', parent_table].edge_index = torch.tensor(tables[child_table][[child_column, parent_column]].values.T.astype('int64'))

    # set connection to target
    target_primary_key = metadata.tables[target_table].primary_key
    data[target_table, 'to', 'target'].edge_index = torch.tensor(tables[target_table][[target_primary_key, target_primary_key]].values.T.astype('int64'))
    data['target', 'from', target_table].edge_index = torch.tensor(tables[target_table][[target_primary_key, target_primary_key]].values.T.astype('int64'))

    # drop ids
    for key in metadata.get_tables():
        id_cols = metadata.get_column_names(key, sdtype='id')
        tables[key] = tables[key].drop(columns=id_cols)

        table_values = tables[key].values

        # if table_values empty, set torch.zeros
        if table_values.size == 0:
            data[key].x = torch.zeros((tables[key].shape[0], 1), dtype=torch.float32)
        else:
            data[key].x = torch.tensor(table_values, dtype=torch.float32)

    
    data['target'].x = torch.zeros((tables[target_table].shape[0], 1), dtype=torch.float32)
    data['target'].y = torch.tensor(y.values.reshape(-1, 1), dtype=torch.float32)

    for key in metadata.get_tables():
        std = data[key].x.std(dim=0)
        std[std == 0] = 1
        data[key].x = ((data[key].x - data[key].x.mean(dim=0)) / std)

    transform = T.Compose([
        T.AddSelfLoops(),
    ])

    if split == "train":
        return transform(data), ht_dict

    return transform(data)


if __name__ == '__main__':
    data = csv_to_hetero("rossmann_subsampled", "historical", "Customers")
    # data = csv_to_hetero("Biodegradability_v1", "molecule", "activity")
    print(data)