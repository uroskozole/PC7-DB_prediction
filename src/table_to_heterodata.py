from pathlib import Path

from torch_geometric.data import HeteroData
import torch_geometric.transforms as T
import torch
import pandas as pd

from utils.metadata import Metadata
from utils.data import load_tables, remove_sdv_columns, make_column_names_unique
from utils.utils import CustomHyperTransformer


DATA_DIR = "./data"

def csv_to_hetero_splits(database_name, target_table, target_column):
    data_train, ht_dict, categories = csv_to_hetero(database_name, target_table, target_column, split='train')
    data_val = csv_to_hetero(database_name, target_table, target_column, split='val', ht_dict=ht_dict, categories=categories)
    data_test = csv_to_hetero(database_name, target_table, target_column, split='test', ht_dict=ht_dict, categories=categories)

    return data_train, data_val, data_test

def csv_to_hetero(database_name, target_table, target_column, split=None, ht_dict=None, categories=None):

    data_path = Path(f'{DATA_DIR}/{database_name}')

    if split is not None:
        data_path = data_path / 'split' / split
    
    metadata = Metadata().load_from_json(f'{DATA_DIR}/{database_name}/metadata.json')

    tables = load_tables(data_path, metadata)
    tables, metadata = remove_sdv_columns(tables, metadata)
    tables, metadata = make_column_names_unique(tables, metadata)

    y = tables[target_table].pop(f'{target_table}_{target_column}')

    data = HeteroData()

    ht_dict = {} if ht_dict is None else ht_dict

    ht_ = None

    # tranform to numerical  
    if categories is None:
        categories = {}
    for key in metadata.get_tables():
        id_cols = metadata.get_column_names(key, sdtype='id')
        temp_table = tables[key].drop(columns=id_cols)

        if key not in ht_dict or split == "train":
            categories[key] = dict()
            for column in metadata.get_column_names(key, sdtype='categorical'):
                categories[key][column] = temp_table[column].unique()
            ht_ = CustomHyperTransformer()
            numerical_dtypes = temp_table.dtypes[temp_table.dtypes == 'float64'].index
            temp_table[numerical_dtypes].fillna(0, inplace=True)
            ht_.fit(temp_table)
            ht_dict[key] = ht_
        else:
            ht_ = ht_dict[key]
            for column in metadata.get_column_names(key, sdtype='categorical'):
                temp_table[column] = pd.Categorical(temp_table[column], categories=categories[key][column])

        temp_table = ht_.transform(temp_table)
        tables[key] = pd.concat([tables[key][id_cols], temp_table], axis=1)

    id_map = {}

    for parent_table_name in metadata.get_tables():
        primary_key = metadata.get_primary_key(parent_table_name)

        if parent_table_name not in id_map:
            id_map[parent_table_name] = {}
        
        if primary_key not in id_map[parent_table_name]:
            id_map[parent_table_name][primary_key] = {}
            idx = 0
            for primary_key_val in tables[parent_table_name][primary_key].unique():
                id_map[parent_table_name][primary_key][primary_key_val]  = idx
                idx += 1 

        for relationship in metadata.relationships:
            if relationship['parent_table_name'] != parent_table_name:
                continue
            if relationship['child_table_name'] not in id_map:
                id_map[relationship['child_table_name']] = {}
            
            id_map[relationship['child_table_name']][relationship['child_foreign_key']] = id_map[parent_table_name][relationship['parent_primary_key']]

            # id_map[parent_table_name][relationship['child_table_name']] = tables[parent_table_name][relationship['parent_primary_key']]

    for table_name in id_map.keys():
        for column_name in id_map[table_name].keys():
            if column_name not in tables[table_name].columns:
                raise ValueError(f"Column {column_name} not found in table {table_name}")
            tables[table_name][column_name] = tables[table_name][column_name].map(id_map[table_name][column_name])


    # set edges based on relationships
    for relationship in metadata.relationships:
        parent_table = relationship['parent_table_name']
        child_table = relationship['child_table_name']
        foreign_key = relationship['child_foreign_key']

        child_primary_key = metadata.get_primary_key(child_table)

        data[parent_table, 'to', child_table].edge_index = torch.tensor(tables[child_table][[foreign_key, child_primary_key]].values.T.astype('int64'))
        data[child_table, 'from', parent_table].edge_index = torch.tensor(tables[child_table][[child_primary_key, foreign_key]].values.T.astype('int64'))

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
        # T.RemoveIsolatedNodes(),
    ])

    if split == "train":
        return transform(data), ht_dict, categories

    return transform(data)


if __name__ == '__main__':
    data = csv_to_hetero("rossmann_subsampled", "historical", "Customers")
    # data = csv_to_hetero("Biodegradability_v1", "molecule", "activity")
    print(data)
