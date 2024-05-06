from torch_geometric.data import HeteroData
import torch_geometric.transforms as T
import torch
import numpy as np
import pandas as pd

from utils.metadata import Metadata
from utils.data import load_tables, remove_sdv_columns
from utils.utils import CustomHyperTransformer

DATA_DIR = "./data"

def csv_to_hetero(database_name, target_table, target_column):
    metadata = Metadata().load_from_json(f'{DATA_DIR}/{database_name}/metadata.json')

    tables = load_tables(f'{DATA_DIR}/{database_name}/', metadata)
    tables, metadata = remove_sdv_columns(tables, metadata)
    y = tables[target_table].pop(target_column)

    data = HeteroData()

    # tranform to numerical
    ht = CustomHyperTransformer()
    for key in metadata.get_tables():
        id_cols = metadata.get_column_names(key, sdtype='id')
        temp_table = tables[key].drop(columns=id_cols)
        temp_table = ht.fit_transform(temp_table)
        tables[key] = pd.concat([tables[key][id_cols], temp_table], axis=1)

    # set connections
    for relationship in metadata.relationships:
        parent_table = relationship['parent_table_name']
        child_table = relationship['child_table_name']
        parent_column = relationship['parent_primary_key']
        child_column = relationship['child_foreign_key']

        tables[parent_table][parent_column] = np.arange(tables[parent_table].shape[0])
        tables[child_table][child_column] = np.arange(tables[child_table].shape[0])

        data[parent_table, 'to', child_table].edge_index = torch.tensor(tables[child_table][[child_column, parent_column]].values.T.astype('int64'))
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
            data[key].x = torch.zeros((tables[key].shape[0], 1)).float()
        else:
            data[key].x = torch.tensor(table_values).float()

    
    data['target'].x = torch.zeros((tables[target_table].shape[0], 1)).float()
    data['target'].y = torch.tensor(y.values.reshape(-1, 1)).float()

    for key in metadata.get_tables():
        std = data[key].x.std(dim=0)
        std[std == 0] = 1
        data[key].x = ((data[key].x - data[key].x.mean(dim=0)) / std)

    transform = T.Compose([
        T.AddSelfLoops(),
    ])

    return transform(data)


if __name__ == '__main__':
    data = csv_to_hetero("rossmann_subsampled", "historical", "Customers")
    # data = csv_to_hetero("Biodegradability_v1", "molecule", "activity")
    print(data)