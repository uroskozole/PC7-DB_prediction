from torch_geometric.data import HeteroData
import torch_geometric.transforms as T
import torch
import numpy as np

from utils.metadata import Metadata
from utils.data import load_tables, remove_sdv_columns
from utils.utils import CustomHyperTransformer

DATA_DIR = "./data"


def get_hetero_rossmann():
    database_name = "rossmann_subsampled"
    metadata = Metadata().load_from_json(f'{DATA_DIR}/{database_name}/metadata.json')

    tables = load_tables(f'{DATA_DIR}/{database_name}/', metadata)
    tables, metadata = remove_sdv_columns(tables, metadata)

    data = HeteroData()

    # reindex historical to start from 1 and not from 300k
    tables['historical']['Id'] = np.arange(1, tables['historical'].shape[0]+1)
    y = tables['historical'].pop('Customers')

    # tranform to numerical
    ht = CustomHyperTransformer()
    tables['store'] = ht.fit_transform(tables['store'])
    tables['historical'] = ht.fit_transform(tables['historical'])

    # remove -1 from indexes to make them start from 0
    data['store', 'to', 'historical'].edge_index = torch.tensor(tables['historical'][['Store', 'Id']].values.T)-1
    data['historical', 'from', 'store'].edge_index = torch.tensor(tables['historical'][['Id', 'Store']].values.T)-1
    data['historical', 'to', 'target'].edge_index = torch.tensor(tables['historical'][['Id', 'Id']].values.T.astype('int64'))-1
    data['target', 'from', 'historical'].edge_index = torch.tensor(tables['historical'][['Id', 'Id']].values.T.astype('int64'))-1

    # drop ids
    tables['store'] = tables['store'].drop(columns=['Store'])
    tables['historical'] = tables['historical'].drop(columns=['Id', 'Store'])

    # build hetero data
    data['store'].x = torch.tensor(tables['store'].values).float()
    data['historical'].x = torch.tensor(tables['historical'].values).float()
    data['target'].x = torch.zeros((tables['historical'].shape[0], 1)).float()
    data['target'].y = torch.tensor(y.values.reshape(-1, 1)).float()

    transform = T.Compose([
        T.AddSelfLoops(),
        T.NormalizeFeatures(),
    ])

    return transform(data)

if __name__ == '__main__':
    data = get_hetero_rossmann()
    print(data)