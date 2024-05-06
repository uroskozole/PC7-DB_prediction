from torch_geometric.data import HeteroData
import torch
import numpy as np

from utils.metadata import Metadata
from utils.data import load_tables, remove_sdv_columns

DATA_DIR = "./data"


def get_hetero_rossmann():
    database_name = "rossmann_subsampled"
    metadata = Metadata().load_from_json(f'{DATA_DIR}/{database_name}/metadata.json')

    tables = load_tables(f'{DATA_DIR}/{database_name}/', metadata)
    tables, metadata = remove_sdv_columns(tables, metadata)

    data = HeteroData()

    # keep only numerical columns for tables store and historical

    tables['store'] = tables['store'][[col for col in tables['store'].columns if tables['store'][col].dtype in ['int64', 'float64']]]
    tables['historical'] = tables['historical'][[col for col in tables['historical'].columns if tables['historical'][col].dtype in ['int64', 'float64']]]


    # reindex historical to start from 1 and not from 300k
    tables['historical']['Id'] = np.arange(1, tables['historical'].shape[0]+1)
    y = tables['historical'].pop('Customers')

    # build hetero data
    data['store'].x = torch.tensor(tables['store'].values).float()
    data['historical'].x = torch.tensor(tables['historical'].values).float()
    data['historical'].y = torch.tensor(y.values).float()

    # remove -1 from indexes to make them start from 0
    data['store', 'to', 'historical'].edge_index = torch.tensor(np.array(tables['historical'][['Store', 'Id']]).T)-1
    data['historical', 'from', 'store'].edge_index = torch.tensor(np.array(tables['historical'][['Id', 'Store']]).T)-1
    return data