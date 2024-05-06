from torch_geometric.data import HeteroData
from torch_geometric.nn import to_hetero
from torch_geometric.nn.models import GAT, EdgeCNN, GCN, GraphSAGE, GIN
from table_to_heterodata import get_hetero_rossmann


def build_hetero_gnn(model_type, data: HeteroData, in_channels: int = 64, hidden_channels: int = 64, 
                     num_layers: int = 2, out_channels: int = 2, aggr: str = 'sum', model_kwargs: dict = {}):
    if model_type == 'GAT':
        model = GAT(in_channels=in_channels, hidden_channels=hidden_channels, num_layers=num_layers, out_channels=out_channels, add_self_loops=False, **model_kwargs)
    elif model_type == 'EdgeCNN':
        model = EdgeCNN(in_channels=in_channels, hidden_channels=hidden_channels, out_channels=out_channels, num_layers=num_layers, **model_kwargs)
    elif model_type == 'GCN':
        model = GCN(in_channels=in_channels, hidden_channels=hidden_channels, out_channels=out_channels, num_layers=num_layers, add_self_loops=False, **model_kwargs)
    elif model_type == 'GraphSAGE':
        model = GraphSAGE(in_channels=in_channels, hidden_channels=hidden_channels, out_channels=out_channels, num_layers=num_layers, **model_kwargs)
    elif model_type == 'GIN':
        model = GIN(in_channels=in_channels, hidden_channels=hidden_channels, out_channels=out_channels, num_layers=num_layers, **model_kwargs)
    # elif model_type == 'SignedGCN':
    #     model = SignedGCN(in_channels=in_channels, hidden_channels=hidden_channels, num_layers=num_layers, **model_kwargs)
    # elif model_type == 'PNA':
    #     model = PNA(in_channels=in_channels, hidden_channels=hidden_channels, out_channels=out_channels, num_layers=num_layers, **model_kwargs)
    else:
        raise ValueError(f"Model type {model_type} not supported.")
    return to_hetero(model, data.metadata(), aggr=aggr)


if __name__ == '__main__':
    import torch_geometric.transforms as T
    from torch_geometric.datasets import OGB_MAG
    # dataset = OGB_MAG(root='./data', preprocess='metapath2vec', transform=T.ToUndirected())
    # dataset = get_hetero_rossmann()
    data = get_hetero_rossmann()
    model = build_hetero_gnn('GraphSAGE', data, in_channels=7, hidden_channels=64, num_layers=2, out_channels=1)
    output = model(data.x_dict, data.edge_index_dict)

    print("Neki")