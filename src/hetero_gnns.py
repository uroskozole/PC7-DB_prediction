from torch_geometric.data import HeteroData
from torch_geometric.nn import to_hetero, Sequential, HeteroDictLinear
from torch_geometric.nn.models import GAT, EdgeCNN, GCN, GraphSAGE, GIN
from table_to_heterodata import get_hetero_rossmann


def build_hetero_gnn(model_type, data: HeteroData, types: list, hidden_channels: int = 64, 
                     num_layers: int = 2, out_channels: int = 2, aggr: str = 'sum', model_kwargs: dict = {}):
    if model_type == 'GAT':
        model = GAT(in_channels=hidden_channels, hidden_channels=hidden_channels, num_layers=num_layers, out_channels=out_channels, add_self_loops=False, **model_kwargs)
    elif model_type == 'EdgeCNN':
        model = EdgeCNN(in_channels=hidden_channels, hidden_channels=hidden_channels, out_channels=out_channels, num_layers=num_layers, **model_kwargs)
    elif model_type == 'GCN':
        model = GCN(in_channels=hidden_channels, hidden_channels=hidden_channels, out_channels=out_channels, num_layers=num_layers, add_self_loops=False, **model_kwargs)
    elif model_type == 'GraphSAGE':
        model = GraphSAGE(in_channels=hidden_channels, hidden_channels=hidden_channels, out_channels=out_channels, num_layers=num_layers, **model_kwargs)
    elif model_type == 'GIN':
        model = GIN(in_channels=hidden_channels, hidden_channels=hidden_channels, out_channels=out_channels, num_layers=num_layers, **model_kwargs)
    # elif model_type == 'SignedGCN':
    #     model = SignedGCN(in_channels=hidden_channels, hidden_channels=hidden_channels, num_layers=num_layers, **model_kwargs)
    # elif model_type == 'PNA':
    #     model = PNA(in_channels=hidden_channels, hidden_channels=hidden_channels, out_channels=out_channels, num_layers=num_layers, **model_kwargs)
    else:
        raise ValueError(f"Model type {model_type} not supported.")
    gnn_model = to_hetero(model, data.metadata(), aggr=aggr)
    model = Sequential('x_dict, edge_index', [
        (HeteroDictLinear(in_channels=-1, out_channels=hidden_channels, types=types), 'x_dict -> x1'),
        (gnn_model, 'x1, edge_index -> y'),
    ])
    return model


if __name__ == '__main__':
    data = get_hetero_rossmann()
    model = build_hetero_gnn('GAT', data, types=list(data.x_dict.keys()), hidden_channels=64, num_layers=2, out_channels=1)
    output = model(data.x_dict, data.edge_index_dict)
    
    assert output['target'].shape[0] == data['target'].y.shape[0]
