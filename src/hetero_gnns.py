import torch
from torch_geometric.data import HeteroData
from torch_geometric.nn import to_hetero, Sequential, HeteroDictLinear
from torch_geometric.nn.models import GAT, EdgeCNN, GraphSAGE, GIN, MLP
from table_to_heterodata import csv_to_hetero

class TargetMLP(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super(TargetMLP, self).__init__()
        self.mlp = MLP(in_channels=in_channels, 
                       hidden_channels=hidden_channels, 
                       out_channels=out_channels, 
                       num_layers=num_layers,
                       norm='layer_norm')

    def forward(self, x_dict):
        out = dict()
        for key, x in x_dict.items():
            if key == 'target':
                out[key] = self.mlp(x)
            else:
                out[key] = x
        return out

def build_hetero_gnn(model_type, data: HeteroData, types: list, hidden_channels: int = 64, 
                     num_layers: int = 2, out_channels: int = 2, aggr: str = 'sum', 
                     model_kwargs: dict = {}, mlp: bool = False):
    """
    model_types: GAT, EdgeCNN, GCN, GraphSAGE, GIN
    """
    if mlp:
        out_channels_mlp = out_channels
        out_channels = hidden_channels
    if model_type == 'GAT':
        model = GAT(in_channels=hidden_channels, hidden_channels=hidden_channels, num_layers=num_layers, out_channels=out_channels, add_self_loops=False, **model_kwargs)
    elif model_type == 'EdgeCNN':
        model = EdgeCNN(in_channels=hidden_channels, hidden_channels=hidden_channels, out_channels=out_channels, num_layers=num_layers, **model_kwargs)
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
    model_layers = [
        (HeteroDictLinear(in_channels=-1, out_channels=hidden_channels, types=types), 'x_dict -> x1'),
        (gnn_model, 'x1, edge_index -> y'),
    ]
    if mlp:
        model_layers.append((TargetMLP(in_channels=hidden_channels, hidden_channels= 2 * hidden_channels, 
                                 out_channels=out_channels_mlp, num_layers=3), 'y -> target'))
    model = Sequential('x_dict, edge_index', model_layers)
    return model


if __name__ == '__main__':
    data = csv_to_hetero('rossmann_subsampled', 'historical', 'Customers')
    model = build_hetero_gnn('GAT', data, types=list(data.x_dict.keys()), hidden_channels=64, num_layers=2, out_channels=1)
    output = model(data.x_dict, data.edge_index_dict)
    
    assert output['target'].shape[0] == data['target'].y.shape[0]
