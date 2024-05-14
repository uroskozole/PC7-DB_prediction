import numpy as np
import scipy.sparse as sp
from torch_geometric.utils import to_scipy_sparse_matrix


def get_connected_components(data):
    homo = data.to_homogeneous()
    adj = to_scipy_sparse_matrix(homo.edge_index.cpu())

    num_components, component = sp.csgraph.connected_components(adj, connection="weak")
    components = dict()
    for i, key in enumerate(data.x_dict.keys()):
        components[key] = component[homo.node_type == i]

    connected_components = []
            
    for component in np.arange(num_components):
        nodes = dict()
        for key, ccs in components.items():
            nodes[key] = np.argwhere(ccs == component).flatten()
        connected_components.append(data.subgraph(nodes))

    return connected_components
