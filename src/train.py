from tqdm import tqdm
from torch import optim
import torch.nn.functional as F
from torch_geometric.sampler import BaseSampler
from torch_geometric.loader import HGTLoader, NodeLoader, NeighborLoader
import torch
import numpy as np

from hetero_gnns import build_hetero_gnn
from table_to_heterodata import csv_to_hetero

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_device(device)

def train(model, data, num_epochs=10000):
    # TODO: add dataloader
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[500, 1000], gamma=0.5)
    model.train()
    pbar = tqdm(range(num_epochs))
    for epoch in pbar:
        for batch in pbar:
            batch = data
            optimizer.zero_grad()
            out = model(batch.x_dict, batch.edge_index_dict)
            loss = F.mse_loss(out['target'], batch['target'].y)
            loss.backward()
            optimizer.step()
            pbar.set_description(f'Loss: {np.sqrt(loss.item()):.4f} | LR: {optimizer.param_groups[0]["lr"]:.6f}')
        scheduler.step()
    return model

if __name__ == '__main__':
    data = csv_to_hetero('rossmann_subsampled', 'historical', 'Customers')
    model = build_hetero_gnn('GIN', data, aggr='mean', types=list(data.x_dict.keys()), hidden_channels=128, num_layers=3, out_channels=1)
    model = train(model, data, num_epochs=1000)

