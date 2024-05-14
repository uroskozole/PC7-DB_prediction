from tqdm import tqdm
from torch import optim
import torch.nn.functional as F
from torch_geometric.sampler import BaseSampler
from torch_geometric.loader import HGTLoader, NodeLoader, NeighborLoader, DataLoader
from torch_geometric.datasets import OGB_MAG
import torch_geometric.transforms as T
import torch
import numpy as np

from hetero_gnns import build_hetero_gnn
from table_to_heterodata import csv_to_hetero, csv_to_hetero_splits

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_device(device)

def train(model, data_train, data_val = None, data_test = None, num_epochs=10000):
    # TODO: add dataloader
    optimizer = optim.AdamW(model.parameters(), lr=0.1)
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150, 500, 1000], gamma=0.1)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, min_lr=0.00001)
    model.train()
    pbar = tqdm(range(num_epochs))
    val_loss = None
    for epoch in pbar:
        batch = data_train
        optimizer.zero_grad()
        out = model(batch.x_dict, batch.edge_index_dict)
        loss = F.mse_loss(out['target'], batch['target'].y)
        loss.backward()
        optimizer.step()

        if data_val is not None and epoch % 1 == 0:
            model.eval()
            with torch.no_grad():
                val_out = model(data_val.x_dict, data_val.edge_index_dict)
                val_loss = F.mse_loss(val_out['target'], data_val['target'].y)
                # pbar.set_description(f'Loss: {np.sqrt(loss.item()):.4f} | Val Loss: {np.sqrt(val_loss.item()):.4f} | LR: {optimizer.param_groups[0]["lr"]:.6f}')
            model.train()
        pbar.set_description(f'Loss: {np.sqrt(loss.item()):.4f} | Val Loss: {np.sqrt(val_loss.item())if val_loss is not None else -1:.4f} | LR: {optimizer.param_groups[0]["lr"]:.6f}')
        scheduler.step(loss)

    # evaluate on test set
    if data_test is not None:
        model.eval()
        with torch.no_grad():
            test_out = model(data_test.x_dict, data_test.edge_index_dict)
            test_loss = F.mse_loss(test_out['target'], data_test['target'].y)
            print(f'Test Loss: {np.sqrt(test_loss.item()):.4f}')
    return model


if __name__ == '__main__':
    data_train, data_val, data_test = csv_to_hetero_splits('rossmann', 'historical', 'Customers')
    # data_train = csv_to_hetero('rossmann', 'historical', 'Customers')

    model = build_hetero_gnn('GraphSAGE', data_train, aggr='mean', types=list(data_train.x_dict.keys()), hidden_channels=128, num_layers=3, out_channels=1)
    train(model, data_train, data_val, data_test, num_epochs=1000)