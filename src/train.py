from tqdm import tqdm
from torch import optim
import torch.nn.functional as F
from torch_geometric.sampler import BaseSampler
from torch_geometric.loader import HGTLoader, NodeLoader, NeighborLoader, DataLoader
from torch_geometric.datasets import OGB_MAG
import torch_geometric.transforms as T
import torch
import numpy as np

from datetime import datetime

from tensorboardX import SummaryWriter

from hetero_gnns import build_hetero_gnn
from table_to_heterodata import csv_to_hetero, csv_to_hetero_splits

device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
torch.set_default_device(device)

def train(model, data_train, data_val = None, data_test = None, num_epochs=10000, patience=50, lr=0.01, weight_decay=0.1, reduce_fac=0.1):
    run_name = f'{model.__class__.__name__}_lr{lr}_weight_decay{weight_decay}_reduce_fac{reduce_fac}'
    writer = SummaryWriter(logdir="logs/" + run_name + "run_datetime" + datetime.now().strftime("%Y%m%d-%H%M%S"))
    # TODO: add dataloader
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80, 150], gamma=0.1)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=reduce_fac, patience=20, min_lr=0.00001)
    model.train()
    pbar = tqdm(range(num_epochs))
    val_loss = None
    best_loss = np.inf
    _patience = patience
    # from samplers import get_connected_components
    # train_connected_components = get_connected_components(data_train, device=device)
    # dataloader = DataLoader(train_connected_components, batch_size=64, shuffle=True, generator=torch.Generator(device=device))
    batch = data_train
    for epoch in pbar:
        train_loss = []
        # for batch in dataloader:
        # batch = data_train
        optimizer.zero_grad()
        out = model(batch.x_dict, batch.edge_index_dict)
        loss = F.mse_loss(out['target'], batch['target'].y)
        loss.backward()
        optimizer.step()
        train_loss.append(np.sqrt(loss.item()))

        if data_val is not None and epoch % 1 == 0:
            model.eval()
            with torch.no_grad():
                val_out = model(data_val.x_dict, data_val.edge_index_dict)
                val_loss = F.mse_loss(val_out['target'], data_val['target'].y)
                # pbar.set_description(f'Loss: {np.sqrt(loss.item()):.4f} | Val Loss: {np.sqrt(val_loss.item()):.4f} | LR: {optimizer.param_groups[0]["lr"]:.6f}')
            if val_loss < best_loss:
                best_loss = val_loss
                _patience = patience
                best_model = model.state_dict()
            else:
                _patience -= 1
                if _patience == 0:
                    print('Early stopping')
                    break
            model.train()
        pbar.set_description(f'Loss: {np.mean(train_loss):.4f} | Val Loss: {np.sqrt(val_loss.item())if val_loss is not None else -1:.4f} | LR: {optimizer.param_groups[0]["lr"]:.6f}')
        # if scheduler is multi step
        if scheduler.__class__.__name__ == 'MultiStepLR':
            scheduler.step()
        else:
            scheduler.step(loss)
        # scheduler.step()
        writer.add_scalar('Loss/train', np.sqrt(loss.item()), epoch)
        writer.add_scalar('Loss/val', np.sqrt(val_loss.item()) if val_loss is not None else -1, epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]["lr"], epoch)

    model.load_state_dict(best_model)

    # evaluate on test set
    if data_test is not None:
        model.eval()
        with torch.no_grad():
            test_out = model(data_test.x_dict, data_test.edge_index_dict)
            test_loss = F.mse_loss(test_out['target'], data_test['target'].y)
            print(f'Test Loss: {np.sqrt(test_loss.item()):.4f}')
            writer.add_scalar('Loss/test', np.sqrt(test_loss.item()), epoch)
        # scheduler.step()

    import matplotlib.pyplot as plt
    preds = test_out['target'].detach().cpu().numpy()
    target = data_test['target'].y.cpu().numpy()
    # plot x=y line
    plt.plot([0, target.max()], [0, target.max()], 'k--')
    plt.scatter(target, preds)
    plt.show()
    return model


if __name__ == '__main__':
    dataset = 'Biodegradability_v1'
    target_table = 'molecule'
    target = 'activity'
    # dataset = "rossmann"
    # target_table = "historical"
    # target = "Customers"
    # data_train, data_val, data_test = csv_to_hetero_splits('rossmann', 'historical', 'Customers')
    data_train, data_val, data_test = csv_to_hetero_splits(dataset, target_table, target)
    
    # sanity check that feature dimensions match
    from utils.metadata import Metadata
    metadata = Metadata().load_from_json(f'data/{dataset}/metadata.json')
    for table in metadata.get_tables():
        print(table, data_train[table].x.shape[1], data_val[table].x.shape[1], data_test[table].x.shape[1])
        assert data_train[table].x.shape[1] == data_val[table].x.shape[1] == data_test[table].x.shape[1]

    model = build_hetero_gnn('GraphSAGE', data_train, aggr='mean', types=list(data_train.x_dict.keys()), hidden_channels=256, num_layers=10, out_channels=1, mlp=False)
    train(model, data_train, data_val, data_test, num_epochs=1000, patience=300, lr=0.0001, weight_decay=0.1, reduce_fac=0.1)