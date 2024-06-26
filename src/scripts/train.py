import argparse
from datetime import datetime

import numpy as np
from tqdm import tqdm
import torch
from torch import optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter

from realog.hetero_gnns import build_hetero_gnn
from realog.table_to_heterodata import csv_to_hetero_splits

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_device(device)

def train(model, data_train, data_val = None, data_test = None, task='regression', num_epochs=10000, patience=50, lr=0.01, weight_decay=0.1, reduce_fac=0.1):
    run_name = f'{model.__class__.__name__}_lr{lr}_weight_decay{weight_decay}_reduce_fac{reduce_fac}'
    writer = SummaryWriter(logdir="logs/" + run_name + "run_datetime" + datetime.now().strftime("%Y%m%d-%H%M%S"))
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=0.00001)
    model.train()
    pbar = tqdm(range(num_epochs))
    val_loss = None
    best_loss = np.inf
    _patience = patience
    
    batch = data_train
    if task == 'classification':
        loss_fn = F.cross_entropy
    elif task == 'regression':
        loss_fn = F.mse_loss

    for epoch in pbar:
        optimizer.zero_grad()
        out = model(batch.x_dict, batch.edge_index_dict)

        loss = loss_fn(out['target'], batch['target'].y)
        loss.backward()
        optimizer.step()

        if data_val is not None and epoch % 1 == 0:
            model.eval()
            with torch.no_grad():
                val_out = model(data_val.x_dict, data_val.edge_index_dict)
                val_loss = loss_fn(val_out['target'], data_val['target'].y)
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
        if task == 'classification':
            val_acc = (val_out['target'].argmax(dim=-1) == data_val['target'].y).sum().item() / data_val['target'].y.shape[0]

            pbar.set_description(f'Loss: {loss.item():.4f} | Val Loss: {val_loss.item() if val_loss is not None else -1:.4f} | Val Acc: {val_acc:.4f} | LR: {optimizer.param_groups[0]["lr"]:.6f}')
        elif task == 'regression':
            pbar.set_description(f'Loss: {np.sqrt(loss.item()):.4f} | Val Loss: {np.sqrt(val_loss.item())if val_loss is not None else -1:.4f} | LR: {optimizer.param_groups[0]["lr"]:.6f}')
        
        # if scheduler is multi step
        if scheduler.__class__.__name__ == 'ReduceLROnPlateau':
            scheduler.step(loss)
        else:
            scheduler.step()

        if task == 'classification':
            train_loss = loss.item()
            val_loss = val_loss.item() if val_loss is not None else -1
        elif task == 'regression':
            train_loss = np.sqrt(loss.item())
            val_loss = np.sqrt(val_loss.item()) if val_loss is not None else -1
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss if val_loss is not None else -1, epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]["lr"], epoch)

    model.load_state_dict(best_model)

    # evaluate on test set
    if data_test is not None:
        model.eval()
        with torch.no_grad():
            test_out = model(data_test.x_dict, data_test.edge_index_dict)
            test_loss = F.mse_loss(test_out['target'], data_test['target'].y)
            print(f'Test Loss: {np.sqrt(test_loss.item()):.4f}, Best Val Loss: {np.sqrt(best_loss.item()):.4f}')
            writer.add_scalar('Loss/test', np.sqrt(test_loss.item()), epoch)

        outs = test_out['target'].detach().cpu().numpy()
        targets = data_test['target'].y.cpu().numpy()
        losses = []
        for i in range(100):
            boot_indices = np.random.choice(range(len(outs)), len(outs), replace=True)
            boot_outs = outs[boot_indices]
            boot_targets = targets[boot_indices]
            boot_loss = np.sqrt(F.mse_loss(torch.tensor(boot_outs), torch.tensor(boot_targets)).item())
            losses.append(boot_loss)
        print(f'Bootstrapped Test Loss: {np.mean(losses):.4f} +/- {np.std(losses)/10:.4f}')

    # plot the residuals
    preds = test_out['target'].detach().cpu().numpy()
    target = data_test['target'].y.cpu().numpy()
    # plot x=y line
    plt.plot([0, target.max()], [0, target.max()], 'k--')
    plt.scatter(target, preds)
    plt.show()
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Biodegradability_v1', choices=['rossmann', 'financial_v1', 'Biodegradability_v1'])
    parser.add_argument('--target_table', type=str, default='molecule')
    parser.add_argument('--target_column', type=str, default='activity')
    parser.add_argument('--task', type=str, default='regression', choices=['classification', 'regression'])
    parser.add_argument('--model', type=str, default='GAT', choices=['GAT', 'EdgeCNN', 'GraphSAGE', 'GIN', 'GATv2'])
    parser.add_argument('--no_skip_connections', default=False, action='store_true')
    args = parser.parse_args()

    dataset = args.dataset
    target_table = args.target_table
    target = args.target_column
    task = args.task
    model_name = args.model
    skip_connections = not args.no_skip_connections

    
    data_train, data_val, data_test = csv_to_hetero_splits(dataset, target_table, target, task, add_skip_connections=skip_connections)
    
    
    # sanity check that feature dimensions match
    for table in data_train.x_dict.keys():
        assert data_train[table].x.shape[1] == data_val[table].x.shape[1] == data_test[table].x.shape[1]

    if task == 'classification':
        out_channels = data_train['target'].num_classes
    elif task == 'regression':
        out_channels = 1
    model_name = 'GIN'
    print(f'Training {model_name} model, dataset: {dataset}, target: {target}, task: {task}')
    model = build_hetero_gnn(model_name, data_train, aggr='sum', types=list(data_train.x_dict.keys()), hidden_channels=128, num_layers=5, out_channels=out_channels, mlp_layers=5, model_kwargs={'dropout': 0.0, 'jk': 'lstm'})
    train(model, data_train, data_val, data_test, task=task, num_epochs=1000, patience=1000, lr=0.01, weight_decay=0.05, reduce_fac=0.5)