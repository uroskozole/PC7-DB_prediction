from copy import deepcopy
import argparse
import pickle

from tqdm import tqdm
from torch import optim
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import RemoveIsolatedNodes
import torch
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score


from realog.utils.metadata import Metadata
from realog.hetero_gnns import build_hetero_gnn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_device(device)

def evaluate(model, testloader, loss_fn, task='regression'):
    model.eval()
    # evaluate on test set
    with torch.no_grad():
        pbar = tqdm(testloader)
        test_loss = []
        test_correct = 0
        test_total = 0
        pred = []
        gt   = []
        for batch in pbar:
            batch = batch.to(device)
            out = model(batch.x_dict, batch.edge_index_dict)
            loss = loss_fn(out['target'], batch['target'].y)
            test_loss.append(loss.item())
            if task == 'classification':
                preds = out['target'].argmax(dim=-1).cpu().numpy()
                targets = batch['target'].y.cpu().numpy()
                pred += preds.tolist()
                gt   += targets.tolist()
                correct = (targets == preds).sum().item()
                total = batch['target'].y.shape[0]
                test_correct += correct
                test_total += total
                
    if task == 'classification':
        f1 = f1_score(gt, pred, average='binary', pos_label=1, zero_division=0)
        precision  = precision_score(gt, pred, average='binary', pos_label=1, zero_division=0)
        recall  = recall_score(gt, pred, average='binary', pos_label=1, zero_division=0)
        print('F1 Score: ', f1)
        print(f'Precision: {precision:.4f} | Recall: {recall:.4f}')
        return test_loss, test_correct / test_total
    elif task == 'regression':
        print('Test Loss: ', np.mean(test_loss))
        return test_loss, np.sqrt(test_loss).mean()


def train(model, data_train, data_val, data_test, task='regression', num_epochs=10000, lr=0.01, weight_decay=0.1, batch_size=64, class_weights=None):
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    model.train()
    # from samplers import get_connected_components
    trainloader = DataLoader(data_train, batch_size=batch_size, shuffle=True, generator=torch.Generator(device=device), drop_last=True)
    valloader = DataLoader(data_val, batch_size=batch_size, shuffle=False, drop_last=False)
    testloader = DataLoader(data_test, batch_size=batch_size, shuffle=False, drop_last=False)
    
    total_steps = len(trainloader) * num_epochs
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr * 10, total_steps=total_steps, pct_start=0.3, anneal_strategy='cos', three_phase=False)


    if task == 'classification':
        loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
        best_val_f1 = 0
    elif task == 'regression':
        loss_fn = F.mse_loss
    
    best_loss = np.inf

    for epoch in range(num_epochs):
        train_loss = []
        train_acc = []
        pbar = tqdm(trainloader)
        model.train()
        for batch in pbar:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch.x_dict, batch.edge_index_dict)

            loss = loss_fn(out['target'], batch['target'].y)
            if task == 'classification':
                acc = (out['target'].argmax(dim=-1) == batch['target'].y).sum().item() / batch['target'].y.size(0)
                train_acc.append(acc)
            train_loss.append(loss.item())
            loss.backward()
            optimizer.step()
            pbar.set_postfix({'train_loss': np.mean(train_loss), 'train_acc': np.mean(train_acc)})
            # OneCycleLR is stepped after each batch
            scheduler.step()

        # validation
        model.eval()
        with torch.no_grad():
            pbar = tqdm(valloader)
            val_loss = []
            val_correct = 0
            val_total = 0
            pred = []
            gt   = []
            for batch in pbar:
                batch = batch.to(device)
                out = model(batch.x_dict, batch.edge_index_dict)
                loss = loss_fn(out['target'], batch['target'].y)
                val_loss.append(loss.item())
                if task == 'classification':
                    correct = (out['target'].argmax(dim=-1) == batch['target'].y).sum().item()
                    total = batch['target'].y.size(0)
                    val_correct += correct
                    val_total += total
                    preds = out['target'].argmax(dim=-1).cpu().numpy().tolist()
                    targets = batch['target'].y.cpu().numpy().tolist()
                    pred += preds
                    gt   += targets
                    pbar.set_postfix({'val_loss': np.mean(val_loss), 'val_acc': val_correct / val_total})
                elif task == 'regression':
                    pbar.set_postfix({'val_loss': np.sqrt(np.mean(val_loss))})
       
        if task == 'classification':
            val_f1 = f1_score(gt, pred, average='binary', pos_label=1, zero_division=0)
            val_p  = precision_score(gt, pred, average='binary', pos_label=1, zero_division=0)
            val_r  = recall_score(gt, pred, average='binary', pos_label=1, zero_division=0)
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_loss = np.mean(val_loss)
                best_model = deepcopy(model.state_dict())
            elif val_f1 == best_val_f1 and np.mean(val_loss) < best_loss:
                best_loss = np.mean(val_loss)
                best_model = deepcopy(model.state_dict())
        elif task == 'regression':
            if np.mean(val_loss) < best_loss:
                best_loss = np.mean(val_loss)
                best_model = deepcopy(model.state_dict())

        
        if task == 'classification': 
            print(f'Epoch: {epoch} | Loss: {np.mean(train_loss):.4f} | Val Loss: {np.mean(val_loss):.4f} | Val Acc: {val_correct / val_total:.4f} | F1: {val_f1:.4f} | Precision: {val_p:.4f} | Recall: {val_r:.4f} | LR: {optimizer.param_groups[0]["lr"]:.6f}')
        elif task == 'regression':
            print(f'Epoch: {epoch} | Loss: {np.sqrt(np.mean(train_loss)):.4f} | Val Loss: {np.sqrt(np.mean(val_loss)):.4f} | LR: {optimizer.param_groups[0]["lr"]:.6f}')
        
        test_loss, test_acc = evaluate(model, testloader, loss_fn, task=task)
        print('Test Loss: ', np.mean(test_loss), 'Test Acc: ', test_acc)

    # Evaluate the best model
    model.load_state_dict(best_model)
    test_loss, test_acc = evaluate(model, testloader, loss_fn, task=task)
    print(f'Best Val Loss: {best_loss:.4f} | Test Loss: {np.mean(test_loss):.4f} | Test Acc: {test_acc:.4f}')
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='financial_v1', choices=['rossmann', 'financial_v1'])
    parser.add_argument('--model', type=str, default='GAT', choices=['GAT', 'EdgeCNN', 'GraphSAGE', 'GIN', 'GATv2'])
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--oversample', action='store_true')
    args = parser.parse_args()

    model = args.model
    dataset = args.dataset
    batch_size = args.batch_size
    if dataset == 'rossmann':
        task = 'regression'
    elif dataset == 'financial_v1':
        task = 'classification'
        oversample = args.oversample

    
    data_dir = 'data'#'/d/hpc/projects/FRI/vh0153/PC7-DB_prediction/data'
    metadata = Metadata().load_from_json(f'{data_dir}/{dataset}/metadata.json')
    with open(f'{data_dir}/{dataset}/train_subgraphs.pkl', 'rb') as f:
        train_data = pickle.load(f)

    with open(f'{data_dir}/{dataset}/val_subgraphs.pkl', 'rb') as f:
        val_data = pickle.load(f)

    with open(f'{data_dir}/{dataset}/test_subgraphs.pkl', 'rb') as f:
        test_data = pickle.load(f)



    if task == 'classification':
        out_channels = train_data[0]['target'].num_classes
        weights = torch.tensor([1.0, 1.0])
    elif task == 'regression':
        out_channels = 1
        weights = None

    
    oversampled_train_data = []
    for i in range(len(train_data)):
        # use a subgraph which includes all tables for model initialization
        if list(train_data[i].x_dict.keys()) == metadata.get_tables() + ['target']:
            idx = i
        if task == 'classification':
            y = train_data[i]['target'].y.item()
            weights[y] += 1
            if y == 1 and oversample:
                # augment the minority class sample
                data_dict = train_data[i].to_dict()
                for table in train_data[i].x_dict.keys():
                    if table != 'target' and table != 'loan' and train_data[i].x_dict[table].size(0) > 1:
                        # randomly sample 0.9 of the rows
                        indices = torch.randperm(train_data[i].x_dict[table].size(0))[:int(0.9 * train_data[i].x_dict[table].size(0))]
                        for edge_type in train_data[i].edge_index_dict.keys():
                            if table == edge_type[0]:
                                mask = torch.isin(train_data[i].edge_index_dict[edge_type][0], indices.cpu())
                            elif table == edge_type[1]:
                                mask = torch.isin(train_data[i].edge_index_dict[edge_type][1], indices.cpu())
                            else:
                                continue
                            data_dict[edge_type]['edge_index'] = train_data[i].edge_index_dict[edge_type][:, mask.cpu()]
                duplicate = RemoveIsolatedNodes()(HeteroData().from_dict(data_dict))
                oversampled_train_data.append(duplicate)
    
    # TODO: Do we want to oversample the minority class and also weight the loss?
    
    if task == 'classification':
        if oversample:
            print('Oversampling')
            train_data += oversampled_train_data
            weights[1] += len(oversampled_train_data)
        weights = 1 / weights
    node_types = metadata.get_tables() + ['target']

    model_kwargs = {'dropout': 0.1, 'jk':'cat'}
    if model == 'GATv2':
        model = 'GAT'
        model_kwargs['v2'] = True
    
    model = build_hetero_gnn(model, train_data[idx], aggr='mean', types=node_types, hidden_channels=256, num_layers=2, out_channels=out_channels, mlp_layers=5, model_kwargs=model_kwargs)
    train(model, train_data, val_data, test_data, task=task, num_epochs=1000, lr=0.00001, weight_decay=0.1, class_weights=weights, batch_size=batch_size)