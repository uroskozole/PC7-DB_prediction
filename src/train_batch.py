from tqdm import tqdm
from torch import optim
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
import torch
import numpy as np
from sklearn.metrics import f1_score


from realog.utils.metadata import Metadata
from realog.hetero_gnns import build_hetero_gnn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_device(device)

def train(model, data_train, data_val, data_test, task='regression', num_epochs=10000, lr=0.01, weight_decay=0.1, class_weights=None):
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80, 150], gamma=0.1)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=0.00001)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=reduce_fac, patience=20, min_lr=0.00001)

    model.train()
    val_loss = None
    best_loss = np.inf
    # from samplers import get_connected_components
    trainloader = DataLoader(data_train, batch_size=32, shuffle=False, 
                            #  generator=torch.Generator(device=device), drop_last=True
                            )
    valloader = DataLoader(data_val, batch_size=32, shuffle=False, 
                        #    generator=torch.Generator(device=device), 
                           drop_last=False)
    testloader = DataLoader(data_test, batch_size=128, shuffle=False, 
                            # generator=torch.Generator(device=device), 
                            drop_last=False)

    if task == 'classification':
        loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
    elif task == 'regression':
        loss_fn = F.mse_loss

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
            # print(f'TRAIN: True distribution: {torch.bincount(batch["target"].y, minlength=2)} | Pred distribution: {torch.bincount(out["target"].argmax(dim=-1))}')
            # # compute recall and precision for class 1
            # pos_mask = batch['target'].y == 1
            # pos_pred_mask = out['target'].argmax(dim=-1) == 1
            # true_pos = (pos_mask & pos_pred_mask).sum().item()
            # false_pos = (~pos_mask & pos_pred_mask).sum().item()
            # false_neg = (pos_mask & ~pos_pred_mask).sum().item()
            # if true_pos + false_pos > 0:
            #     precision = true_pos / (true_pos + false_pos)
            # else:
            #     precision = np.nan
            # if true_pos + false_neg > 0:
            #     recall = true_pos / (true_pos + false_neg)
            # else:
            #     recall = np.nan
            # print(f'Precision: {precision:.4f} | Recall: {recall:.4f}')

        # validation
        model.eval()
        with torch.no_grad():
            pbar = tqdm(valloader)
            val_loss = []
            val_correct = 0
            val_total = 0
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
                    # print(f'VAL  : True distribution: {torch.bincount(batch["target"].y, minlength=2)} | Pred distribution: {torch.bincount(out["target"].argmax(dim=-1))}')
                    # compute recall and precision for class 1
                    # pos_mask = batch['target'].y == 1
                    # pos_pred_mask = out['target'].argmax(dim=-1) == 1
                    # true_pos = (pos_mask & pos_pred_mask).sum().item()
                    # false_pos = (~pos_mask & pos_pred_mask).sum().item()
                    # false_neg = (pos_mask & ~pos_pred_mask).sum().item()
                    # precision = true_pos / (true_pos + false_pos) if true_pos + false_pos > 0 else np.nan
                    # recall = true_pos / (true_pos + false_neg) if true_pos + false_neg > 0 else np.nan
                    # print(f'Precision: {precision:.4f} | Recall: {recall:.4f}')
                pbar.set_postfix({'val_loss': np.mean(val_loss), 'val_acc': val_correct / val_total})
            
        if np.mean(val_loss) < best_loss:
            best_loss = np.mean(val_loss)
            best_model = model.state_dict()

        
        if task == 'classification': 
            print(f'Epoch: {epoch} | Loss: {np.mean(train_loss):.4f} | Val Loss: {np.mean(val_loss):.4f} | Val Acc: {val_correct / val_total:.4f} | LR: {optimizer.param_groups[0]["lr"]:.6f}')
        elif task == 'regression':
            print(f'Epoch: {epoch} | Loss: {np.mean(train_loss):.4f} | Val Loss: {np.mean(val_loss):.4f} | LR: {optimizer.param_groups[0]["lr"]:.6f}')
        
        # if task == 'classification':
        #     pbar.set_postfix({'train_loss': np.mean(train_loss), 'train_acc': np.mean(train_acc)})
        # elif task == 'regression':
        #     pbar.set_description(f'Loss: {np.sqrt(loss.item()):.4f} | Val Loss: {np.sqrt(val_loss.item())if val_loss is not None else -1:.4f} | LR: {optimizer.param_groups[0]["lr"]:.6f}')
        
        # if scheduler is multi step
        if scheduler.__class__.__name__ == 'ReduceLROnPlateau':
            scheduler.step(loss)
        else:
            scheduler.step()

    model.load_state_dict(best_model)

    # evaluate on test set
    with torch.no_grad():
        pbar = tqdm(testloader)
        test_loss = []
        test_correct = 0
        test_total = 0
        for batch in pbar:
            batch = batch.to(device)
            out = model(batch.x_dict, batch.edge_index_dict)
            loss = loss_fn(out['target'], batch['target'].y)
            test_loss.append(loss.item())
            if task == 'classification':
                preds = out['target'].argmax(dim=-1).cpu().numpy()
                targets = batch['target'].y.cpu().numpy()
                correct = (targets == preds).sum().item()
                total = batch['target'].y.shape[0]
                test_correct += correct
                test_total += total
                # print(f'TEST : True distribution: {torch.bincount(batch["target"].y, minlength=2)} | Pred distribution: {torch.bincount(out["target"].argmax(dim=-1))}') 
                # compute recall and precision for class 1
                pos_mask = batch['target'].y == 1
                pos_pred_mask = out['target'].argmax(dim=-1) == 1
                true_pos = (pos_mask & pos_pred_mask).sum().item()
                false_pos = (~pos_mask & pos_pred_mask).sum().item()
                false_neg = (pos_mask & ~pos_pred_mask).sum().item()

                precision = true_pos / (true_pos + false_pos) if true_pos + false_pos > 0 else np.nan
                recall = true_pos / (true_pos + false_neg) if true_pos + false_neg > 0 else np.nan
                print('F1 Score: ', f1_score(targets, preds))
                print(f'Precision: {precision:.4f} | Recall: {recall:.4f}')
                print('Test Loss: ', np.mean(test_loss), 'Test Acc: ', test_correct / test_total)
            

    print(f'Best Val Loss: {best_loss:.4f} | Test Loss: {np.mean(test_loss):.4f} | Test Acc: {test_correct / test_total:.4f}')
    return model


if __name__ == '__main__':
    # dataset = 'Biodegradability_v1'
    # target_table = 'molecule'
    # target = 'activity'
    # task = 'regression'

    # dataset = "rossmann"
    # target_table = "historical"
    # target = "Customers"
    # task = 'regression'
    
    dataset = 'financial_v1'
    task = 'classification'

    import pickle
    metadata = Metadata().load_from_json(f'data/{dataset}/metadata.json')
    with open(f'data/{dataset}/train_subgraphs.pkl', 'rb') as f:
        train_data = pickle.load(f)

    with open(f'data/{dataset}/val_subgraphs.pkl', 'rb') as f:
        val_data = pickle.load(f)

    with open(f'data/{dataset}/test_subgraphs.pkl', 'rb') as f:
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
            if y == 1:
                oversampled_train_data.append(train_data[i])
    # train_data += oversampled_train_data * 4
    print(weights)
    weights = 1 / weights
    node_types = metadata.get_tables() + ['target']

    print(len(train_data), len(val_data), len(test_data))
    
    model = build_hetero_gnn('GIN', train_data[idx], aggr='mean', types=node_types, hidden_channels=128, num_layers=4, out_channels=out_channels, mlp_layers=3, model_kwargs={'dropout': 0.1})
    train(model, train_data, val_data, test_data, task=task, num_epochs=1, lr=0.0001, weight_decay=0.1, class_weights=weights)