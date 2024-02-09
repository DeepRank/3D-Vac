import numpy as np

from copy import deepcopy
import pickle
import random

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from egnn import EGNNModel
from data_proccess_fn import data_process_fn

from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from scipy.special import expit

SPLIT = "ALLELE" # "SHUFFLE"

DATA_PATH = './pandora_dataset.pt'
ALLELE_SPLIT_PATH = './allele_split.pkl'

exclude_keys_all = ['angle_index', 'angle_targets', 'dihedral_index', 'dihedral_targets', 'residue_index', 'residue_id', ]

def get_split_according_to_dict(data_list, split_dict, key=lambda x: x.id):
    train_data = []
    val_data = []
    test_data = []
    
    for data in data_list:
        if key(data) in split_dict['train']:
            train_data.append(data)
        elif key(data) in split_dict['val']:
            val_data.append(data)
        elif key(data) in split_dict['test']:
            test_data.append(data)
        else:
            pass
            
    return train_data, val_data, test_data

def evaluate(model, loader, data_process_fn, device='cuda'): 
    model.to(device)
    model.eval()  # Set the model to evaluation mode

    total_loss = 0.0
    all_outputs = []
    all_targets = []
    all_bin_targets = []

    with torch.no_grad():  # Do not calculate gradients
        for i, batch in enumerate(loader):
            y_reg, y_bin = batch.y_reg, batch.y_bin
            indices = torch.arange(batch.x.shape[0])

            batch = data_process_fn(batch)
            outputs = model(batch.to(device)).squeeze()

            targets = y_bin

            loss = F.binary_cross_entropy_with_logits(outputs.squeeze(), targets.to(device).float())
            total_loss += loss.item()

            # Store outputs and targets for AUC calculation
            all_outputs.extend(outputs.detach().cpu().numpy())
            all_targets.extend(targets.detach().cpu().numpy())

        # Compute AUC
        auc_score = roc_auc_score(all_targets, expit(all_outputs))

    avg_loss = total_loss / len(loader)  # Compute the average loss
    accuracy = accuracy_score(all_targets, np.array(all_outputs) > 0.)

    return avg_loss, accuracy, auc_score

def train(model, optimizer, train_loader, val_loader, test_loader, data_process_fn, epochs, scheduler=None, device='cuda', log_interval=50, log_dict=None):
    model.to(device)
    model.train()

    best_model = deepcopy(model.state_dict())
    best_val_auc = 0

    if log_dict is None:
        log_dict = {
            'train_loss': [],
            'train_acc_final': [],
            'train_auc_final': [],
            'val_loss': [],
            'val_acc': [],
            'val_auc': [],
            'test_loss': [],
            'test_acc': [],
            'test_auc': [],
            'test_auc': [],
        }

    len_loader = len(train_loader)


    for epoch in range(epochs):

        if epoch % log_interval == 0:
            print(f'Epoch {epoch + 1}/{epochs}')

        for i in range(len_loader):
            batch = next(iter(train_loader))

            y_reg, y_bin = batch.y_reg, batch.y_bin
            indices = torch.arange(batch.x.shape[0])

            batch = data_process_fn(batch)
            outputs = model(batch.to(device))

            targets = y_bin

            loss =  F.binary_cross_entropy_with_logits(outputs.squeeze(), targets.to(device).float())

            if i % log_interval == 0:
                accuracy = accuracy_score(y_bin.detach().cpu().numpy(), outputs.detach().cpu().numpy() > 0.)
                print(f'Epoch {epoch + 1}, Batch {i + 1}, Loss {loss.item()}, Accuracy {accuracy}')

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        log_dict['train_loss'].append(loss.item())

        val_loss, val_acc, val_auc = evaluate(model, val_loader, data_process_fn, device=device)
        test_loss, test_acc, test_auc = evaluate(model, test_loader, data_process_fn, device=device)

        print(f'Epoch {epoch + 1}, Val Loss {val_loss}, Val Accuracy {val_acc}, Val AUC {val_auc}')
        print(f'Epoch {epoch + 1}, Test Loss {test_loss}, Test Accuracy {test_acc}, Test AUC {test_auc}')

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model = deepcopy(model.state_dict())

        log_dict['val_loss'].append(val_loss)
        log_dict['val_acc'].append(val_acc)
        log_dict['val_auc'].append(val_auc)

        log_dict['test_loss'].append(test_loss)
        log_dict['test_acc'].append(test_acc)
        log_dict['test_auc'].append(test_auc)


        if scheduler is not None:
            scheduler.step(val_auc)
            if optimizer.param_groups[0]['lr'] < 1e-6:
                print('Early stopping')
                break

    train_loss, train_acc_final, train_auc_final = evaluate(model, train_loader, data_process_fn, device=device)

    log_dict['train_acc_final'].append(train_acc_final)
    log_dict['train_auc_final'].append(train_auc_final)

    return best_model, log_dict



if __name__ == '__main__':
    dataset = torch.load(DATA_PATH)

    if SPLIT == "ALLELE":
        train_data, val_data, test_data = get_split_according_to_dict(dataset, pickle.load(open(ALLELE_SPLIT_PATH, 'rb')))
    elif SPLIT == "SHUFFLE":
        seed = 42
        random.seed(seed)
        random.shuffle(dataset)

        train_data = dataset[:int(len(dataset) * 0.8)]
        val_data = dataset[int(len(dataset) * 0.8):int(len(dataset) * 0.9)]
        test_data = dataset[int(len(dataset) * 0.9):]



    loader_train = DataLoader(train_data, batch_size=512, shuffle=True, exclude_keys=exclude_keys_all)
    loader_val = DataLoader(val_data, batch_size=512, shuffle=False, exclude_keys=exclude_keys_all)
    loader_test = DataLoader(test_data, batch_size=512, shuffle=False, exclude_keys=exclude_keys_all)
        

    gnn = EGNNModel(
                num_layers=3,
                emb_dim=128,
                edge_dim=1,
                in_dim=23,
                out_dim=1,
                pool='peptide_sum',
                cond_method='strong',
                deep_conditioning=False,
                separate_entity_embeddings=False,
                embed=True,
                rbf=True,
                rbf_max=30,
                rbf_dim=64,
                update_pos=False,
                dropout=0.1,
            )


    optimizer = torch.optim.AdamW(gnn.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.1, verbose=True, mode='max')

    best_model, log_dict = train(gnn, optimizer, loader_train, loader_val, loader_test, data_process_fn, epochs=150, scheduler=scheduler, device='cuda', log_interval=200, log_dict=None)

    torch.save(best_model, './best_model.pt')
    pickle.dump(log_dict, open('./log_dict.pkl', 'wb'))