import numpy as np
import pandas as pd
import fire
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

import csv


#DATA_PATH = "/projects/0/einf2380/data/pMHCI/egnn_data/supervised/pandora_dataset.pt"
DATA_PATH = '/home/dmarz/test_cases/final_folders/egnn_extended_full_dataset.pt'
#DATASET_CSV = '/projects/0/einf2380/data/external/processed/I/CrossValidations/full_dataset.csv'
OUTPUT_PATH = '/projects/0/einf2380/data/pMHCI/trained_models/EGNN'

exclude_keys_all = [
    "angle_index",
    "angle_targets",
    "dihedral_index",
    "dihedral_targets",
    "residue_index",
    "residue_id",
]


def get_split_from_csv(data_list, train_csv, val_csv, test_csv, key=lambda x: x.id):
    train_ids = pd.read_csv(train_csv)['ID'].tolist()
    val_ids = pd.read_csv(val_csv)['ID'].tolist()
    test_ids = pd.read_csv(test_csv)['ID'].tolist()
    
    train_data = []
    val_data = []
    test_data = []

    for data in data_list:
        if key(data) in train_ids:
            train_data.append(data)
        elif key(data) in val_ids:
            val_data.append(data)
        elif key(data) in test_ids:
            test_data.append(data)
        else:
            pass

    return train_data, val_data, test_data


def evaluate(
    model, loader, data_process_fn, device="cuda", export_csv=False, split_name="split"
):
    model.to(device)
    model.eval()  # Set the model to evaluation mode

    total_loss = 0.0
    all_outputs = []
    all_targets = []
    all_bin_targets = []

    if export_csv:
        csv_data = []

    with torch.no_grad():  # Do not calculate gradients
        for i, batch in enumerate(loader):
            y_reg, y_bin = batch.y_reg, batch.y_bin
            indices = torch.arange(batch.x.shape[0])

            batch = data_process_fn(batch)
            outputs = model(batch.to(device)).squeeze()

            targets = y_bin

            loss = F.binary_cross_entropy_with_logits(
                outputs.squeeze(), targets.to(device).float()
            )
            total_loss += loss.item()

            # Store outputs and targets for AUC calculation
            all_outputs.extend(outputs.detach().cpu().numpy())
            all_targets.extend(targets.detach().cpu().numpy())

            if export_csv:
                for j in range(len(batch.id)):
                    csv_data.append([batch.id[j], outputs[j].item(), targets[j].item()])

        # Compute AUC
        auc_score = roc_auc_score(all_targets, expit(all_outputs))

    avg_loss = total_loss / len(loader)  # Compute the average loss
    accuracy = accuracy_score(all_targets, np.array(all_outputs) > 0.0)

    if export_csv:
        with open(f"{OUTPUT_PATH}/{split_name}_test_results.csv", "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Key", "Output", "Target"])
            writer.writerows(csv_data)

    return avg_loss, accuracy, auc_score


def train(
    model,
    optimizer,
    train_loader,
    val_loader,
    test_loader,
    data_process_fn,
    epochs,
    scheduler=None,
    device="cuda",
    log_interval=50,
    log_dict=None,
    export_csv=False,
    split_name="split",
    run_test=True,
):
    model.to(device)
    model.train()

    best_model = deepcopy(model.state_dict())
    best_val_auc = 0

    if log_dict is None:
        log_dict = {
            "train_loss": [],
            "train_acc_final": [],
            "train_auc_final": [],
            "val_loss": [],
            "val_acc": [],
            "val_auc": [],
            "val_acc_final": [],
            "val_auc_final": [],
            "test_loss": [],
            "test_acc": [],
            "test_auc": [],
        }

    len_loader = len(train_loader)

    for epoch in range(epochs):

        if epoch % log_interval == 0:
            print(f"Epoch {epoch + 1}/{epochs}")

        for i in range(len_loader):
            batch = next(iter(train_loader))

            y_reg, y_bin = batch.y_reg, batch.y_bin
            indices = torch.arange(batch.x.shape[0])

            batch = data_process_fn(batch)
            outputs = model(batch.to(device))

            targets = y_bin

            loss = F.binary_cross_entropy_with_logits(
                outputs.squeeze(), targets.to(device).float()
            )

            if i % log_interval == 0:
                accuracy = accuracy_score(
                    y_bin.detach().cpu().numpy(), outputs.detach().cpu().numpy() > 0.0
                )
                print(
                    f"Epoch {epoch + 1}, Batch {i + 1}, Loss {loss.item()}, Accuracy {accuracy}"
                )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        log_dict["train_loss"].append(loss.item())

        val_loss, val_acc, val_auc = evaluate(
            model,
            val_loader,
            data_process_fn,
            device=device,
            export_csv=False,
            split_name="val",
        )
        if run_test:
            test_loss, test_acc, test_auc = evaluate(
                model,
                test_loader,
                data_process_fn,
                device=device,
                export_csv=False,
                split_name="test",
            )

        print(
            f"Epoch {epoch + 1}, Val Loss {val_loss}, Val Accuracy {val_acc}, Val AUC {val_auc}"
        )
        if run_test:
            print(
                f"Epoch {epoch + 1}, Test Loss {test_loss}, Test Accuracy {test_acc}, Test AUC {test_auc}"
            )

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model = deepcopy(model.state_dict())

        log_dict["val_loss"].append(val_loss)
        log_dict["val_acc"].append(val_acc)
        log_dict["val_auc"].append(val_auc)

        if run_test:
            log_dict["test_loss"].append(test_loss)
            log_dict["test_acc"].append(test_acc)
            log_dict["test_auc"].append(test_auc)

        if scheduler is not None:
            scheduler.step(val_auc)
            if optimizer.param_groups[0]["lr"] < 1e-6:
                print("Early stopping")
                break

    if run_test:
        evaluate(
            model,
            test_loader,
            data_process_fn,
            device=device,
            export_csv=export_csv,
            split_name=f"{split_name}_final_test"
        )

    train_loss, train_acc_final, train_auc_final = evaluate(
        model,
        train_loader,
        data_process_fn,
        device=device,
        export_csv=export_csv,
        split_name=f"{split_name}_final_train",
    )
    
    log_dict["train_acc_final"].append(train_acc_final)
    log_dict["train_auc_final"].append(train_auc_final)
    
    val_loss, val_acc_final, val_auc_final = evaluate(
        model,
        val_loader,
        data_process_fn,
        device=device,
        export_csv=export_csv,
        split_name=f"{split_name}_final_val",
    )

    log_dict["val_acc_final"].append(val_acc_final)
    log_dict["val_auc_final"].append(val_auc_final)

    return best_model, log_dict


def main(
    train_csv, val_csv, test_csv, experiment, fold=42, run_test=True,
    export_csv=True, subset_1k=False, subset_10k=False, num_epochs=150,
):
    
    SEED = int(fold)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    dataset = torch.load(DATA_PATH)
    # dataset = [x for x in dataset if x.id in data_ids]

    train_data, val_data, test_data = get_split_from_csv(
        dataset, train_csv, val_csv, test_csv)

    if subset_1k:
        train_data = random.sample(train_data, min(len(train_data), 1000))
        split_name = f"{experiment}_1k"
    elif subset_10k:
        train_data = random.sample(train_data, min(len(train_data), 10000))
        split_name = f"{experiment}_10k"
    else:
        split_name = experiment

    loader_train = DataLoader(
        train_data, batch_size=512, shuffle=True, exclude_keys=exclude_keys_all
    )
    loader_val = DataLoader(
        val_data, batch_size=512, shuffle=False, exclude_keys=exclude_keys_all
    )
    loader_test = DataLoader(
        test_data, batch_size=512, shuffle=False, exclude_keys=exclude_keys_all
    )

    print('Data loaded. Setting Network')
    gnn = EGNNModel(
        num_layers=3,
        emb_dim=128,
        edge_dim=1,
        in_dim=23,
        out_dim=1,
        pool="peptide_sum",
        cond_method="strong",
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
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.1, verbose=True, mode="max"
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(len(train_data))
    print(experiment)
    print(subset_1k)
    print(device)

    print('Start training')
    
    best_model, log_dict = train(
        gnn,
        optimizer,
        loader_train,
        loader_val,
        loader_test,
        data_process_fn,
        epochs=num_epochs,
        scheduler=scheduler,
        device=device,
        log_interval=200,
        log_dict=None,
        export_csv=export_csv,
        split_name=split_name,
        run_test=run_test,
    )

    torch.save(best_model, f"{OUTPUT_PATH}/{split_name}_best_model.pt")
    pickle.dump(log_dict, open(f"{OUTPUT_PATH}/{split_name}_log_dict.pkl", "wb"))


if __name__ == "__main__":
    fire.Fire(main)
