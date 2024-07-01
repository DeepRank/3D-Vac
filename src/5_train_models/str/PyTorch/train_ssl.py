import pickle
import random
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch_scatter
from data_proccess_fn import data_process_fn
from egnn import EGNNModel
from scipy.special import expit
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score

import torch_geometric
from torch_geometric.loader import DataLoader
import fire
import csv


DATA_PATH = "/projects/0/einf2380/data/pMHCI/egnn_data/ssl/train/propedia_residue_mhc_1.pt"
EVAL_DATA_PATH = "/projects/0/einf2380/data/pMHCI/egnn_data/ssl/eval/pandora_dataset_allele_test.pt"

SUP_DATA_PATH = "/projects/0/einf2380/data/pMHCI/egnn_data/supervised/pandora_dataset.pt"
SUP_ALLELE_SPLIT_PATH = "/projects/0/einf2380/data/pMHCI/egnn_data/supervised/allele_split.pkl"

CSV_PATH = '/projects/0/einf2380/data/external/processed/I/ssl_data_efficiency'
OUTPUT_PATH = '/projects/0/einf2380/data/pMHCI/trained_models/3DSSL'

device = "cuda" if torch.cuda.is_available() else "cpu"

exclude_keys_all = [
    "angle_index",
    "angle_targets",
    "dihedral_index",
    "dihedral_targets",
    "residue_index",
    "residue_id",
]


def get_split_according_to_dict(data_list, split_dict, key=lambda x: x.id):
    train_data = []
    val_data = []
    test_data = []

    for data in data_list:
        if key(data) in split_dict["train"]:
            train_data.append(data)
        elif key(data) in split_dict["val"]:
            val_data.append(data)
        elif key(data) in split_dict["test"]:
            test_data.append(data)
        else:
            pass

    return train_data, val_data, test_data


class ResidueTask(object):
    def __init__(
        self,
        task_name="residue_prediction",
        mask_ratio_range=[0.2, 0.2],
        mask_value=22,
        value_filter=None,
        field_filter=None,
        value_filter_eval=1,
        field_filter_eval="entity",
        label_smoothing=0.0,
    ):
        self.task_name = task_name
        self.mask_ratio_range = mask_ratio_range
        self.mask_value = mask_value

        self.value_filter = value_filter
        self.field_filter = field_filter

        self.value_filter_eval = value_filter_eval
        self.field_filter_eval = field_filter_eval

        self.label_smoothing = label_smoothing
        self.loss = F.cross_entropy

    def get_targets_and_indices(self, data):
        if self.value_filter is not None:
            targetable_indices = (
                torch.nonzero(data[self.field_filter] == self.value_filter)
                .squeeze()
                .to(data.x.device)
            )
        else:
            targetable_indices = torch.arange(data.num_nodes).to(data.x.device)

        mask_ratio = np.random.uniform(*self.mask_ratio_range)
        num_nodes_to_mask = int(mask_ratio * targetable_indices.shape[0])

        targetable_indices_subset = torch.randperm(targetable_indices.shape[0])[
            :num_nodes_to_mask
        ].to(data.x.device)
        target_indices = targetable_indices[targetable_indices_subset]

        targets = data.x[target_indices]
        data.x[target_indices] = self.mask_value

        return targets, target_indices

    def calculate_loss(self, predictions, targets):
        return self.loss(predictions, targets, label_smoothing=self.label_smoothing)

    def prepare_data_for_log_prob_calculation(self, data):
        if isinstance(data, torch_geometric.data.Batch):
            data_list = data.to_data_list()
        else:
            data_list = [data]

        data_prepared = []
        targets_list = []

        for data_num_id, data in enumerate(data_list):
            temp_data_list = []

            if self.value_filter_eval is not None:
                target_indices = (
                    torch.nonzero(
                        data[self.field_filter_eval] == self.value_filter_eval
                    )
                    .squeeze()
                    .to(data.x.device)
                )
            else:
                target_indices = torch.arange(data.num_nodes).to(data.x.device)

            targets = data.x[target_indices]

            targets_list.append(targets)

            for i, target_node_index in enumerate(target_indices):
                temp_data = data.clone()
                temp_data.x[target_node_index] = self.mask_value
                temp_data.y = targets[i].unsqueeze(0)
                temp_data.index = target_node_index.item()

                temp_data.num_id = data_num_id

                data_prepared.append(temp_data)

        data = torch_geometric.data.Batch.from_data_list(data_prepared)
        targets = torch.cat(targets_list, dim=0)
        target_indices = data.index

        return data, targets, target_indices

    def calculate_data_log_prob(
        self, predictions, targets, target_indices, data=None, reduce=False
    ):
        log_prob = F.log_softmax(predictions, dim=-1)[
            torch.arange(predictions.shape[0]), targets
        ]

        if reduce == False:
            return log_prob

        if data is None or data.num_id is None:
            batch = torch.zeros(targets.shape[0], dtype=torch.long).to(targets.device)
        else:
            batch = data.num_id

        return torch_scatter.scatter_add(log_prob, batch, dim=0)


def eval(model, task, test_loader, data_process_fn, device=device, log_interval=20, export_csv=False, split_name="eval"):
    model.eval()

    log_dict = {}

    diff_ids = set()

    if export_csv:
        csv_data = []

    with torch.no_grad():
        for i, data in enumerate(test_loader):
            if i % 10 == 0:
                print(f"{i/len(test_loader) * 100:.2f}%")
            data = data.to(device)

            for i, id in enumerate(data.id):
                if id not in log_dict:
                    log_dict[id] = {
                        "y_bin": data.y_bin[i].item(),
                        "y_reg": data.y_reg[i].item(),
                        "residue-add": 0,
                    }

            task_name = task.task_name

            if "residue" in task_name:
                task_name_ = "residue"
            else:
                task_name_ = task_name

            data_t = data.clone().to(device)

            data_t, targets, target_indices = (
                task.prepare_data_for_log_prob_calculation(data_t)
            )

            data_t = data_process_fn(data_t)

            outputs = model(data_t.to(device))[target_indices]

            log_prob = task.calculate_data_log_prob(
                outputs, targets.to(device), target_indices, data=data_t, reduce=False
            )

            for metric in ["add"]:
                log_prob_ = torch_scatter.scatter_add(
                    log_prob, data_t.num_id, dim=0, dim_size=data_t.num_id.max() + 1
                )

                for i in range(log_prob_.shape[0]):
                    id = data.id[i]
                    diff_ids.add(id)
                    log_dict[id]["residue-add"] = log_prob_[i].item()
                    if export_csv:
                        csv_data.append([id, log_prob_[i].item(), data.y_bin[i].item()])

        auc = roc_auc_score(
            [log_dict[id]["y_bin"] for id in log_dict],
            [log_dict[id]["residue-add"] for id in log_dict],
        )
        print("AUC on allele-split test set after SSL training:", auc)

        if export_csv:
            with open(f"{OUTPUT_PATH}/{split_name}_test_results.csv", "w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["Key", "Log_prob", "Target"])
                writer.writerows(csv_data)

    return [log_dict, {"auc": auc}]


def train(
    model,
    optimizer,
    task,
    train_loader,
    data_process_fn,
    epochs,
    scheduler=None,
    device=device,
    log_interval=50,
    log_dict=None,
):
    model.to(device)
    model.train()

    best_val_auc = 0

    if log_dict is None:
        log_dict = {
            "train_loss": [],
            "train_acc": [],
        }

    len_loader = len(train_loader)

    for epoch in range(epochs):
        if epoch % log_interval == 0:
            print(f"Epoch {epoch + 1}/{epochs}")

        for i in range(len_loader):
            batch = next(iter(train_loader))
            batch = data_process_fn(batch).to(device)

            targets, target_indices = task.get_targets_and_indices(batch)

            outputs = model(batch)
            outputs = outputs[target_indices]

            loss = task.calculate_loss(outputs, targets)

            if i % log_interval == 0:
                accuracy = accuracy_score(
                    targets.detach().cpu().numpy(),
                    outputs.argmax(dim=-1).detach().cpu().numpy() > 0.0,
                )
                print(
                    f"Epoch {epoch + 1}, Batch {i + 1}, Loss {loss.item()}, Accuracy {accuracy}"
                )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        log_dict["train_loss"].append(loss.item())
        log_dict["train_acc"].append(accuracy)

        if scheduler is not None:
            scheduler.step(loss)
            if optimizer.param_groups[0]["lr"] < 1e-6:
                print("Early stopping")
                break

    return model, log_dict


def main(data_choice="xray", seed=42, num_epochs=600, export_csv=True):
    SEED = int(seed)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    
    if data_choice == "xray":
        dataset = torch.load(DATA_PATH)
        batch_size=128
    elif data_choice == "pandora_1k":
        dataset = torch.load(SUP_DATA_PATH)
        train_data, _, _ = get_split_according_to_dict(
            dataset, pickle.load(open(SUP_ALLELE_SPLIT_PATH, "rb"))
        )
        # First take all of the positive data, then subsample the correct amount
        train_data = [data for data in train_data if data.y_bin == 1]
        train_data = random.sample(train_data, min(len(train_data), 1000))
        dataset = train_data
        
        batch_size = 128
        
    elif data_choice == "pandora_5k":
        dataset = torch.load(SUP_DATA_PATH)
        train_data, _, _ = get_split_according_to_dict(
            dataset, pickle.load(open(SUP_ALLELE_SPLIT_PATH, "rb"))
        )
        # First take all of the positive data, then subsample the correct amount
        train_data = [data for data in train_data if data.y_bin == 1]
        train_data = random.sample(train_data, min(len(train_data), 5000))
        dataset = train_data

        batch_size = 128
    elif data_choice == "pandora_10k":
        dataset = torch.load(SUP_DATA_PATH)
        train_data, _, _ = get_split_according_to_dict(
            dataset, pickle.load(open(SUP_ALLELE_SPLIT_PATH, "rb"))
        )
        # First take all of the positive data, then subsample the correct amount
        train_data = [data for data in train_data if data.y_bin == 1]
        train_data = random.sample(train_data, min(len(train_data), 10000))
        dataset = train_data

        batch_size = 128
    elif data_choice == "pandora_all":
        dataset = torch.load(SUP_DATA_PATH)
        train_data, _, _ = get_split_according_to_dict(
            dataset, pickle.load(open(SUP_ALLELE_SPLIT_PATH, "rb"))
        )
        train_data = [data for data in train_data if data.y_bin == 1]
        dataset = train_data
        
        batch_size = 512
    elif data_choice == "xray_pandora_all":
        dataset_xray = torch.load(DATA_PATH)
        dataset_pandora = torch.load(SUP_DATA_PATH)
        train_data, _, _ = get_split_according_to_dict(
            dataset_pandora, pickle.load(open(SUP_ALLELE_SPLIT_PATH, "rb"))
        )
        train_data = [data for data in train_data if data.y_bin == 1]

        # trim so train_data and dataset_xray have the same common elements
        xray_keys = set()
        for k,v in dataset_xray[0]:
            xray_keys.add(k)
        
        pandora_keys = set()
        for k,v in train_data[0]:
            pandora_keys.add(k)

        common_keys = xray_keys.intersection(pandora_keys)

        merged_dataset = []
        for el in train_data:
            new_el = torch_geometric.data.Data(
                **{k: v for k, v in el if k in common_keys}
            )
            merged_dataset.append(new_el)
        for el in dataset_xray:
            new_el = torch_geometric.data.Data(
                **{k: v for k, v in el if k in common_keys}
            )
            merged_dataset.append(new_el)

        batch_size = 512

        dataset = merged_dataset
        
    train_data_df = pd.DataFrame({'ID': [x['id'] for x in dataset]})
    train_data_df.to_csv(f'{CSV_PATH}/{data_choice}_{seed}_train.csv', index=False)

    loader_train = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, exclude_keys=exclude_keys_all
    )

    gnn = EGNNModel(
        num_layers=3,
        emb_dim=128,
        edge_dim=1,
        in_dim=23,
        out_dim=23,
        pool="none",
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

    task = ResidueTask()

    optimizer = torch.optim.AdamW(gnn.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=30, factor=0.1, verbose=True, mode="min"
    )

    print("Training model")
    best_model, log_dict = train(
        gnn,
        optimizer,
        task,
        loader_train,
        data_process_fn,
        epochs=num_epochs,
        scheduler=scheduler,
        device="cuda",
        log_interval=200,
        log_dict=None,
    )

    print("Training done, saving model and logs")
    torch.save(best_model, f"{OUTPUT_PATH}/{data_choice}_{seed}_model_ssl.pt")
    pickle.dump(log_dict, open(f"{OUTPUT_PATH}/{data_choice}_{seed}_log_dict_ssl.pkl", "wb"))

    print("Evaluating model on allele-split test set")
    eval_dataset = torch.load(EVAL_DATA_PATH)
    loader_eval = DataLoader(
        eval_dataset, batch_size=128, shuffle=False, exclude_keys=exclude_keys_all
    )
    log_dict_eval = eval(
        best_model, task, loader_eval, data_process_fn, export_csv=export_csv, split_name=f"{data_choice}_{seed}"
    )

    print("Evaluating done, saving logs")
    pickle.dump(log_dict_eval, open(f"{OUTPUT_PATH}/{data_choice}_{seed}_log_dict_eval_ssl.pkl", "wb"))


if __name__ == "__main__":
    fire.Fire(main)
