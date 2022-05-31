from urllib.request import Request
import torch
import argparse
from torch import nn
from torch.utils.data import Subset
from torch.utils.data import DataLoader
import copy
import os.path as path
import os
import sys
sys.path.append(path.abspath("../../../../"))
from CNN.models import MlpRegBaseline
from CNN.datasets import Reg_Seq_Dataset, load_reg_seq_data
from CNN.datasets import sig_norm, li_norm, custom_norm #normalization methods for ba_values
import random
# import multiprocessing as mp
from mpi4py import MPI

# DEFINE CLI ARGUMENTS
#---------------------

arg_parser = argparse.ArgumentParser(
    description="Fully connected layer to generate a model which predicts the binding affinity based on one-hot encoded \
    peptide sequence. Works only for a fixed length of 9 residues. Takes as input the csv file (header free) containing \
    the list of peptides, the column for the peptides and a threshold for binding affinity to define binders."
)

arg_parser.add_argument("--csv-file", "-f",
    help="Absolute path of the csv file",
    default="../../../../../../data/binding_data/BA_pMHCI.csv"
)
arg_parser.add_argument("--peptide-column", "-p",
    type=int,
    help="Column index of peptide's sequence in the csv file",
    default=2
)
arg_parser.add_argument("--threshold", "-t",
    help="Binding affinity threshold to define binders, default 500",
    type=float,
    default=500e3
)
arg_parser.add_argument("--encoder", "-e",
    help="Choose the encoder for peptides. Can be `sparse` (onehot) or `blosum`. Default blosum.",
    choices=["blosum", "sparse"],
    default="blosum"
)
arg_parser.add_argument("--normalize-function", "-n",
    help="The normalizing method used for ba_values transformation. Either sigmoid or li.",
    choices=["sigmoid", "li", "custom"],
    default="sigmoid"
)
arg_parser.add_argument("--proportions", "-P",
    help="Percentages for training, test and validation.",
    nargs="+",
    default=["70", "20", "10"],
)
arg_parser.add_argument("--neurons", "-N",
    help="Number of neurons per layer. Default 1000.",
    default=5,
    type=int
)
arg_parser.add_argument("--batch", "-B",
    help="Batch size. Default 64.",
    type=int,
    default=64
)
arg_parser.add_argument("--epochs", "-E",
    help="Number of times to iterate through the whole dataset. Default 150.",
    default=10,
    type=int
)
arg_parser.add_argument("--model-name", "-o",
    help="Name of the model name.",
    required=True
)

a = arg_parser.parse_args()

# MPI INITIALIZATION
#-------------------

mpi_conn = MPI.COMM_WORLD
rank = mpi_conn.Get_rank()
size = mpi_conn.Get_size()
datasets = []

best_model = {
    "validation_rate": 100,
    "model": None,
    "best_epoch": None,
    "test_data": None
}

# onehot (sparse) encoding

# FUNCTIONS AND USEFUL STUFF
#----------------------------

# define the train function
def train_f(dataloader, model, loss_fn, optimizer,device,e):
    model.train()
    # print(f"training epoch {e} on {rank}")
    for X,y in dataloader:
        # forward propagation
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred,torch.reshape(y,(-1,1)).float())

        # backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# define the function used for evaluation
def evaluate(dataloader, model):
    model.eval()
    with torch.no_grad():
        y_vals = torch.tensor([])
        pred_vals = torch.tensor([])
        for X,y in dataloader:
            pred = model(X)
            y_vals = torch.cat((y_vals, y), 0)
            pred_vals = torch.cat((pred_vals, pred), 0)
    return y_vals, pred_vals

# proportions
train_p = int(a.proportions[0])/100
validation_p = int(a.proportions[1])/100
test_p = int(a.proportions[2])/100

# hyperparamaters (all defined as arguments)
neurons_per_layer = a.neurons
batch = a.batch
epochs = a.epochs

# if CUDA cores available, use them and not CPU
device = ("cpu", "cuda")[torch.cuda.is_available()]

# DATA PREPROCESSING
#----------------------------------------------
if rank == 0:
    print("Loading data...")
    csv_peptides, csv_ba_values = load_reg_seq_data(a.csv_file, a.threshold)
    print("Data loaded, splitting into unique test datasets...")

    # SEPARATE TRAIN VALIDATION AND TEST DATASETS
    # -------------------------------------------

    ds_l = len(csv_peptides)# used to retrieve the indices for train and validation without the current test

    test_peptides_indices = list(range(ds_l)) #this will be substracted each iteration and newly created datasets will not have redundant test sets
    num_test_sets = int(100*test_p)


    train_per_dataset = int(train_p*ds_l)
    validation_per_dataset = int(validation_p*ds_l)
    test_per_dataset = int(test_p*ds_l)

    for d in range(num_test_sets):
        dataset = Reg_Seq_Dataset(csv_peptides, csv_ba_values, a.encoder)
        test_indices = random.sample(test_peptides_indices, test_per_dataset)

        ds_indices = list(range(ds_l)) # create the indices range and remove the indices of test
        for i in sorted(test_indices, reverse = True):
            del ds_indices[ds_indices.index(i)] # remove the test indice so it is not used for training and val
            del test_peptides_indices[test_peptides_indices.index(i)] # remove it from the test peptide copy to avoid test redundancies in following datasets

        # create the train and validation indices without the test indices in it:
        train_indices = random.sample(ds_indices, train_per_dataset)
        for i in sorted(train_indices, reverse = True):
            del ds_indices[ds_indices.index(i)] # remove the train indices for no redundancies with validation
        validation_indices = random.sample(ds_indices, validation_per_dataset)
        
        # check for redundancies:
        if len([train for train in train_indices if train in validation_indices]) > 0:
            print("Something went wrong, there is redundancies in the train and validation dataset")
        else:

            training_mean = torch.log(dataset.ba_values[train_indices]).mean()
            training_std = torch.log(dataset.ba_values[train_indices]).std()
            training_min = torch.log(dataset.ba_values[train_indices]).min()
            training_max = torch.log(dataset.ba_values[train_indices]).max()
            
            if a.normalize_function == "sigmoid":
                dataset.ba_values = sig_norm(dataset.ba_values, training_mean, training_std)
            if a.normalize_function == "li":
                dataset.ba_values = li_norm(dataset.ba_values, training_max, training_min)
            if a.normalize_function == "custom":
                dataset.ba_values = custom_norm(dataset.ba_values, training_max, training_min)

            train = Subset(dataset, train_indices)
            validation = Subset(dataset, validation_indices)

            # create the dataloaders iterators:
            train_dataloader = DataLoader(train, batch_size=batch)
            validation_dataloader = DataLoader(validation, batch_size=batch)

            datasets.append({
                "train_dataloader": copy.deepcopy(train_dataloader),
                "validation_dataloader": copy.deepcopy(validation_dataloader),
                "test_indices": test_indices,
                "renorm_values": {
                    "training_max": training_max,
                    "training_min": training_min,
                    "training_mean": training_mean,
                    "training_std": training_std
                },
            })
    # CREATE MULTIPROCESSING
    #-----------------------

# split = mpi_conn.bcast(datasets)[rank]
split = mpi_conn.scatter(datasets)
if rank==0:
    print(f"{num_test_sets} datasets created, models dispatched, starting training...")
# that's the whole multiprocessing code for slaves
train_dataloader = split["train_dataloader"]
validation_dataloader = split["validation_dataloader"]
# TRAIN THE MODEL
#----------------

model = MlpRegBaseline(outputs=1, neurons_per_layer= neurons_per_layer).to(device)

loss_fn = nn.BCELoss()        
optimizer = torch.optim.Adam(model.parameters())

train_losses = []
validation_losses = []

for e in range(epochs):
    # calculate train loss:
    train_y_ba, train_pred_ba = evaluate(train_dataloader, model)
    train_loss = torch.absolute(train_y_ba - torch.reshape(train_pred_ba, (-1,)))
    train_loss_mean = train_loss.mean()
    train_losses.append(train_loss_mean)

    validation_y_ba, validation_pred_ba = evaluate(validation_dataloader, model)
    validation_loss = torch.absolute(validation_y_ba - torch.reshape(validation_pred_ba, (-1,)))
    validation_loss_mean = validation_loss.mean()
    validation_losses.append(validation_loss_mean)

    # update the best model if validation loss improves
    if (validation_loss_mean < best_model["validation_rate"]):
        best_model["model"] = copy.deepcopy(model)
        best_model["validation_rate"] = validation_loss_mean
        best_model["best_epoch"] = e

    # train the model
    train_f(train_dataloader, model, loss_fn, optimizer, device,e)
print(f"Training on {rank} finished.")

# save the model:
best_model["train_losses"] = train_losses
best_model["validation_losses"] = validation_losses
best_model["model"] = best_model["model"].state_dict()
best_model["test_indices"] = split["test_indices"]
best_model["renorm_values"] = split["renorm_values"]

# GATHER THE DATA
#--------------
models = mpi_conn.gather(best_model, root=0)

if rank == 0:
    model_path = f"trained_models/mlp_reg_baseline_{a.neurons}_neurons_{a.epochs}_epochs_{a.model_name}.pt"
    to_save = {
        "arguments": a,
        "models_data": models
    }
    torch.save(to_save, model_path)
    print(f"trained models for `{a.model_name}` gathered, saved at: {model_path}")