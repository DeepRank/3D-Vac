import torch
import argparse
from torch import nn
from torch.utils.data import Subset
from torch.utils.data import DataLoader
import copy
import os.path as path
import sys
sys.path.append(path.abspath("../../../../"))
from CNN.SeqBased_models import MlpRegBaseline, train_f, evaluate
from CNN.datasets import Class_Seq_Dataset # class and function to generate shuffled dataset
# import multiprocessing as mp
from mpi4py import MPI
from sklearn.model_selection import StratifiedKFold # used for normal cross validation
from sklearn.model_selection import train_test_split
from sklearn.model_selection import LeaveOneGroupOut
from torchsummary import summary
import numpy as np

# DEFINE CLI ARGUMENTS
#---------------------

arg_parser = argparse.ArgumentParser(
    description="""Fully connected layer to generate a model which predicts binders based on one-hot encoded
    peptide sequence. Works only for a fixed length of 9 residues. Takes as input the csv file (header free) containing
    the list of peptides, the column for the peptides and a threshold for binding affinity to define binders.
    In case where the encoder is sparse_with_allele, the path to the pseudoseq csv file (allele to seq) should be provided
    with the --pseudoseq-csv-path argument.
    """
)
arg_parser.add_argument("--csv-file", "-f",
    help="Name of the csv file in data/external/processed containing the cluster column. Default BA_pMHCI.csv",
    default="/home/daqop/mountpoint_snellius/3D-Vac/data/external/processed/all_hla_j_4.csv"
)
arg_parser.add_argument("--threshold", "-t",
    help="Binding affinity threshold to define binders, default 500.",
    type=float,
    default=500
)
arg_parser.add_argument("--cluster", "-c",
    help="""
    By providing this argument, will perform a scikit LeavOneGroupOut crossvalidation grouped by cluster,
    shuffled KFold otherwise. If --train-clusters and --test-clusters are provided, will use those instead.
    """,
    default=False,
    action="store_true"
)
arg_parser.add_argument("--train-clusters", "-T",
    help="List of clusters used for training",
    nargs="+",
    type=int
)
arg_parser.add_argument("--test-clusters", "-V",
    help="List of clusters used for testing",
    nargs="+",
    type=int
)
arg_parser.add_argument("--cluster-column", "-C",
    help="Column in the csv file for the cluster set, default to `cluster`.",
    default="cluster"
)
arg_parser.add_argument("--encoder", "-e",
    help="Choose the encoder for peptides. Can be `sparse` (onehot) or `blosum`. Default blosum.",
    choices=["blosum", "sparse", "mixed", "sparse_with_allele"],
    default="sparse"
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
dataset = None # this object containing the pytorch loaded dataset will be broadcasted to all slaves
datasets = [] # this array will be filled with jobs for slaves

# This will be saved in a pickle file as the best model for each cross validation (each MPI job)
best_model = {
    "validation_rate": 0,
    "model": None,
    "best_epoch": None,
    "test_data": None
}


# hyperparamaters (all defined as arguments)
neurons_per_layer = a.neurons
batch = a.batch
epochs = a.epochs

# if CUDA cores available, use them and not CPU
device = ("cpu", torch.device('cuda:0'))[torch.cuda.is_available()]

# DATA PREPROCESSING
#----------------------------------------------
if rank == 0:
    print(f"Device used: {device}")
    print("Loading data for splitting...")
    csv_path = path.abspath(a.csv_file)
    dataset = Class_Seq_Dataset(
        csv_path,
        a.encoder,
        device,
        threshold = a.threshold,
        cluster_column = (None,a.cluster_column)[a.cluster],
    )
    ds_l = dataset.peptides.shape[0]
    print("Data loaded, splitting into unique test datasets...")

    # SEPARATE TRAIN VALIDATION AND TEST DATASETS
    # -------------------------------------------
    if a.cluster == False:
        print("Splitting into shuffled datasets..")
        kfold = StratifiedKFold(n_splits=10, shuffle=True)
        # split the whole dataset using sklearn.metrics kfold function:
        for train_idx, test_idx in kfold.split(dataset.peptides, dataset.labels):
            # the train dataset is further splited into validation with scaled proportion:
            train_idx,validation_idx = train_test_split(train_idx, test_size=2/9, stratify=dataset.labels[train_idx]) #2/9*0,9=0.2

            datasets.append([
                train_idx,
                validation_idx,
                test_idx,
            ])
    
    else: # perform clustered dataset split
        if len(a.train_clusters) > 0:
            print("Splitting into provided clusters..")
            for i in range(len(a.train_clusters)):
                train_idx = torch.tensor([j for j,g in enumerate(dataset.groups) if g == a.train_clusters[i]])
                val_test_idx = torch.tensor([j for j,g in enumerate(dataset.groups) if g in a.test_clusters])
                val_idx, test_idx = train_test_split(val_test_idx, test_size=0.5, stratify=dataset.labels[val_test_idx])

                datasets.append([
                    train_idx,
                    val_idx,
                    test_idx
                ])
        else:
            print("Splitting into clustered datasets..")
            kfold = LeaveOneGroupOut()       
            # same as shuffled but using sklearn.metrics leavonegroupout function
            for train_idx, test_idx in kfold.split(dataset.peptides, dataset.labels, dataset.groups):
                # here the test is from groups not in train,
                # the validation is comming from the same group as train.
                train_idx,validation_idx = train_test_split(train_idx, test_size=2/9, stratify=dataset.labels[train_idx]) #2/9*0,9=0.2

                datasets.append([
                    train_idx,
                    validation_idx,
                    test_idx
                ])

# CREATE MULTIPROCESSING
#-----------------------
split = mpi_conn.scatter(datasets)  # master sending tasks
dataset = mpi_conn.bcast(dataset) # master broadcasting the loaded dataset
# slaves receiving work
train_subset = Subset(dataset, split[0])
validation_subset = Subset(dataset, split[1])
test_subset = Subset(dataset, split[2])

train_dataloader = DataLoader(train_subset, batch_size=batch)
validation_dataloader = DataLoader(validation_subset, batch_size=batch)
test_dataloader = DataLoader(test_subset, batch_size=batch)

# TRAIN THE MODEL
#----------------

# instantiate the model class
input_dimensions = dataset.input_size
model = MlpRegBaseline(outputs=2, neurons_per_layer= neurons_per_layer, input=input_dimensions).to(device)

if rank==0:
    print(f"{size} datasets created, models loaded, starting training using this architecture:")
    summary(model, input_size=(input_dimensions, 1, 1))

loss_fn = nn.CrossEntropyLoss()        
optimizer = torch.optim.Adam(model.parameters())

train_accuracies, validation_accuracies, test_accuracies = [], [], []
train_losses, validation_losses, test_losses = [], [], []
train_tpr, validation_tpr, test_tpr = [], [], []
train_tnr, validation_tnr, test_tnr = [], [], []

for e in range(epochs):
    # calculate metrics:
    tr_accuracy, tr_tpr, tr_tnr, tr_losses = evaluate(train_dataloader, model, loss_fn, device)
    train_accuracies.append(tr_accuracy);train_tpr.append(tr_tpr); train_tnr.append(tr_tnr);
    train_losses.append(tr_losses); 
    
    val_accuracy, val_tpr, val_tnr, val_losses = evaluate(validation_dataloader, model, loss_fn, device)
    validation_accuracies.append(val_accuracy);validation_tpr.append(val_tpr)
    validation_tnr.append(val_tnr); validation_losses.append(val_losses);

    t_accuracy, t_tpr, t_tnr, t_losses = evaluate(test_dataloader, model, loss_fn, device)
    test_accuracies.append(t_accuracy);test_tpr.append(t_tpr); test_tnr.append(t_tnr); 
    test_losses.append(t_losses);

    # update the best model if validation loss improves
    if (val_accuracy > best_model["validation_rate"]):
        best_model["model"] = copy.deepcopy(model)
        best_model["validation_rate"] = val_accuracy
        best_model["best_epoch"] = e

    # train the model over one epoch
    train_f(train_dataloader, model, loss_fn, optimizer)
print(f"Training on {rank} finished.")

# save the model:
best_model["train_accuracies"] = train_accuracies
best_model["validation_accuracies"] = validation_accuracies
best_model["test_accuracies"] = test_accuracies

best_model["train_tpr"] = train_tpr
best_model["validation_tpr"] = validation_tpr
best_model["test_tpr"] = test_tpr

best_model["train_tnr"] = train_tnr
best_model["validation_tnr"] = validation_tnr
best_model["test_tnr"] = test_tnr

best_model["train_losses"] = train_losses
best_model["validation_losses"] = validation_losses
best_model["test_losses"] = test_losses

best_model["model"] = best_model["model"].state_dict()
best_model["test_indices"] = split[2]

# GATHER THE DATA
#--------------
models = mpi_conn.gather(best_model, root=0) # master receiving trained models

if rank == 0:
    model_path = f"trained_models/mlp_classification_{a.encoder}_encoder_{a.neurons}_neurons_{a.epochs}_epochs_{a.model_name}_{a.batch}_batch_size.pt"
    to_save = {
        "arguments": a,
        "models_data": models
    }
    torch.save(to_save, model_path)
    print(f"trained models for `{a.model_name}` gathered, saved at: {model_path}")