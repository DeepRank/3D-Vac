import torch
import argparse
from torch import nn
from torch.utils.data import Subset
from torch.utils.data import DataLoader
import copy
import os.path as path
import os
import sys
sys.path.append(path.abspath("."))
sys.path.append(path.abspath("../"))
sys.path.append(path.abspath("../../"))
sys.path.append(path.abspath("../../../../"))
from seq.SeqBased_models import MlpRegBaseline, train_f, evaluate
from seq.datasets import Class_Seq_Dataset, create_unique_csv # class and function to generate shuffled dataset
# import multiprocessing as mp
from mpi4py import MPI
from sklearn.model_selection import StratifiedKFold, KFold # used for normal cross validation
from sklearn.model_selection import train_test_split
from sklearn.model_selection import LeaveOneGroupOut
from torchsummary import summary
import numpy as np
import pickle
import time
import pandas as pd
import random

# DEFINE CLI ARGUMENTS
#---------------------

arg_parser = argparse.ArgumentParser(
    description="""Fully connected layer to generate a model which predicts binders based on different type of encoded
    peptide and allele sequence. Works for peptide length not greater than 15. Takes as input the csv file (header free)
    containing the list of peptides, the column for the peptides and a threshold for binding affinity to define binders.
    In case where the encoder includes allele as well, the CSV file should contain a "pseudoseq" column of the allele.
    Can be spawned for shuffled training, clustered and clustered with specific clusters used for training and test
    (the validation set is a subset of training). In that case, the number of sbatch tasks should be set to the number of
    unique train clusters representing the number of folds. If several clusters should be used for the same fold, make sure 
    to regroup those clusters into one beforehand.
    """
)
arg_parser.add_argument("--allele-to-pseudosequence-csv",
    help="Path to allele containing the mhcflurry allele to pseudosequence mapping. Default /projects/0/einf2380/data/external/unprocessed/mhcflurry.allele_sequences.csv",
    default="/projects/0/einf2380/data/external/unprocessed/mhcflurry.allele_sequences.csv"
)
arg_parser.add_argument("--csv-file", "-f",
    help="Name of the csv file in data/external/processed containing the cluster column. \n \
        Works as a train and validation set if provided with --test-csv.",
    default="/home/daqop/mountpoint_snellius/3D-Vac/data/external/processed/all_hla_j_4.csv"
)
arg_parser.add_argument("--test-csv",
    help="Path to csv containing test cases.",
    default=False
)
arg_parser.add_argument("--trained-models-path", "-p",
    help="""
    Absolute path to the folder where saved models and associated data are stored.
    Default: /home/lepikhovd/3D-Vac/src/5_train_models/CNN/I/classification/seq/trained_models
    """,
    default="/home/lepikhovd/3D-Vac/src/5_train_models/CNN/I/classification/seq/trained_models"
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
    help="Column in the csv file listing clusters to filter, default to `cluster`.",
    default="cluster"
)
arg_parser.add_argument("--encoder", "-e",
    help="Choose the encoder for peptides. Can be `sparse` (onehot) or `blosum`. Default blosum_with_allele.",
    choices=["blosum", "sparse", "blosum_with_allele", "sparse_with_allele"],
    default="blosum_with_allele"
)
arg_parser.add_argument("--neurons", "-N",
    help="Number of neurons per layer. Default 5.",
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
arg_parser.add_argument("--task",
    help="Use for classification or regression. Default classification",
    choices=["classification", "regression"],
    default="classification"
)
arg_parser.add_argument("--no-clean",
    help="If True, doesn't remove intermediate csv file",
    default=False,
    action="store_true"
)

a = arg_parser.parse_args()

# MPI INITIALIZATION
#-------------------

mpi_conn = MPI.COMM_WORLD
rank = mpi_conn.Get_rank()
size = mpi_conn.Get_size()
if rank != 0:
    datasets = None
    dataset = None

# This will be saved in a pickle file as the best model for each cross validation (each MPI job)
best_model = {
    "validation_rate": 0,
    "validation_loss": 0, #in case of regression
    "model": None,
    "best_epoch": None,
    "test_data": None
}


# hyperparamaters (all defined as arguments)
neurons_per_layer = a.neurons
batch = a.batch
epochs = a.epochs
loss_fn = (nn.BCELoss(), nn.CrossEntropyLoss())[a.task == "classification"]

# if CUDA cores available, use them and not CPU
device = ("cpu", torch.device('cuda'))[torch.cuda.is_available()]

# DATA PREPROCESSING
#----------------------------------------------
if rank == 0:
    datasets = []
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Device used: {device}")
    print("Loading data for splitting...")
    csv_path = path.abspath(a.csv_file)
    if type(a.test_csv) == str:
        #create a unique csv file
        csv_path = create_unique_csv(a.csv_file, a.test_csv, a.model_name)

    dataset = Class_Seq_Dataset(
        csv_path,
        a.encoder,
        device,
        threshold = a.threshold,
        cluster_column = (None,a.cluster_column)[a.cluster],
        task=a.task,
        allele_to_pseudosequence_csv_path=a.allele_to_pseudosequence_csv
    )
    ds_l = dataset.peptides.shape[0]
    print("Data loaded, splitting into unique test datasets...")
    
    if type(a.test_csv) == str:
        # create_unique_csv concatenates the a.csv_path with 
        # a.test_csv into one csv. `test` column tells which
        # case is test (1) or which is used for train and validation (0)
        train_val_idx = dataset.df.loc[dataset.df.test == 0].index
        test_idx = dataset.df.loc[dataset.df.test == 1].index
        if a.task == "classification":
            train_idx,validation_idx = train_test_split(train_val_idx, test_size=2/9, stratify=dataset.labels[train_val_idx]) #2/9*0,9=0.2
        else:
            train_idx,validation_idx = train_test_split(train_val_idx, test_size=2/9) #2/9*0,9=0.2
        
        datasets = [[
            train_idx,
            validation_idx,
            test_idx
        ]]*size

        if not a.no_clean:
            os.remove(csv_path)
        

    # SEPARATE TRAIN VALIDATION AND TEST DATASETS
    # -------------------------------------------
    elif a.cluster == False:
        print("Splitting into shuffled datasets..")
        kfold = (KFold(n_splits=size, shuffle=True), StratifiedKFold(n_splits=size, shuffle=True))[a.task == "classification"]
        # split the whole dataset using sklearn.metrics kfold function:
        for train_idx, test_idx in kfold.split(dataset.peptides, dataset.labels):
            # the train dataset is further splited into validation with scaled proportion:
            if a.task == "classification":
                train_idx,validation_idx = train_test_split(train_idx, test_size=2/9, stratify=dataset.labels[train_idx]) #2/9*0,9=0.2
            else:
                train_idx,validation_idx = train_test_split(train_idx, test_size=2/9) #2/9*0,9=0.2

            datasets.append([
                train_idx,
                validation_idx,
                test_idx,
            ])
    else: # perform clustered dataset split
        if a.train_clusters != None:
            # add a 0 group containing trash cluster:
            a.train_clusters.append(0) # the trash cluster is used only for training, always when train_clusters is provided
            print(f"Splitting into provided clusters {size} times..")
            print(f"Clusters used for train and validation (shuffled and split 80%-20%): {a.train_clusters}")
            print(f"Clusters used for test: {a.test_clusters}")
            # here the validation is a subset of train while test is a unique subset
            val_train_idx = torch.tensor([j for j,g in enumerate(dataset.groups) if g in a.train_clusters], dtype=torch.long)

            test_idx = [j for j,g in enumerate(dataset.groups) if g in a.test_clusters]
            train_idx, val_idx = train_test_split(val_train_idx, test_size=0.2, stratify=dataset.labels[val_train_idx])

            datasets = [[
                train_idx.tolist(),
                val_idx.tolist(),
                test_idx
            ]]*size
        else:
            print("Splitting into clustered datasets..")
            kfold = LeaveOneGroupOut()       
            # same as shuffled but using sklearn.metrics leavonegroupout function
            for train_idx, test_idx in kfold.split(dataset.peptides, dataset.labels, dataset.groups):
                # if 0 in dataset.groups[test_idx]:
                #     continue
                # here the test is from groups not in train,
                # the validation is comming from the same group as train.
                if a.task == "classification":
                    train_idx,validation_idx = train_test_split(train_idx, test_size=2/9, stratify=dataset.labels[train_idx]) #2/9*0,9=0.2
                else:
                    train_idx,validation_idx = train_test_split(train_idx, test_size=2/9) #2/9*0,9=0.2

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
input_dimensions = dataset.input_shape
model = MlpRegBaseline(n_output_nodes=(1,2)[a.task == "classification"], neurons_per_layer= neurons_per_layer, input_shape=input_dimensions).to(device)
optimizer = torch.optim.Adam(model.parameters())

if rank==0:
    print(f"Number of samples for training, validation and test for the {len(datasets)} folds:")
    for i, ds in enumerate(datasets):
        print(f"Fold {i}: {len(ds[0])}, {len(ds[1])}, {len(ds[2])}")
    print(f"{size} datasets created, models loaded, starting training using this architecture:")
    summary(
        model, 
        input_dimensions,
        # col_names = ("output_size", "num_params")
    )


train_accuracies, validation_accuracies, test_accuracies = [], [], []
train_losses, validation_losses, test_losses = [], [], []
train_tpr, validation_tpr, test_tpr = [], [], []
train_tnr, validation_tnr, test_tnr = [], [], []
#train_auc, validation_auc, test_auc = [], [], []

start_training_time = time.time()
for e in range(epochs):
    # calculate metrics:
    tr_accuracy, tr_tpr, tr_tnr, tr_losses = evaluate(train_dataloader, model, loss_fn, device, a.task)
    train_accuracies.append(tr_accuracy)
    train_tpr.append(tr_tpr)
    train_tnr.append(tr_tnr)
    train_losses.append(tr_losses) 
    #train_auc.append(tr_auc)
    
    val_accuracy, val_tpr, val_tnr, val_losses  = evaluate(validation_dataloader, model, loss_fn, device, a.task)
    validation_accuracies.append(val_accuracy)
    validation_tpr.append(val_tpr)
    validation_tnr.append(val_tnr)
    validation_losses.append(val_losses)
    #validation_auc.append(val_auc)

    t_accuracy, t_tpr, t_tnr, t_losses = evaluate(test_dataloader, model, loss_fn, device, a.task)
    test_accuracies.append(t_accuracy)
    test_tpr.append(t_tpr)
    test_tnr.append(t_tnr)
    test_losses.append(t_losses)
    #test_auc.append(t_auc)

    # update the best model if validation loss improves
    if a.task == "classification":
        if (val_accuracy > best_model["validation_rate"]):
            best_model["model"] = copy.deepcopy(model)
            best_model["validation_rate"] = val_accuracy
            best_model["best_epoch"] = e
    elif e == 0:
        best_model["validation_loss"] = val_losses
    else:
        if (val_losses < best_model["validation_loss"]):
            best_model["model"] = copy.deepcopy(model)
            best_model["validation_loss"] = val_losses
            best_model["best_epoch"] = e
    # train the model over one epoch
    train_f(train_dataloader, model, loss_fn, optimizer, device)
end_training_time = time.time()
training_time = (end_training_time - start_training_time)/60
print(f"Training on {rank} finished. Time for training: {training_time} minutes")

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

# best_model["train_auc"] = train_auc
# best_model["validation_auc"] = validation_auc
# best_model["test_auc"] = test_auc

best_model["model"] = best_model["model"].state_dict()
best_model["test_indices"] = split[2]

# GATHER THE DATA
#--------------
models = mpi_conn.gather(best_model, root=0) # master receiving trained models
all_training_times = mpi_conn.gather(training_time)

if rank == 0:
    trained_models_path = a.trained_models_path
    model_path = f"{trained_models_path}/mlp_classification_{a.encoder}_encoder_{a.neurons}_neurons_{a.epochs}_epochs_{a.model_name}_{a.batch}_batch_size.pt"
    to_save = {
        "arguments": a,
        "models_data": models
    }
    torch.save(to_save, model_path)
    print(f"trained models for `{a.model_name}` gathered, saved at: {model_path}")