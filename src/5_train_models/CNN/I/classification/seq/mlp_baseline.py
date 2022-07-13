import torch
import argparse
from torch import nn
from torch.utils.data import Subset
from torch.utils.data import DataLoader
import copy
import os.path as path
import sys
sys.path.append(path.abspath("../../../../"))
from CNN.models import MlpRegBaseline
from CNN.datasets import Class_Seq_Dataset, load_class_seq_data # class and function to generate shuffled dataset
from CNN.I.classification.seq import data_path # path to the data folder relative to the location of the __init__.py file
# import multiprocessing as mp
from mpi4py import MPI
from sklearn.model_selection import StratifiedKFold # used for normal cross validation
from sklearn.model_selection import train_test_split
from sklearn.model_selection import LeaveOneGroupOut

# DEFINE CLI ARGUMENTS
#---------------------

arg_parser = argparse.ArgumentParser(
    description="Fully connected layer to generate a model which predicts binders based on one-hot encoded \
    peptide sequence. Works only for a fixed length of 9 residues. Takes as input the csv file (header free) containing \
    the list of peptides, the column for the peptides and a threshold for binding affinity to define binders."
)

arg_parser.add_argument("--csv-file", "-f",
    help="Name of the csv file in data/external/processed containing the cluster column. Default BA_pMHCI.csv",
    default="BA_pMHCI.csv"
)
arg_parser.add_argument("--threshold", "-t",
    help="Binding affinity threshold to define binders, default 500.",
    type=float,
    default=500
)
arg_parser.add_argument("--cluster", "-c",
    help="By providing this argument, will perform a scikit LeavOneGroupOut crossvalidation grouped by cluster, shuffled KFold otherwise.",
    default=False,
    action="store_true"
)
arg_parser.add_argument("--encoder", "-e",
    help="Choose the encoder for peptides. Can be `sparse` (onehot) or `blosum`. Default blosum.",
    choices=["blosum", "sparse", "mixed"],
    default="mixed"
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
datasets = [] # this array will be filled with jobs for slaves

# This will be saved in a pickle file as the best model for each cross validation (each MPI job)
best_model = {
    "validation_rate": 0,
    "model": None,
    "best_epoch": None,
    "test_data": None
}

# FUNCTIONS AND USEFUL STUFF
#----------------------------

# define the train function
def train_f(dataloader, model, loss_fn, optimizer):
    """Function used to train the model over the whole training dataset.

    Args:
        dataloader (DataLoader): Instance of the DataLoader object iterating
        over the train dataset.
        model (obj): Model to train.
        loss_fn (obj): Function used to measure the difference between the truth
        and prediction.
        optimizer (obj): Function used to perform backpropagation and update of
        weights.
    """
    model.train()
    # print(f"training epoch {e} on {rank}")
    for X,y in dataloader:
        # forward propagation
        pred = model(X)
        loss = loss_fn(pred,y)

        # backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# define the function used for evaluation
def evaluate(dataloader, model, loss_fn, device):
    """### CALCULATE SENSITIVITY, SPECIFICITY AND OTHER METRICS ###
    Calculate the confusion tensor (AKA matching matrix) by dividing predictions
    with true values.
    To assess performances, first calculate absolute values and adjust it to 
    the total number of expected postives, negatives:

    Args:
        dataloader (DataLoader): Dataloader containing the iterator for the 
        dataset we want to evaluate.
        model (obj): Model being evaluated.
        loss_fn (obj): Function used to measure the difference between the truth
        and prediction.
        device (string): "cpu" or torch.device("cuda")

    Returns:
        _type_: _description_
    """
    tpr = [] # sensitivity
    tnr = [] # specificity
    accuracies = []
    losses = []
    model.eval()
    with torch.no_grad():
        for X,y in dataloader:
            logits = model(X)
            pred = logits.max(1)[1]

            confusion = pred/y # absolute values for metrics
            tot = y.shape[0] # total number of prediction
            pos = float((y == 1.).sum()) # total number of positives (truth 1)
            neg = float((y == 0.).sum()) # total number of negatives (truth 0)

            loss = loss_fn(logits, y)

            tpr.append( float((confusion == 1.).sum()/pos) ) # true positive rate = prediction 1/truth 1
            tnr.append( float(torch.isnan(confusion).sum()/neg) ) # true negative rate = predicted 0/truth 0
            accuracies.append(float((pred==y).sum()/tot))
            losses.append(loss.float())

    tpr = torch.tensor(tpr, device=device)
    tnr = torch.tensor(tnr, device=device)
    accuracies = torch.tensor(accuracies, device=device)
    losses = torch.tensor(losses, device=device)

    return accuracies.mean(), tpr.mean(), tnr.mean(), losses.mean()

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
    print("Loading data...")
    csv_path = path.abspath(f"{data_path}external/processed/{a.csv_file}")
    csv_peptides, csv_labels, groups = load_class_seq_data(csv_path, a.threshold)
    dataset = Class_Seq_Dataset(csv_peptides, csv_labels, a.encoder, device)
    ds_l = dataset.peptides.shape[0]
    print("Data loaded, splitting into unique test datasets...")

    # SEPARATE TRAIN VALIDATION AND TEST DATASETS
    # -------------------------------------------
    train_means = []
    validation_means = []
    test_means = []
    if a.cluster == False:
        print("Splitting into shuffled datasets..")
        kfold = StratifiedKFold(n_splits=10, shuffle=True)
        datasets = []
        # split the whole dataset using sklearn.metrics kfold function:
        for train_idx, test_idx in kfold.split(dataset.peptides, dataset.labels):
            # the train dataset is further splited into validation with scaled proportion:
            train_idx,validation_idx = train_test_split(train_idx, test_size=2/9, stratify=dataset.labels[train_idx]) #2/9*0,9=0.2
            train_subset = Subset(dataset, train_idx) 
            validation_subset = Subset(dataset, validation_idx) 
            test_subset = Subset(dataset, test_idx)

            # create the dataloaders to dispatch on slaves:
            train_dataloader = DataLoader(train_subset, batch_size=batch)
            validation_dataloader = DataLoader(validation_subset, batch_size=batch)
            test_dataloader = DataLoader(test_subset, batch_size=batch)

            datasets.append({
                "train_dataloader": train_dataloader,
                "validation_dataloader": validation_dataloader,
                "test_dataloader": test_dataloader,
                "test_indices": test_idx
            })
    
    else: # perform clustered dataset split
        print("Splitting into clustered datasets")
        kfold = LeaveOneGroupOut()       
        # same as shuffled but using sklearn.metrics leavonegroupout function
        for train_idx, test_idx in kfold.split(dataset.peptides, dataset.labels, groups):
            # here the test is from groups not in train,
            # the validation is comming from the same group as train.
            train_idx,validation_idx = train_test_split(train_idx, test_size=2/9, stratify=dataset.labels[train_idx]) #2/9*0,9=0.2

            train_subset = Subset(dataset, train_idx) 
            validation_subset = Subset(dataset, validation_idx) 
            test_subset = Subset(dataset, test_idx)

            train_dataloader = DataLoader(train_subset, batch_size=batch)
            validation_dataloader = DataLoader(validation_subset, batch_size=batch)
            test_dataloader = DataLoader(test_subset, batch_size=batch)

            datasets.append({
                "train_dataloader": train_dataloader,
                "validation_dataloader": validation_dataloader,
                "test_indices": test_idx,
                "test_dataloader": test_dataloader
            })

# CREATE MULTIPROCESSING
#-----------------------
split = mpi_conn.scatter(datasets)  # master sending tasks

if rank==0:
    print(f"10 datasets created, models dispatched, starting training...")

# slaves receiving work
train_dataloader = split["train_dataloader"]
validation_dataloader = split["validation_dataloader"]
test_dataloader = split["test_dataloader"]

# TRAIN THE MODEL
#----------------

# instantiate the model class
input_dimensions = (20*9, 40*9)[a.encoder == "mixed"]
model = MlpRegBaseline(outputs=2, neurons_per_layer= neurons_per_layer, input=input_dimensions).to(device)

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
best_model["test_indices"] = split["test_indices"]

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