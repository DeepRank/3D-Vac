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
from MLP_data import Class_Seq_Dataset, create_unique_csv # class and function to generate shuffled dataset
# import multiprocessing as mp
#from mpi4py import MPI
from sklearn.model_selection import StratifiedKFold, KFold # used for normal cross validation
from sklearn.model_selection import train_test_split
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from torchsummary import summary
import numpy as np
import pickle
import time
import pandas as pd
import random

from sklearn import metrics

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
    default=False, #"/home/daqop/mountpoint_snellius/3D-Vac/data/external/processed/all_hla_j_4.csv"
)
arg_parser.add_argument("--test-csv",
    help="Path to csv containing test cases.",
    default=False
)
arg_parser.add_argument("--train-csv",
    help="Path to csv containing training cases.",
    default=False
)
arg_parser.add_argument("--valid-csv",
    help="Path to csv containing validation cases.",
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
arg_parser.add_argument("--experiment", 
    help="Experiment name to use for output files.",
    required=True
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
arg_parser.add_argument("--clean",
    help="If True, removes intermediate csv file",
    default=False,
    action="store_true"
)

a = arg_parser.parse_args()


# FUNCTIONS AND USEFUL STUFF
#----------------------------

def from_ic50(ic50, max_ic50=50000.0):
    """ Originally from mhcflurry2.0 regression_target module
    Convert ic50s to regression targets in the range [0.0, 1.0].
    
    Parameters
    ----------
    ic50 : numpy.array of float

    Returns
    -------
    numpy.array of float

    """
    x = 1.0 - (np.log(np.maximum(ic50, 1e-12)) / np.log(max_ic50))
    return np.minimum(
        1.0,
        np.maximum(0.0, x))


def to_ic50(x, max_ic50=50000.0):
    """ Originally from mhcflurry2.0 regression_target module
    Convert regression targets in the range [0.0, 1.0] to ic50s in the range
    [0, 50000.0].
    
    Parameters
    ----------
    x : numpy.array of float

    Returns
    -------
    numpy.array of float
    """
    return max_ic50 ** (1.0 - x)

# define the train function
def train_f(dataloader, model, loss_fn, optimizer, device):
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
        X = X.to(device)
        y = y.to(device)
        # forward propagation
        pred = model(X)
        loss = loss_fn(pred, y)

        # backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# define the function used for evaluation
def evaluate(dataloader, model, loss_fn, device, task="classification"):
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
        task (string): Evaluate either on classification or regression

    Returns:
        _type_: _description_
    """
    logits = torch.empty(0).to(device)
    #logits = []
    pred = []
    targets = torch.empty(0).to(device)
    tpr = [] # sensitivity
    tnr = [] # specificity
    accuracies = []
    losses = []
    model.eval()
    with torch.no_grad():
        for X,y in dataloader:
            X = X.to(device)
            y = y.to(device)
            mb_logits = model(X)
            #logits.extend(mb_logits)
            logits = torch.cat((logits, mb_logits), dim=0)
            targets = torch.cat((targets, y), dim=0)
            #targets.extend([float(i) for i in y.tolist()])
            loss = loss_fn(mb_logits, y)
            losses.append(loss.float())

            if task == "regression":
                ic50_logits = to_ic50(mb_logits)
                ic50_y = to_ic50(y)
                pred = torch.tensor([int(val < 500) for val in ic50_logits], dtype=torch.float32).to(device)
                y = torch.tensor([int(val < 500) for val in ic50_y], dtype=torch.int32).to(device)
            else:
                pred = mb_logits.max(1)[1]

            confusion = pred/y # absolute values for metrics
            tot = y.shape[0] # total number of prediction
            pos = float((y == 1.).sum()) # total number of positives (truth 1)
            neg = float((y == 0.).sum()) # total number of negatives (truth 0)

            tpr.append( float((confusion == 1.).sum()/pos) ) # true positive rate = prediction 1/truth 1
            tnr.append( float(torch.isnan(confusion).sum()/neg) ) # true negative rate = predicted 0/truth 0
            accuracies.append(float((pred==y).sum()/tot))
    
    if task == 'classification':
        auc = roc_auc_score(targets, logits.cpu())

    tpr = torch.tensor(tpr, device=device)
    tnr = torch.tensor(tnr, device=device)
    accuracies = torch.tensor(accuracies, device=device)
    losses = torch.tensor(losses, device=device)

    return accuracies.mean(), tpr.mean(), tnr.mean(), losses.mean()#, auc

class MlpRegBaseline(nn.Module):
    def __init__(self, input_shape = (21,82), neurons_per_layer=1000, n_output_nodes=1 ):
        """This class is the architecture of the multi-layer perceptron used both
        for classification and regression.
        Used by CNN/I/regression/seq/mlp_baseline.py and CNN/I/classification/seq/mlp_baseline.py.

        Args:
            input (int, optional): Dimensions of the flattened matrix representing the peptide. Defaults to 20*9.
            neurons_per_layer (int, optional): Neurons per layer. Defaults to 1000.
            n_output_nodes (int, optional): Number of output neurons. Defaults to 1.
        """

        super(MlpRegBaseline,self).__init__()

        # self.conv_layer = nn.Sequential(
        #     nn.BatchNorm1d(input_shape[0]),

        #     nn.Conv1d(
        #         in_channels = input_shape[0], #82
        #         out_channels = 200, 
        #         kernel_size = 1,
        #     ),
        #     nn.ReLU(),
        #     nn.MaxPool1d(
        #         kernel_size = 2,
        #         padding= 2,
        #     )

        # )

        # # input size for the linear layer:
        # self.flatten_shape = self.conv_layer(torch.randn(input_shape).unsqueeze(0)).flatten().shape[0] 
        self.n_output_nodes = n_output_nodes
        self.flatten_shape = input_shape[0] * input_shape[1]

        self.linear = nn.Sequential(
            # input layer
            nn.Flatten(),
            nn.BatchNorm1d(self.flatten_shape),

            # hidden layers
            nn.Linear(self.flatten_shape, neurons_per_layer),
            nn.ReLU(),
            nn.Dropout(),

            nn.Linear(neurons_per_layer, neurons_per_layer),
            nn.ReLU(),
            nn.Dropout(),

            nn.Linear(neurons_per_layer, neurons_per_layer),
            nn.ReLU(),
            nn.Dropout(),

            # output layer
            nn.Linear(neurons_per_layer, n_output_nodes),
            # if using BCE:
        )
    def forward(self,x):
        #torch.Size([12509, 21, 82])
        # conv_out = self.conv_layer(x)
        x = self.linear(x)
        if self.n_output_nodes == 1:
            #s = nn.Sigmoid()
            x = torch.sigmoid(x)
            x = x.squeeze(1)
        return x

# MPI INITIALIZATION
#-------------------

# mpi_conn = MPI.COMM_WORLD
# rank = mpi_conn.Get_rank()
# size = mpi_conn.Get_size()
# if rank != 0:
#     datasets = None
#     dataset = None


OUT_CSV_PATH = '/projects/0/einf2380/data/pop_paper_data/mlp_outputs'
EARLYSTOP_PATIENTCE=10
EARLYSTOP_MAXGAP=0.06
EARLYSTOP_MINEPOCH=50

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
#datasets = []
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device used: {device}")
print("Loading data for splitting...")
if type(a.csv_file)==str:
    csv_path = path.abspath(a.csv_file)
if type(a.test_csv) == str:
    #create a unique csv file, including only peptides up to 15-mers
    csv_path = create_unique_csv(a.train_csv, a.valid_csv, a.test_csv, a.model_name, experiment = a.experiment)

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

# create_unique_csv concatenates the a.csv_path with 
# a.test_csv into one csv. `test` column tells which
# case is test (1) or which is used for train and validation (0)

#test_idx = pd.read_csv(a.test_csv)['ID'].tolist()
test_idx = dataset.df.loc[dataset.df.phase == 'test'].index
    
#train_idx = pd.read_csv(a.train_csv)['ID'].tolist()
train_idx = dataset.df.loc[dataset.df.phase == 'train'].index
#validation_idx = pd.read_csv(a.valid_csv)['ID'].tolist()
validation_idx = dataset.df.loc[dataset.df.phase == 'valid'].index

# CREATE MULTIPROCESSING
#-----------------------
# split = mpi_conn.scatter(datasets)  # master sending tasks
# dataset = mpi_conn.bcast(dataset) # master broadcasting the loaded dataset
# # slaves receiving work
train_subset = Subset(dataset, train_idx)
validation_subset = Subset(dataset, validation_idx)
test_subset = Subset(dataset, test_idx)

train_dataloader = DataLoader(train_subset, batch_size=batch)
validation_dataloader = DataLoader(validation_subset, batch_size=batch)
test_dataloader = DataLoader(test_subset, batch_size=batch)

# TRAIN THE MODEL
#----------------

# instantiate the model class
input_dimensions = dataset.input_shape
model = MlpRegBaseline(n_output_nodes=(1,2)[a.task == "classification"], neurons_per_layer= neurons_per_layer, input_shape=input_dimensions).to(device)
optimizer = torch.optim.Adam(model.parameters())

print(f"Number of samples for training, validation and test:")
print(f" {len(train_idx)}, {len(validation_idx)}, {len(test_idx)}")
print(f"Starting training using this architecture:")
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

earlystop_counter = 0

start_training_time = time.time()
for e in range(epochs):
    print(f'Epoch {e}')
    train_f(train_dataloader, model, loss_fn, optimizer, device)
    # calculate metrics:
    tr_accuracy, tr_tpr, tr_tnr, tr_losses = evaluate(train_dataloader, model, loss_fn, device, a.task)
    print(f'Train Acc: {tr_accuracy:.4f}, Train Loss: {tr_losses:.4f},')
    train_accuracies.append(tr_accuracy)
    train_tpr.append(tr_tpr)
    train_tnr.append(tr_tnr)
    train_losses.append(tr_losses)
    #train_auc.append(tr_auc)
    
    val_accuracy, val_tpr, val_tnr, val_losses  = evaluate(validation_dataloader, model, loss_fn, device, a.task)
    print(f'Val Acc:   {val_accuracy:.4f}, Val Loss:   {val_losses:.4f},')
    validation_accuracies.append(val_accuracy)
    validation_tpr.append(val_tpr)
    validation_tnr.append(val_tnr)
    validation_losses.append(val_losses)
    #validation_auc.append(val_auc)

    t_accuracy, t_tpr, t_tnr, t_losses = evaluate(test_dataloader, model, loss_fn, device, a.task)
    print(f'Test Acc:  {t_accuracy:.4f}, Test Loss:  {t_losses:.4f},')
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
    
    # Early stopping
    best_loss = min(validation_losses)
    
    if best_loss is None:
        best_loss = val_losses
        best_epoch = epoch
    elif val_losses < (best_loss - EARLYSTOP_MAXGAP):
        best_loss = val_losses
        best_epoch = epoch
        earlystop_counter = 0
    else:
        earlystop_counter += 1
        if earlystop_counter >= EARLYSTOP_PATIENTCE:
            early_stop = True

    if e <= EARLYSTOP_MINEPOCH:
        early_stop = False
        
    if early_stop:
        print(f"Early stop at n epochs: {e}\n Validation loss has not improved for {EARLYSTOP_PATIENTCE} epochs")
        break
        
    
    
end_training_time = time.time()
training_time = (end_training_time - start_training_time)/60
print(f"Training finished. Time for training: {training_time} minutes")

print('Generate final test outputs')
logits = torch.empty(0).to(device)
targets = torch.empty(0).to(device)
with torch.no_grad():
    for X,y in test_dataloader:
        X = X.to(device)
        y = y.to(device)
        mb_logits = model(X)
        #logits.extend(mb_logits)
        logits = torch.cat((logits, mb_logits), dim=0)
        targets = torch.cat((targets, y), dim=0)
        #targets.extend([float(x) for x in y.tolist()])
    if a.task == 'regression':
        #ic50_logits = to_ic50(logits)
        ic50_y = to_ic50(targets)
        #logits = torch.tensor([int(val < 500) for val in ic50_logits], dtype=torch.float32).to(device)
        targets = torch.tensor([int(val < 500) for val in ic50_y], dtype=torch.int32).to(device)

IDs = pd.read_csv(csv_path)
#targets = targets.to('cpu').tolist()
targets = [float(x) for x in targets.to('cpu')]
if IDs.loc[test_idx]['label'].tolist() != targets:
    #raise Exception('Test labels and model targets do not coincide')
    print('WARNING: Test labels and model targets do not coincide')
    for i in range(1, 10012):
        if IDs.loc[test_idx]['label'].tolist()[i-1] != targets[i-1]:
            print(f'Index: {i-1}')
            print(f'ID: {IDs.loc[test_idx[i-1]]["ID"]}')
            print(f'BA: {IDs.loc[test_idx[i-1]]["measurement_value"]}')
            print(f' Targets value: {targets[i-1]}')

best_model['final_outputs'] = pd.DataFrame({'ID' :IDs.loc[test_idx]['ID'].tolist(), 
                                'OUTPUT_1': logits.to('cpu').tolist(), 
                                'Target' : IDs.loc[test_idx]['label'].tolist()})

if a.clean:
    os.remove(csv_path)
    
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
best_model["test_indices"] = test_idx

# GATHER THE DATA
#--------------
# models = mpi_conn.gather(best_model, root=0) # master receiving trained models
# all_training_times = mpi_conn.gather(training_time)

trained_models_path = a.trained_models_path
model_path = f"{trained_models_path}/mlp_{a.task}_{a.experiment}.pt"
to_save = {
    "arguments": a,
    "best_model": best_model
}
torch.save(to_save, model_path)
print(f"trained models for `{a.model_name}` gathered, saved at: {model_path}")


if a.experiment.split('_')[0] == 'Shuffled':
    exp_folder = 'shuffled_crossval'
elif a.experiment.split('_')[0] == 'AlleleClustered':
    exp_folder = 'allele_crossval'
else:
    exp_folder = ''

out_csv = f'{OUT_CSV_PATH}/{exp_folder}/{a.experiment}.csv'
best_model['final_outputs'].to_csv(out_csv, index=False)
print(f'Output collected in {out_csv}')