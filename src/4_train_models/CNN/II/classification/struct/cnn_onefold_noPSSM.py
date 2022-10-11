import argparse
import os.path as path
import os
import sys
sys.path.append(path.abspath("../../../../"))
from CNN.I.classification.seq import data_path # path to the data folder relative to the location of the __init__.py file
from CNN.CNN_models import *
# import multiprocessing as mp
from deeprank.learn import DataSet, NeuralNet
from deeprank.learn.modelGenerator import *

# DEFINE CLI ARGUMENTS
#---------------------

arg_parser = argparse.ArgumentParser(
    description = "Script used to load the train, validation and test HDF5 files generated with 5/CNN/split_train_h5.py \
    and train 10 models using CnnClassificationBaseline architecture from CNN/3D_models.py file. \
    The script should be run using cnn_baseline_cuda.sh or cnn_baseline.sh. \
    The script expects both shuffled and clustered HDF5 splits to be in the {--split-path}/shuffled/<0:9>/ \
    and in the {--split-path}/clustered/<0:9>/ folders (works only for 10 fold xvalidation). By default this script \
    will perform a shuffled cross validation, for a clustered please provide --cluster argument.",
)

# arg_parser.add_argument("--splits-path", "-s",
#     help="Path to shuffled and clustered folders containing subfolders named from 0 to 9 each with \
#     train.hdf5, valid.hdf5 and test.hdf5 files as the splits for each fold. Default path \
#     /projects/0/einf2380/data/pMHCII/features_output_folder/CNN/hla_drb1_0101_15mers/splits",
#     default="/projects/0/einf2380/data/pMHCII/features_output_folder/CNN/hla_drb1_0101_15mers/splits"
# )
arg_parser.add_argument("--cluster", "-c",
    help="By providing this argument, will perform a scikit LeavOneGroupOut crossvalidation grouped by cluster, shuffled KFold otherwise.",
    default=0,
    type=int,
    choices=[0,1]
)
arg_parser.add_argument("--batch", "-B",
    help="Batch size. Default 64.",
    type=int,
    default=64
)
arg_parser.add_argument("--epochs", "-E",
    help="Number of times to iterate through the whole dataset. Default 10.",
    default=10,
    type=int
)
arg_parser.add_argument("--exp-name", "-x",
    help="Name of the folder where 10 subfolders will be created for each cross validation. Required.",
    required = True,
)
arg_parser.add_argument("--with-cuda", "-C",
    help="By default True. It is not provided directly by the user but is hardcoded in cnn_baseline.sh or cnn_baseline.cuda.sh\
    To use cuda cores please run the cnn_baseline_cuda.sh (--with-cuda automatically set to True), otherwise \
    cnn_baseline.sh file (--with-cuda set to False).",
    default=0,
    type=int
)
arg_parser.add_argument("--model", "-m",
    help="Model architecture to use. It will be imported from ",
    default='CnnClassificationBaseline'
)
arg_parser.add_argument("--task-id", "-t",
    help="Task id signaling which cross-validation fold this job is running",
    required=True,
    type=int
)

a = arg_parser.parse_args()


datasets = []

sys.stdout = open(os.devnull, 'w')


# LOAD DATA
#----------


splits_path = f'/projects/0/einf2380/data/pMHCII/features_output_folder/CNN/hla_drb1_0101_15mers'#{a.exp_name}'
# chose to load either clustered or shuffled data
split_type = ('shuffled', 'clustered')[a.cluster]

train_db = f"{splits_path}/{split_type}/{a.task_id}/train.hdf5"
val_db = f"{splits_path}/{split_type}/{a.task_id}/valid.hdf5"
test_db = f"{splits_path}/{split_type}/{a.task_id}/test.hdf5"

# create output directory:
output_dir = f'/projects/0/einf2380/data/pMHCII/trained_models/CNN/classification/{a.exp_name}/struct/{a.model}/{split_type}'

outdir = f"{output_dir}/{a.task_id}"
try:
    os.makedirs(outdir)
except OSError as error:
    print(f"folder {outdir} already created")
    

data_set = DataSet(train_database=train_db,
    test_database = test_db,
    valid_database = val_db,
    chain1 = "M",
    chain2 = "P",
    grid_info = (35,30,30),
    select_feature = {#'AtomicDensities_ind': 'all',
        "Feature_ind": ['Edesolv',
        'RCD_*', 'bsa', 'charge', 'coulomb', 'vdwaals']},
    select_target = "BIN_CLASS",
    normalize_features = True,
    normalize_targets = False,
    pair_chain_feature = None,
    mapfly = False,
    tqdm = True,
    clip_features = False,
    process = True,
)

architecture='' #Useless, but it makes vscode not complain
exec('architecture='+a.model)
model = NeuralNet(data_set=data_set,
    model = architecture,
    task = "class",
    chain1 = "M",
    chain2 = "P",
    cuda = bool(a.with_cuda),
    ngpu = a.with_cuda,
    plot = True,
    save_classmetrics = True,
    outdir = outdir
)

model.train(
    nepoch = a.epochs,
    divide_trainset = None,
    train_batch_size = a.batch,
    save_model = "best",
    save_epoch = "all",
    hdf5 = "metrics.hdf5",
    num_workers=12
)

# START TRAINING
#---------------