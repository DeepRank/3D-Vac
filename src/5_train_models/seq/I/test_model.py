import pandas
import torch
import os.path as path
import sys
sys.path.append(path.abspath("."))
sys.path.append(path.abspath("../"))
sys.path.append(path.abspath("../../"))
sys.path.append(path.abspath("../../../../"))

from seq.SeqBased_models import MlpRegBaseline, train_f, evaluate
from seq.datasets import Class_Seq_Dataset, create_unique_csv

from sklearn.model_selection import StratifiedKFold, KFold # used for normal cross validation
from sklearn.model_selection import train_test_split
from sklearn.model_selection import LeaveOneGroupOut

from torch.utils.data import Subset
from torch.utils.data import DataLoader
from mhcflurry.regression_target import from_ic50, to_ic50
from sklearn import metrics

n_models = 10
neurons_per_layer = 500
batch = 64
epochs = 50
threshold=500
task='regression'
encoder = 'blosum_with_allele'
allele_to_pseudosequence_csv="/projects/0/einf2380/data/external/unprocessed/mhcflurry.allele_sequences.csv"
device = 'cpu'

def test_model(csv_path, pth_path):
    datasets = []
    dataset = Class_Seq_Dataset(
        csv_path,
        encoder,
        device,
        threshold = threshold,
        cluster_column = None,
        task=task,
        allele_to_pseudosequence_csv_path=allele_to_pseudosequence_csv
    )

    train_val_idx = dataset.df.loc[dataset.df.test == 0].index
    test_idx = dataset.df.loc[dataset.df.test == 1].index

    #train_idx,validation_idx = train_test_split(train_val_idx, test_size=2/9) #2/9*0,9=0.2

    test_subset = Subset(dataset, test_idx)

    test_dataloader = DataLoader(test_subset, batch_size=batch)

    input_dimensions = dataset.input_shape
    AUCs = []
    # all_targets = []
    # all_preds = []
    print('Single MLP AUCs:')
    pretrained_models = torch.load(pth_path)
    for i in range(n_models):
        model = MlpRegBaseline(n_output_nodes=1, neurons_per_layer=neurons_per_layer, input_shape=input_dimensions).to(device)
        ## to change so it loops over the 10 models
        model.load_state_dict(pretrained_models['models_data'][i]['model'])
        model.eval()

        logits = []
        reg_targets = []
        with torch.no_grad():
            for X,y in test_dataloader:
                X.to(device)
                y.to(device)
                mb_logits = model(X)
                logits.extend(mb_logits)
                reg_targets.extend(y)

        reg_threshold = from_ic50(threshold)
        targets = [float(x > reg_threshold) for x in reg_targets]
        # all_targets.append(targets)
        # all_preds.append(logits)

        auc = metrics.roc_auc_score(targets, logits)
        print(auc)
        AUCs.append(auc)

    print(f'Mean {sum(AUCs)/len(AUCs)}')

    # P = []
    # for j in range(len(all_preds[0])):
    #     preds = [all_preds[i][j] for i in range(len(all_preds))]
    #     P.append(sum(preds)/len(preds))
    
    # ensemble_auc = metrics.roc_auc_score(all_targets[0], P)
    # print(f'Ensamble predictions AUC {ensemble_auc}')

    return AUCs


#print('SHUFFLED AUCs')
#csv_path = '/home/dmarz/3D-Vac/src/5_train_models/seq/I/experiments/mhc_vs_mlp/train_validation_test_cases_exp_shuffled-B04HF.csv'
#pth_path = '/projects/0/einf2380/data/pMHCI/trained_models/MLP_rerun/shuffled/mlp_classification_blosum_with_allele_encoder_500_neurons_50_epochs_exp_shuffled_64_batch_size.pt'
#shuffled_aucs = test_model(csv_path, pth_path)

print('\n')
print('OLD SHUFFLED AUCs')
csv_path = '/home/dmarz/3D-Vac/src/5_train_models/seq/I/experiments/mhc_vs_mlp/train_validation_test_cases_exp_shuffled-B04HF.csv'
pth_path = '/projects/0/einf2380/data/pMHCI/trained_models/MLP/mlp_classification_blosum_with_allele_encoder_500_neurons_50_epochs_exp_shuffled_64_batch_size.pt'
old_shuffled_aucs = test_model(csv_path, pth_path)

#print('\n')
#print('ALLELE-CLUSTERED AUCs')
#csv_path = '/home/dmarz/3D-Vac/src/5_train_models/seq/I/experiments/mhc_vs_mlp/train_validation_test_cases_exp_clustered_alleles-CPJFI.csv'
#pth_path = '/projects/0/einf2380/data/pMHCI/trained_models/MLP_rerun/shuffled/mlp_classification_blosum_with_allele_encoder_500_neurons_50_epochs_exp_clustered_alleles_64_batch_size.pt'
#allele_aucs = test_model(csv_path, pth_path)

print('\n')
print('OLD ALLELE-CLUSTERED AUCs')
csv_path = '/home/dmarz/3D-Vac/src/5_train_models/seq/I/experiments/mhc_vs_mlp/train_validation_test_cases_exp_clustered_alleles-CPJFI.csv'
pth_path = '/projects/0/einf2380/data/pMHCI/trained_models/MLP/mlp_classification_blosum_with_allele_encoder_500_neurons_50_epochs_exp_clustered_alleles_64_batch_size.pt'
old_allele_aucs = test_model(csv_path, pth_path)

'''Error when loading Daniil's pretrained networks:
RuntimeError: Error(s) in loading state_dict for MlpRegBaseline:
        size mismatch for linear.1.weight: copying a param with shape torch.Size([1722]) from checkpoint, the shape in current model is torch.Size([2050]).
        size mismatch for linear.1.bias: copying a param with shape torch.Size([1722]) from checkpoint, the shape in current model is torch.Size([2050]).
        size mismatch for linear.1.running_mean: copying a param with shape torch.Size([1722]) from checkpoint, the shape in current model is torch.Size([2050]).
        size mismatch for linear.1.running_var: copying a param with shape torch.Size([1722]) from checkpoint, the shape in current model is torch.Size([2050]).
        size mismatch for linear.2.weight: copying a param with shape torch.Size([500, 1722]) from checkpoint, the shape in current model is torch.Size([500, 2050]).

'''
