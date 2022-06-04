from torch.utils.data import Dataset
import torch
import blosum
import pickle
import pandas as pd

# encoding functions:
aminoacids = ('ACDEFGHIKLMNPQRSTVWY')
def peptide2onehot(peptide):
    AA_eye = torch.eye(20, dtype=torch.float)
    return [AA_eye[aminoacids.index(res)].tolist() for res in peptide]

def peptide2blosum(peptide):
    mat = blosum.BLOSUM(62)
    blosum_t = [[]]
    blosum_aa = ["A"]
    for aa in mat.keys():
        if aa[0] in aminoacids and aa[1] in aminoacids:
            if len(blosum_t[-1]) < 20:
                blosum_t[-1].append(mat[aa])
            else:
                blosum_aa.append(aa[0])
                blosum_t.append([mat[aa]])
    blosum_t = torch.tensor(blosum_t)
    blosum_aa = "".join(blosum_aa)
    return [blosum_t[blosum_aa.index(res)].tolist() for res in peptide]

# the whole dataset class, which extends the torch Dataset class
class Reg_Seq_Dataset(Dataset):
    def __init__(self, csv_peptides,csv_ba_values, encoder):
        self.csv_peptides = csv_peptides
        self.csv_ba_values = csv_ba_values
        self.ba_values = torch.tensor(self.csv_ba_values, dtype=torch.float64)
        if encoder == "blosum":
            self.peptides = torch.tensor([peptide2blosum(p) for p in self.csv_peptides])
        else:
            self.peptides = torch.tensor([peptide2onehot(p) for p in self.csv_peptides])
    def __getitem__(self, idx):
        return self.peptides[idx], self.ba_values[idx]
    def __len__(self):
        return len(self.peptides)

class Class_Seq_Dataset(Dataset):
    def __init__(self, csv_peptides,labels, encoder):
        self.csv_peptides = csv_peptides
        self.labels = torch.tensor(labels)
        if encoder == "blosum":
            self.peptides = torch.tensor([peptide2blosum(p) for p in self.csv_peptides])
        else:
            self.peptides = torch.tensor([peptide2onehot(p) for p in self.csv_peptides])
    def __getitem__(self, idx):
        return self.peptides[idx], self.labels[idx]
    def __len__(self):
        return len(self.peptides)

def load_reg_seq_data(csv_file, threshold):
    csv_peptides = []
    csv_ba_values = []
    with open(csv_file) as csv_f:
        rows = [row.replace("\n", "").split(",") for row in csv_f]
        for row in rows:
            if float(row[3]) <= threshold:
                csv_peptides.append(row[2])
                csv_ba_values.append(float(row[3]))
    return csv_peptides, csv_ba_values

def load_class_seq_data(csv_file, threshold, group=False): # if cluster_file is set, performs a clustered data loading
    csv_peptides = []
    labels = []
    groups = [] 
    df = pd.read_csv(csv_file)

    csv_peptides = df["peptide"].tolist()

    # binder or non binder if the mean value of redundant peptides less than the threshold:
    labels = [(0.,1.,)[ df[df["peptide"] == peptide]["measurement_value"].mean() < threshold ] for peptide in csv_peptides]

    if group: # peptides grouped by clusters for the clustered classification
        groups = df["cluster"].tolist() # used for LeaveOneGroupOut sklearn function when doing the clustered classification
        return csv_peptides, labels, groups
    return csv_peptides, labels

# functions to transform/transform back binding affinity values
def sig_norm(ds,training_mean,training_std):
    ds = torch.log(ds)
    ds = (ds-training_mean)/training_std
    return torch.sigmoid(ds)

def sig_denorm(ds, training_mean, training_std):
    ds = torch.logit(ds)
    ds = ds*training_std+training_mean
    return torch.exp(ds)

def li_norm(ds,training_max,training_min):
    ds = torch.log(ds)
    return (ds-training_min)/(training_max - training_min)

def li_denorm(ds, training_max, training_min):
    ds = ds*(training_max - training_min)+training_min
    return torch.exp(ds)

def custom_norm(ds): # custom normalization
    return ds

def custom_denorm(ds):
    return ds