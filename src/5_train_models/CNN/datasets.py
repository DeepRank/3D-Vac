from torch.utils.data import Dataset
import torch
import blosum
import pickle

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

class Clust_Class_Seq_Dataset(Dataset):
    def __init__(self, first_cluster, last_cluster, clusters, encoder):
        self.peptides = []
        self.labels = []
        for i in range(first_cluster, last_cluster+1):
            for peptide,label in clusters[str(i)]:
                if encoder == "sparse": 
                    peptide = peptide2onehot(peptide)
                if encoder == "blosum":
                    peptide = peptide2blosum(peptide)
                self.peptides.extend(peptide)
                self.labels.extend(label)
        self.labels = torch.tensor(self.labels).long()
        self.peptides = torch.stack(self.peptides)
        def __getitem__(self,idx):
            return self.peptides[idx], self.labels[idx]
        def __len__(self,idx):
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

def load_clust_class_seq_data(cluster_file, csv_path):
    csv_rows = [row.split(",") for row in open(csv_path)]
    pkl_clusters = pickle.load(open(cluster_file))
    clusters = {}
    for c_idx, c_peptides in enumerate(pkl_clusters.values()):
        cluster = []
        for peptide in c_peptides:
            ba_values = torch.tensor([float(row[3]) for row in csv_rows if row[2] == peptide])
            label = (0.,1.)[ba_values.mean() <= 500]
            cluster.append((peptide,label))
        clusters[str(c_idx)] = cluster
    return clusters

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

def load_class_seq_data(csv_file, threshold):
    csv_peptides = []
    labels = []
    with open(csv_file) as csv_f:
        rows = [row.replace("\n", "").split(",") for row in csv_f]
        for row in rows:
            csv_peptides.append(row[2])
            labels.append((0.,1.)[float(row[3]) <= threshold])
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