from torch.utils.data import Dataset
import torch
import blosum
<<<<<<< HEAD
import pickle
=======
>>>>>>> 149a17263114ce545c6cc406d999ff55bb6cff49
import pandas as pd

# encoding functions:
aminoacids = ('ACDEFGHIKLMNPQRSTVWY')
def peptide2onehot(peptide):
    AA_eye = torch.eye(20, dtype=torch.float)
    return [AA_eye[aminoacids.index(res)].tolist() for res in peptide]

def peptide2blosum(peptide):
<<<<<<< HEAD
=======
    """Function used to generate a multidimentional array (which is later 
    converted to tensor) containing arrays of BLOSUM62 encoded residue.

    Args:
        peptide (string): sequence of the peptide

    Returns:
        array: array containing len(peptide) arrays (of len 20) of encoded residue
    """
>>>>>>> 149a17263114ce545c6cc406d999ff55bb6cff49
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

def peptide2mixed(peptide):
<<<<<<< HEAD
=======
    """Function used to encode the residues with a first vector of 20
    (onehot encoding) followed by a vector of 20 for BLOSUM62.

    Args:
        peptide (string): Sequence of the peptide.

    Returns:
        _type_: Array containing len(peptides) of arrays (of len 40) encoded residue.
    """
>>>>>>> 149a17263114ce545c6cc406d999ff55bb6cff49
    AA_eye = torch.eye(20, dtype=torch.float)
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
    return [blosum_t[blosum_aa.index(res)].tolist() + AA_eye[aminoacids.index(res)].tolist() for res in peptide]

# the whole dataset class, which extends the torch Dataset class
class Reg_Seq_Dataset(Dataset):
    def __init__(self, csv_peptides,csv_ba_values, encoder):
<<<<<<< HEAD
=======
        """Class used to store the dataset for the sequence based regression. This function
        was not developed further when the focus shifted towards classification.

        Args:
            csv_peptides (array): Array of peptides generated before calling the function.
            csv_ba_values (array): Labels of csv_peptides in the same order as csv_peptides.
            encoder (string): Type of encoding used. Might be `sparse`, `blosum` or `mixed`.
        """
>>>>>>> 149a17263114ce545c6cc406d999ff55bb6cff49
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
    def __init__(self, csv_peptides,labels, encoder, device):
<<<<<<< HEAD
=======
        """Class used to store the dataset for the sequence based classification.

        Args:
            csv_peptides (array): Array of peptides generated before calling the function.
            labels (array): Binder/non-binder labels in the same order as csv_peptides.
            encoder (string): Type of encoding used. Might be `sparse`, `blosum` or `mixed`.
            device (torch.device): can be either "cpu" or torch.device("cuda:0").
        """
>>>>>>> 149a17263114ce545c6cc406d999ff55bb6cff49
        self.csv_peptides = csv_peptides
        self.labels = torch.tensor(labels).long()
        if encoder == "blosum":
            self.peptides = torch.tensor([peptide2blosum(p) for p in self.csv_peptides])
        else:
            self.peptides = torch.tensor([peptide2onehot(p) for p in self.csv_peptides])
        if encoder == "mixed":
            self.peptides = torch.tensor([peptide2mixed(p) for p in self.csv_peptides])
        self.peptides = self.peptides.to(device)
        self.labels = self.labels.to(device)
    def __getitem__(self, idx):
        return self.peptides[idx], self.labels[idx]
    def __len__(self):
        return len(self.peptides)

def load_reg_seq_data(csv_file, threshold):
<<<<<<< HEAD
    csv_peptides = []
    csv_ba_values = []
    with open(csv_file) as csv_f:
        rows = [row.replace("\n", "").split(",") for row in csv_f]
        for row in rows:
            if float(row[3]) <= threshold:
                csv_peptides.append(row[2])
                csv_ba_values.append(float(row[3]))
    return csv_peptides, csv_ba_values

def load_class_seq_data(csv_file, threshold): # if cluster_file is set, performs a clustered data loading
    csv_peptides = []
    labels = []
=======
    """Function used to read the data from the csv_file and generate the csv_peptides
    and csv_ba_values arrays.

    Args:
        csv_file (string): Path to db1.
        threshold (int): Threshold to define binding/non binding.

    Returns:
        csv_peptides: Array used as an argument for Class_Seq_Dataset.
        csv_ba_values: Array used as an argument for Class_Seq_Dataset.
    """
    df = pd.read_csv(csv_file)
    csv_peptides = df["peptide"].tolist()
    csv_ba_values = [value for value in df["measurement_value"] if value <= threshold]
    return csv_peptides, csv_ba_values

def load_class_seq_data(csv_file, threshold): # if cluster_file is set, performs a clustered data loading
    """Function used to read the data from the csv_file and generate the csv_peptides,
    labels and groups arrays.

    Args:
        csv_file (string): Path to db1.
        threshold (int): Threshold to define binding/non binding.

    Returns:
        csv_peptides: Array used as an argument for Class_Seq_Dataset.
        labels: Array used as an argument for Class_Seq_Dataset.
        groups: Array indicating cluster to which csv_peptides value belong.
    """
>>>>>>> 149a17263114ce545c6cc406d999ff55bb6cff49
    df = pd.read_csv(csv_file)

    csv_peptides = df["peptide"].tolist()

<<<<<<< HEAD
    # binder or non binder if the mean value of redundant peptides less than the threshold:
    labels = [(0.,1.,)[value < threshold] for value in df["measurement_value"]]
=======
    # binder or non binder if the value of the peptide is less than the threshold (redundant peptides will have
    # different values)
    labels = [(0.,1.,)[value < threshold] for value in df["measurement_value"]]

    # binder or non binder if the mean value of redundant peptides less than the threshold:
>>>>>>> 149a17263114ce545c6cc406d999ff55bb6cff49
    # labels = [(0.,1.,)[ df[df["peptide"] == peptide]["measurement_value"].mean() < threshold ] for peptide in csv_peptides]

    # peptides grouped by clusters for the clustered classification
    groups = df["cluster"].tolist() # used for LeaveOneGroupOut sklearn function when doing the clustered classification
    return csv_peptides, labels, groups

# functions to transform/transform back binding affinity values
def sig_norm(ds,training_mean,training_std):
<<<<<<< HEAD
=======
    """Function used for a sigmoid transformation of binding affinity values.

    Args:
        ds (torch.tensor): Tensor containing binding affinity values which will be transformed.
        training_mean (float): Mean value used for normalization.
        training_std (float): Standard deviation used for normalization.

    Returns:
        torch.tensor: Transformed binding affinity values.
    """
>>>>>>> 149a17263114ce545c6cc406d999ff55bb6cff49
    ds = torch.log(ds)
    ds = (ds-training_mean)/training_std
    return torch.sigmoid(ds)

def sig_denorm(ds, training_mean, training_std):
<<<<<<< HEAD
=======
    """Function used to go back to the initial value of binding affinity from the sigmoid tranformation.

    Args:
        ds (torch.tensor): Tensor containing binding affinity values which will be transformed.
        training_mean (float): Mean value used for normalization.
        training_std (float): Standard deviation used for normalization.

    Returns:
        torch.tensor: Transformed binding affinity values.
    """
>>>>>>> 149a17263114ce545c6cc406d999ff55bb6cff49
    ds = torch.logit(ds)
    ds = ds*training_std+training_mean
    return torch.exp(ds)

def li_norm(ds,training_max,training_min):
<<<<<<< HEAD
=======
    """Function used for a normalization of binding affinity values.

    Args:
        ds (torch.tensor): Tensor containing binding affinity values which will be transformed.
        training_max (float): Max value of binding affinities used for the normalization.
        training_min (float): Min value used for normalization.

    Returns:
        torch.tensor: Transformed binding affinity values.
    """
>>>>>>> 149a17263114ce545c6cc406d999ff55bb6cff49
    ds = torch.log(ds)
    return (ds-training_min)/(training_max - training_min)

def li_denorm(ds, training_max, training_min):
<<<<<<< HEAD
=======
    """Function used to obtain binding affinity values from li normalized values.

    Args:
        ds (torch.tensor): Tensor containing binding affinity values which will be transformed.
        training_max (float): Max value of binding affinities used for the normalization.
        training_min (float): Min value used for normalization.

    Returns:
        torch.tensor: Transformed binding affinity values.
    """
>>>>>>> 149a17263114ce545c6cc406d999ff55bb6cff49
    ds = ds*(training_max - training_min)+training_min
    return torch.exp(ds)

def custom_norm(ds): # custom normalization
    return ds

def custom_denorm(ds):
    return ds