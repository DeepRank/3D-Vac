from torch.utils.data import Dataset
import torch
import blosum
import pandas as pd


# encoding functions:
aminoacids = ('ACDEFGHIKLMNPQRSTVWYX')
def allele_peptide2onehot(allele, peptide):
    not_in_allele = [res for res in allele if res not in aminoacids]
    AA_eye = torch.eye(len(aminoacids), dtype=torch.float)
    allele_arr = [AA_eye[aminoacids.index(res)].tolist() for res in allele]
    peptide_arr = [AA_eye[aminoacids.index(res)].tolist() for res in peptide]
    return allele_arr + peptide_arr

def peptide2onehot(peptide):
    AA_eye = torch.eye(len(aminoacids), dtype=torch.float)
    return [AA_eye[aminoacids.index(res)].tolist() for res in peptide]

def length_agnostic_encode_p(p):
    rep_len = 15
    x = "X"*(rep_len-len(p))
    half_x = x[:len(x)//2]
    left_al = x + p
    right_al = p + x
    center_al = half_x + p + half_x
    if len(center_al) < 15:
        center_al = center_al + "X"*(15-len(center_al))
    return left_al + center_al + right_al

def peptide2blosum(peptide):
    """Function used to generate a multidimentional array (which is later 
    converted to tensor) containing arrays of BLOSUM62 encoded residue.

    Args:
        peptide (string): sequence of the peptide

    Returns:
        array: array containing len(peptide) arrays (of len 20) of encoded residue
    """
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
    """Function used to encode the residues with a first vector of 20
    (onehot encoding) followed by a vector of 20 for BLOSUM62.

    Args:
        peptide (string): Sequence of the peptide.

    Returns:
        _type_: Array containing len(peptides) of arrays (of len 40) encoded residue.
    """
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
        """Class used to store the dataset for the sequence based regression. This function
        was not developed further when the focus shifted towards classification.

        Args:
            csv_peptides (array): Array of peptides generated before calling the function.
            csv_ba_values (array): Labels of csv_peptides in the same order as csv_peptides.
            encoder (string): Type of encoding used. Might be `sparse`, `blosum` or `mixed`.
        """
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
    def __init__(
        self, csv, encoder, device,
        threshold=500,
        cluster_column=None,
    ):
        """Class used to store the dataset for the sequence based classification.

        Args:
            csv_peptides (array): Array of peptides generated before calling the function.
            labels (array): Binder/non-binder labels in the same order as csv_peptides.
            encoder (string): Type of encoding used. Might be `sparse`, `blosum` or `mixed`.
            device (torch.device): can be either "cpu" or torch.device("cuda:0").
        """
        df = pd.read_csv(csv)
        self.df = df.loc[df["peptide"].str.len() <= 15]
        self.threshold = threshold
        self.cluster_column = cluster_column

        self.csv_peptides, self.labels, self.groups = self.load_class_seq_data()

        self.csv_peptides = [length_agnostic_encode_p(p) for p in self.csv_peptides]
        self.labels = torch.tensor(self.labels).long()

        if encoder == "sparse_with_allele":
            pseudosequences = self.df["pseudosequence"].tolist()
            self.peptides = torch.tensor([allele_peptide2onehot(a, p) for a, p in zip(pseudosequences, self.csv_peptides)])
        if encoder == "blosum":
            self.peptides = torch.tensor([peptide2blosum(p) for p in self.csv_peptides])
        if encoder == "sparse":
            self.peptides = torch.tensor([peptide2onehot(p) for p in self.csv_peptides])
        if encoder == "mixed":
            self.peptides = torch.tensor([peptide2mixed(p) for p in self.csv_peptides])

        self.peptides = self.peptides.to(device)
        self.labels = self.labels.to(device)
        self.input_size = self.peptides.shape[1] * self.peptides.shape[2]
        
    def __getitem__(self, idx):
        return self.peptides[idx], self.labels[idx]

    def __len__(self):
        return len(self.peptides)

    def load_class_seq_data(self): # if cluster_file is set, performs a clustered data loading
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

        csv_peptides = self.df["peptide"].tolist()

        # binder or non binder if the value of the peptide is less than the threshold (redundant peptides will have
        # different values)
        # labels = [(0.,1.,)[value < self.threshold] for value in self.df["measurement_value"]]

        # binder or non binder if the mean value of redundant peptides less than the threshold:
        labels = [(0.,1.,)[ self.df[self.df["peptide"] == peptide]["measurement_value"].mean() < self.threshold ] for peptide in csv_peptides]

        # peptides grouped by clusters for the clustered classification
        groups = []
        if self.cluster_column != None:
            groups = self.df[self.cluster_column].tolist()# used for LeaveOneGroupOut sklearn function when doing the clustered classification
        return csv_peptides, labels, groups

def load_reg_seq_data(csv_file, threshold):
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


# functions to transform/transform back binding affinity values
def sig_norm(ds,training_mean,training_std):
    """Function used for a sigmoid transformation of binding affinity values.

    Args:
        ds (torch.tensor): Tensor containing binding affinity values which will be transformed.
        training_mean (float): Mean value used for normalization.
        training_std (float): Standard deviation used for normalization.

    Returns:
        torch.tensor: Transformed binding affinity values.
    """
    ds = torch.log(ds)
    ds = (ds-training_mean)/training_std
    return torch.sigmoid(ds)

def sig_denorm(ds, training_mean, training_std):
    """Function used to go back to the initial value of binding affinity from the sigmoid tranformation.

    Args:
        ds (torch.tensor): Tensor containing binding affinity values which will be transformed.
        training_mean (float): Mean value used for normalization.
        training_std (float): Standard deviation used for normalization.

    Returns:
        torch.tensor: Transformed binding affinity values.
    """
    ds = torch.logit(ds)
    ds = ds*training_std+training_mean
    return torch.exp(ds)

def li_norm(ds,training_max,training_min):
    """Function used for a normalization of binding affinity values.

    Args:
        ds (torch.tensor): Tensor containing binding affinity values which will be transformed.
        training_max (float): Max value of binding affinities used for the normalization.
        training_min (float): Min value used for normalization.

    Returns:
        torch.tensor: Transformed binding affinity values.
    """
    ds = torch.log(ds)
    return (ds-training_min)/(training_max - training_min)

def li_denorm(ds, training_max, training_min):
    """Function used to obtain binding affinity values from li normalized values.

    Args:
        ds (torch.tensor): Tensor containing binding affinity values which will be transformed.
        training_max (float): Max value of binding affinities used for the normalization.
        training_min (float): Min value used for normalization.

    Returns:
        torch.tensor: Transformed binding affinity values.
    """
    ds = ds*(training_max - training_min)+training_min
    return torch.exp(ds)

def custom_norm(ds): # custom normalization
    return ds

def custom_denorm(ds):
    return ds


if __name__ == "__main__":
    dataset = Class_Seq_Dataset(
        "/home/daqop/mountpoint_snellius/3D-Vac/data/external/processed/hla0201_pseudoseq.csv",
        device="cpu",
        encoder="sparse_with_allele")
    print(dataset.peptides.shape)