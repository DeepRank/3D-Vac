from torch.utils.data import Dataset
import torch
import blosum
import pandas as pd
import random
from mhcflurry.regression_target import from_ic50, to_ic50
import numpy
import time
import os

# define the aminoacid alphabet and build the one-hot tensor:
aminoacids = ('ACDEFGHIKLMNPQRSTVWYX')
AA_eye = torch.eye(len(aminoacids), dtype=torch.float16)
matrix = blosum.BLOSUM(62)

# build the blosum62 tensor and its alphabet:
# blosum_t = [[]]
# blosum_aa = ["A"]
# for aa in mat.keys():
#     if aa in aminoacids: 
#         for t in mat[aa]:
#             if t in aminoacids:
#                 if len(blosum_t[-1]) < len(aminoacids):
#                     blosum_t[-1].append(mat[aa])
#                 else:
#                     blosum_aa.append(aa)
#                     blosum_t.append([mat[aa]])
# blosum_t = torch.tensor(blosum_t, dtype=torch.float16)
# blosum_aa = "".join(blosum_aa)

def seq_to_mat(res, matrix=matrix):
    return list(matrix[res].values())
    

def length_agnostic_encode_p(p):
    """Build a length agnostic representation of peptide having a sequence length not greater than 15.
    MHCflurry type of encoding. Given any length, returns a concatenation of left, center and right align
    of the peptide sequence. For instance, given a peptide "ACGHDGDDF": "ACGHDGDDFXXXXXX", "XXXACGHDGDDFXXX" and
    "XXXXXXACGHDGDDF" are its left-aligned, center-aligned and right aligned representations, respectively.

    Args:
        p (String): Amino acid sequence of the peptide.

    Returns:
        String: Concatenation of left align, center align and right align of the peptide
    """
    rep_len = 15 # this can be updated if in the future the threshold for peptide representation is modified
    x = "X"*(rep_len-len(p))
    half_x = x[:len(x)//2]
    left_al = p + x
    right_al = x + p
    center_al = half_x + p + half_x
    if len(center_al) < 15:
        center_al = center_al + "X"*(15-len(center_al))
    return left_al + center_al + right_al

def allele_peptide2blosum(allele, peptide):
    """Encodes the allele with the peptide using the BLOSUM62 encoding matrice.

    Args:
        allele (String): Pseudosequence of the allele
        peptide (String): Sequence of the peptide

    Returns:
        List: Concatenated representation of the allele and peptide with shape (allele+peptide)*21
    """
    peptide_arr = [seq_to_mat(res) for res in peptide]
    allele_arr = [seq_to_mat(res) for res in allele]
    return allele_arr + peptide_arr

def allele_peptide2onehot(allele, peptide):
    """Encodes the allele with the peptide using one-hot representation.

    Args:
        allele (String): Pseudosequence of the allele
        peptide (String): Sequence of the peptide

    Returns:
        List: Concatenated representation of the allele and peptide with shape (allele+peptide)*21
    """
    allele_arr = [AA_eye[aminoacids.index(res)].tolist() for res in allele]
    peptide_arr = [AA_eye[aminoacids.index(res)].tolist() for res in peptide]
    return allele_arr + peptide_arr

def peptide2onehot(peptide):
    """Encodes the peptide into a one-hot representation.

    Args:
        peptide (String): Sequence of the peptide

    Returns:
        List: One hot representation of the peptide with shape peptide*21
    """
    return [AA_eye[aminoacids.index(res)].tolist() for res in peptide]

def peptide2blosum(peptide):
    """Encodes the peptide into a BLOSUM62 representation.

    Args:
        peptide (String): Sequence of the peptide

    Returns:
        List: One hot representation of the peptide with shape peptide*21
    """
    return [seq_to_mat(res) for res in peptide]

class Class_Seq_Dataset(Dataset):
    def __init__(
        self, csv, encoder, device,
        threshold=500,
        cluster_column=None,
        task="classification",
        allele_to_pseudosequence_csv_path="/projects/0/einf2380/data/external/unprocessed/mhcflurry.allele_sequences.csv"
    ):
        """Class used to store the dataset for the sequence based MLP. Prepare the data both for classification and regression
        as well as different data splitting methods (shuffled, clustered or train (and validation) and test in different csv).
        Once the data is loaded it can be split using these different methods.
        The features are the encoding used for peptides or allele and peptides. Supports different length peptides.

        Args:
            csv (array): Path to DB1 csv.
            encoder (string): Encoding methods. Peptide only or peptide + allele encoding.
            device (string): CPU or CUDA.
            threshold (array): Value to define binders/non binders for the classification task.
            cluster_column (string): Column of the csv file to use containing cluster mapping for samples. Only for clustered data.
            allele_to_pseudosoquence_csv_path (string): path to the mhcflurry csv file containing mapings for alleles to pseudosequences
        """
        self.task = task
        self.threshold = threshold
        self.cluster_column = cluster_column
        allele_to_pseudoseq_df = pd.read_csv(allele_to_pseudosequence_csv_path)
        allele_to_pseudoseq = dict(zip(allele_to_pseudoseq_df.allele, allele_to_pseudoseq_df.sequence))
        
        df = pd.read_csv(csv)
        self.df = df.loc[df.peptide.str.len() <= 15]
        self.df = self.df.loc[self.df.allele.isin(list(allele_to_pseudoseq.keys()))]
        self.pseudosequences = [allele_to_pseudoseq[a] for a in self.df.allele]

        self.labels, self.groups = self.load_class_seq_data()

        self.csv_peptides = [length_agnostic_encode_p(p) for p in self.df.peptide.tolist()]

        if encoder == "blosum_with_allele":
            self.peptides = torch.tensor([allele_peptide2blosum(a, p) for a, p in zip(self.pseudosequences, self.csv_peptides)])
        if encoder == "sparse_with_allele":
            self.peptides = torch.tensor([allele_peptide2onehot(a, p) for a, p in zip(self.pseudosequences, self.csv_peptides)])
        if encoder == "blosum":
            self.peptides = torch.tensor([peptide2blosum(p) for p in self.csv_peptides])
        if encoder == "sparse":
            self.peptides = torch.tensor([peptide2onehot(p) for p in self.csv_peptides])
        self.input_shape = (self.peptides.shape[1], self.peptides.shape[2])

        # convert NaN trash cluster to 0:
        # self.groups = torch.tensor(self.groups)+1
        # self.groups = torch.nan_to_num(self.groups)
        
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


        # binder or non binder if the value of the peptide is less than the threshold (redundant peptides will have
        # different values)
        if self.task == "classification":
            labels = [(0,1)[value < self.threshold] for value in self.df["measurement_value"]]
            labels = torch.tensor(labels, dtype=torch.long)

            # binder or non binder if the mean value of redundant peptides less than the threshold:
            # labels = [(0.,1.,)[ self.df[self.df["peptide"] == peptide]["measurement_value"].mean() < self.threshold ] for peptide in csv_peptides]
        else:
            labels = self.load_reg_data()

        # peptides grouped by clusters for the clustered classification
        groups = []
        if self.cluster_column != None:
            groups = torch.tensor(self.df[self.cluster_column].tolist(), dtype=torch.float16)# used for LeaveOneGroupOut sklearn function when doing the clustered classification
        return labels, groups

    def load_reg_data(self):
        """Converts measurement_value into float values between 0 and 1 using MHCflurry 2.0 ic50 conversion.

        Returns:
            torch.tensor: Transformed binding affinity values.
        """
        measurements = numpy.array(self.df.measurement_value.tolist())
        x = from_ic50(measurements) 
        return torch.tensor(x, dtype=torch.float32)
    

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

def create_unique_csv(train_csv, test_csv, model_name):
    """Concatenates train_csv (containing validation as well) with the test_csv into one csv which can be
    loaded into the Class_Seq_Dataset. This csv will have an added `test` column indicating which sample
    is used for train and validation (test == 0) and which is used for test (test == 1).

    Args:
        train_csv (_type_): csv containing train and validation cases
        test_csv (_type_): _description_
        model_name (_type_): _description_

    Returns:
        _type_: _description_
    """
    aa = list("ABCDEFGHIJKLMOPQRSTUV0123456789")
    rand_str = "".join(random.sample(aa, 5))
    tvt_csv_path = f"train_validation_test_cases_{model_name}-{rand_str}.csv"
    test_df = pd.read_csv(test_csv)
    test_df["test"] = 1
    train_validation_df = pd.read_csv(train_csv)
    train_validation_df["test"] = 0
    concatenated_csv = pd.concat([train_validation_df, test_df], ignore_index=True).to_csv(tvt_csv_path, index=False)
    csv_path = os.path.abspath(tvt_csv_path) 
    return csv_path


if __name__ == "__main__":
    # dataset = Class_Seq_Dataset(
    #     "/home/daqop/mountpoint_snellius/experiments/BA_pMHCI_human_quantitative_only_eq_shuffled_train_validation.csv",
    #     device="cpu",
    #     encoder="blosum_with_allele",
    #     cluster_column="cluster_set_10",
    #     task="regression",
    #     allele_to_pseudosequence_csv_path="/home/daqop/mountpoint_snellius/3D-Vac/data/external/unprocessed/mhcflurry.allele_sequences.csv"
    # )

    dataset = Class_Seq_Dataset(
        "/projects/0/einf2380/data/external/processed/I/experiments/BA_pMHCI_human_quantitative_only_eq_shuffled_train_validation.csv",
        device="cpu",
        encoder="blosum_with_allele",
        cluster_column="cluster_set_10",
        task="regression",
        allele_to_pseudosequence_csv_path="/projects/0/einf2380/data/external/unprocessed/mhcflurry.allele_sequences.csv"
    )