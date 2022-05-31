from torch.utils.data import Dataset
import torch
import blosum


# the whole dataset class, which extends the torch Dataset class
class Peptides(Dataset):
    def peptide2onehot(self,peptide):
        AA_eye = torch.eye(20, dtype=torch.float)
        return [AA_eye[self.aminoacids.index(res)].tolist() for res in peptide]

    def peptide2blosum(self,peptide):
        mat = blosum.BLOSUM(62)
        blosum_t = [[]]
        blosum_aa = ["A"]
        for aa in mat.keys():
            if aa[0] in self.aminoacids and aa[1] in self.aminoacids:
                if len(blosum_t[-1]) < 20:
                    blosum_t[-1].append(mat[aa])
                else:
                    blosum_aa.append(aa[0])
                    blosum_t.append([mat[aa]])
        blosum_t = torch.tensor(blosum_t)
        blosum_aa = "".join(blosum_aa)
        return [blosum_t[blosum_aa.index(res)].tolist() for res in peptide]

    def __init__(self, csv_peptides,csv_ba_values, encoder):
        self.aminoacids = ('ACDEFGHIKLMNPQRSTVWY')
        self.csv_peptides = csv_peptides
        self.csv_ba_values = csv_ba_values
        self.ba_values = torch.tensor(self.csv_ba_values, dtype=torch.float64)
        if encoder == "blosum":
            self.peptides = torch.tensor([self.peptide2blosum(p) for p in self.csv_peptides])
        else:
            self.peptides = torch.tensor([self.peptide2onehot(p) for p in self.csv_peptides])
    def __getitem__(self, idx):
        return self.peptides[idx], self.ba_values[idx]
    def __len__(self):
        return len(self.peptides)

def load_reg_data(csv_file, threshold):
    csv_peptides = []
    csv_ba_values = []
    with open(csv_file) as csv_f:
        rows = [row.replace("\n", "").split(",") for row in csv_f]
        for row in rows:
            if float(row[3]) <= threshold:
                csv_peptides.append(row[2])
                csv_ba_values.append(float(row[3]))
    return csv_peptides, csv_ba_values