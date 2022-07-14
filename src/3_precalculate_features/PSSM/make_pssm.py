from Bio.Align import MultipleSeqAlignment
from Bio import AlignIO
from Bio.Align import AlignInfo

align = AlignIO.read("/projects/0/einf2380/data/pMHCI/msa/hla.fasta", "fasta")
msa = MultipleSeqAlignment(align)
pssm = AlignInfo.SummaryInfo(msa).pos_specific_score_matrix().__str__()

with open("/projects/0/einf2380/data/pMHCI/msa/hla.pssm", "w") as f:
    f.write(pssm)