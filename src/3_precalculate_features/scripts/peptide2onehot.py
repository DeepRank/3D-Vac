# this script should be runned after map_pssm2pdb.py
import glob
from pdb2sql import pdb2sql
import pickle
from mpi4py import MPI


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

pssm_folders = glob.glob("/projects/0/einf2380/data/pMHCI/pssm_mapped/BA/*/*")
csv_file = "/home/lepikhovd/binding_data/BA_pMHCI.csv"
pssm_template_path = "/projects/0/einf2380/data/pMHCI/pssm_mapped/BA/60520_65418/BA_62160_5EU4/pssm/BA_62160.M.pdb.pssm"
peptide_pdb_sequences = dict()
peptide_sequences = dict()


d3to1 = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N', 
'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W', 
'ALA': 'A', 'VAL':'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M'}

# make the peptide_sequences 
with open(csv_file) as f:
    rows = [row.replace("\n","").split(",") for row in f]
    peptide_sequences = {row[0]:row[2] for row in rows}

# retrieve the first row of the pssm_template to make the template for the pseudo-PSSM
pssm_template = []
with open(pssm_template_path) as template_f:
    rows = [row.replace("\n", "").split() for row in template_f]
    pssm_template = rows[0]

# make the vectors for each peptide following the pssm_template format
step = int(len(peptide_sequences)/size)
start = int(rank*step)
end = int((rank+1)*step)

if rank == 0:
    print(size)
if rank != size-1:
    peptide_batch = dict(list(peptide_sequences.items())[start:end])
else:
    peptide_batch = dict(list(peptide_sequences.items())[start:])

peptides_pssm = []
for idx, (id,sequence) in enumerate(peptide_batch.items()):
    # if idx == 0:
        peptide_pssm_rows = [pssm_template]
        for i,res in enumerate(sequence):
            pdbresi = str(i+1)
            pdbresn = res
            seqresi = pdbresi
            seqresn = pdbresn
            peptide_pssm_row = [pdbresi,pdbresn,seqresi,seqresn,*[str(0)]*21]
            onehot_pos = pssm_template.index(res.strip())
            peptide_pssm_row[onehot_pos] = str(1)
            peptide_pssm_rows.append(peptide_pssm_row) 
        #write the file
        peptide_pssm_path = [path for path in pssm_folders if id in path.split("/")[-1]][0] + "/pssm"
        peptide_pssm_file = glob.glob(f"{peptide_pssm_path}/*")[0].split("/")[-1].replace("M", "P")
        peptide_pssm_complete_path = f"{peptide_pssm_path}/{peptide_pssm_file}";
        print(peptide_pssm_complete_path)
        to_write= "\n".join(["\t".join(row) for row in peptide_pssm_rows])
        with open(peptide_pssm_complete_path, "wb") as peptide_f:
            to_write = to_write.encode("utf8").strip()
            peptide_f.write(to_write)




    

# make the peptide_pdb_sequences:
# for model in pdb_folders:
#     # print(f"file location: {model}")
#     pdb_file = glob.glob(f"{model}/pdb/*.pdb")[0];
#     pdb_id = pdb_file.split("/")[-1].split(".")[0]
#     db = pdb2sql(pdb_file) 
#     residues = {resSeq:resName for resName, resSeq in db.get("resName, resSeq", chainID=["P"])};
#     residues = "".join([d3to1[residue] for residue in residues.values()])
#     peptide_pdb_sequences[pdb_id] = residues


        
#check if no differences between peptide_pdb_sequences and peptide_sequences:
# for id,peptide in peptide_pdb_sequences.items():
#     if (peptide_sequences[id] != peptide):
#         print(f"the peptide sequence of {id} pdb file is not the same as in the csv files")
#         print(f"csv file: {peptide_sequences[id]}, pdb structure: {peptide}")