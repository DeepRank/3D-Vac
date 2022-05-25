import glob
import os
import pickle

input_folder = "/projects/0/einf2380/data/pMHCI/features_input_folder"
pdb_folder = f"{input_folder}/pdb"
pssm_folder = f"{input_folder}/pssm"

# pkl_f = open("./features_input_files.pkl", "rb")
# data = pickle.load(pkl_f)
# pkl_f.close()

print("globing pdb_files")
pdb_files = glob.glob('/projects/0/einf2380/data/pMHCI/pssm_mapped/BA/*/*/pdb/*.pdb')
# pdb_files = data["pdb_files"]
print("globing pssm_files")
pssm_files = glob.glob('/projects/0/einf2380/data/pMHCI/pssm_mapped/BA/*/*/pssm/*.pssm')
# pssm_files = data["pssm_files"]

#pkl stuff:
# pkl_f = open("./features_input_files.pkl", "wb")
# pickle.dump({"pdb_files":pdb_files, "pssm_files":pssm_files}, pkl_f)



print(f"pdb_files len: {len(pdb_files)}")
print(f"pssm_files len: {len(pssm_files)}")

print("creating symlinks for pdbs:")
for i,pdb in enumerate(pdb_files):
    # if i == 0:
        pdb_file_name = pdb.split("/")[-1]
        dest = f"{pdb_folder}/{pdb_file_name}"
        print(dest)
        try:
            os.symlink(pdb, dest)
        except OSError as error:
            print(error)

print("creating symlinks for pssms:")
for i,pssm in enumerate(pssm_files):
    # if i == 0:
        pssm_file_name = pssm.split("/")[-1]
        dest = f"{pssm_folder}/{pssm_file_name}"
        print(dest)
        try:
            os.symlink(pssm, dest)
        except OSError as error:
            print(error)