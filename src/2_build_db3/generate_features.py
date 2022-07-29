from deeprank.generate import *
from mpi4py import MPI
import h5py

comm = MPI.COMM_WORLD

input_folder = "/projects/0/einf2380/data/pMHCI/features_input_folder"
pdb_folder = [f"{input_folder}/pdb"]
pssm_folder = f"{input_folder}/pssm"
h5out = "/projects/0/einf2380/data/pMHCI/features_output_folder/hla_a_02_01_9_length_peptide/hla_a_02_01_9_length_peptide.hdf5"

database = DataGenerator(
    pdb_source=pdb_folder,
    pssm_source = pssm_folder,
    chain1="M",
    data_augmentation=None,
    chain2="P",
    compute_features=[
        "deeprank.features.AtomicFeature",
        "deeprank.features.FullPSSM",
        "deeprank.features.BSA",
        "deeprank.features.ResidueDensity",
        "edesolv_feature"
    ],
    compute_targets=[
        "threshold_classification"
    ],
    hdf5=h5out,
    mpi_comm = comm
)

database.create_database(prog_bar=True)

# mapping features:
grid_info = {
    "number_of_points": [35,30,30],
    # "number_of_points": [1,1,1],
    "resolution": [1.,1.,1.]
}

database.map_features(grid_info, try_sparse = True, prog_bar=True)
