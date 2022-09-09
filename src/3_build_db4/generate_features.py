from deeprank.generate import *
from mpi4py import MPI
import h5py
import argparse

arg_parser = argparse.ArgumentParser(
    description="""
    Generate DeepRank 3D-grid HDF5 files
    """
)
arg_parser.add_argument("--input-folder", "-i",
    help="""
    Path to the input folder
    """,
    default="/projects/0/einf2380/data/pMHCI/features_input_folder"
)
arg_parser.add_argument("--h5out", "-o",
    help="""
    Output hdf5 file
    """,
    default="/projects/0/einf2380/data/pMHCI/features_output_folder/hla_a_02_01_9_length_peptide/hla_a_02_01_9_length_peptide.hdf5"
)

a = arg_parser.parse_args()

comm = MPI.COMM_WORLD

pdb_folder = [f"{a.input_folder}/pdb"]
pssm_folder = f"{a.input_folder}/pssm"

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
    hdf5=a.h5out,
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
