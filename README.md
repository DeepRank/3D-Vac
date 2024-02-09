# Personalized cancer vaccine design through 3D modelling boosted geometric learning (3D-Vac)

Welcome to the repository for the collaborative project "Personalized Cancer Vaccine Design through 3D Modelling Boosted Geometric Learning," a joint effort between the [eScience Center](https://www.esciencecenter.nl/) and Radboudumc, as part of the OEC 2021 initiative.

This repository hosts the code utilized in executing the experiments outlined in the paper "Improving Generalizability for MHC-I Binding Peptide Predictions through Structure-Based Geometric Deep Learning", available as a pre-print [here](https://www.biorxiv.org/content/10.1101/2023.12.04.569776v2.abstract).

*Key Notes*

- The 3D pMHC-I models were generated employing [PANDORA](https://github.com/X-lab-3D/PANDORA).
- Implementation and execution of CNN and GNN models were accomplished using the [DeepRank](https://github.com/DeepRank/deeprank) and [DeepRank2](https://github.com/DeepRank/deeprank2) packages, respectively.

Feel free to explore and utilize the resources provided within this repository. If you have any questions or feedback, please don't hesitate to reach out.

## Table of contents

- [Personalized cancer vaccine design through 3D modelling boosted geometric learning (3D-Vac)](#personalized-cancer-vaccine-design-through-3d-modelling-boosted-geometric-learning-3d-vac)
  - [Table of contents](#table-of-contents)
  - [How to run the pipeline](#how-to-run-the-pipeline)
    - [1: DB1](#1-db1)
    - [2: DB2](#2-db2)
    - [3: DB3](#3-db3)
      - [3.1: Selecting the PANDORA-generated 3D-models](#31-selecting-the-pandora-generated-3d-models)
      - [3.2: Aligning structures](#32-aligning-structures)
    - [4: DB4](#4-db4)
      - [4.1: 3D-grids](#41-3d-grids)
      - [4.2: Graphs](#42-graphs)
    - [5: Training](#5-training)
      - [5.1: CNN](#51-cnn)
      - [5.2: GNN](#52-gnn)
      - [5.3: EGNN](#53-egnn)
      - [5.4: SSL](#54-ssl)

## How to run the pipeline

Within the `src/` directory, you can find organized folders labeled by step numbers (e.g., 1, 2, etc.). Each of these folders contains both `.py` and `.sh` scripts. The key scripts intended for submission to the job scheduler are the numbered `.sh` scripts (e.g., `1_generate_ids_file_BA.sh`). These scripts orchestrate the execution of the corresponding `.py` scripts, tailored for specific experiments or modes. In cases where multiple scripts share the same number, they pertain to the same job but cater to different experiments or modes.

For optimal performance, we recommend utilizing adequate resources, preferably GPUs, for running these experiments effectively. Furthermore, the `.sh` scripts provided are designed for working with a [SLURM](https://slurm.schedmd.com/overview.html) workload manager.

Note that you will need to change all the paths in the scripts with your data's paths.

### 1: DB1

DB1 contains pMHC-I peptide sequences, MHC allele names and their experimental binding affinities (BAs). These data are input of [PANDORA](https://github.com/X-lab-3D/PANDORA). It contains TBD data points, quantitative measurements only, from [MHCFlurry 2.0 S3 dataset](https://data.mendeley.com/datasets/zx3kjzc3yx/3). DB1 can be found TBD. The scripts for generating DB1 can be found in `src/1_build_db1/`.

- First prepare the binding affinity target: `sbatch 1_generate_ids_file_BA.sh`.
- Then generate the dataset: `sbatch 2_generate_db1.sh`.
- You can perform data clustering with `cluster_peptides.py` for the peptides and with `get_mhci_alleles_clusters.ipynb` for the MHC alleles. Note that we included only the alleles' clustering experiments only in the paper. For the peptides clustering, run: `python cluster_peptides --file BA_pMHCI.csv --clusters 10`.
  - Inputs: generated DB1 in `data/external/processed`.
  - Output: a .pkl file in `data/external/processed` containing the clusters.
  - Run `python cluster_peptides --help` for more details on which matrix to use and have info on the format of the pkl file.
  - Visualize the cluster sequence logo as well as the proportion of positive/negative with the `exploration/draw_clusters.ipynb` script.

### 2: DB2

DB2 contains structural 3D models for the pMHC complexes in DB1. These data are output of PANDORA. It contains TBD PDB models, output of PANDORA. DB2 can be found TBD. The scripts for generating DB2 can be found in `src/2_build_db2/`.

- To build DB2, run: `sbatch 1_build_db2.sh`.
  - It takes care of checking which models are missing, distributing computations accross the nodes and cleaning the incomplete outputs at the end. `modelling_job.py` is implicitly called and it's the actual script taking care of the modelling. To change specific modelling options, like anchors restraints standard deviation, number of models, C domain etc., modify this script.
  - More details about the ouput folder structure can be found [here](https://github.com/DeepRank/3D-Vac/blob/paper/src/2_build_db2/README.md). 

### 3: DB3

DB3 contains the selected 3D models and their PSSMs. Note that PSSM features have not been used in the final version of the project, but you can find details about how to compute them [here](https://github.com/DeepRank/3D-Vac/blob/paper/src/3_build_db3/README.md). It contains TBD PDB models, output of PANDORA (best model only for each data point). DB3 can be found on TBD. The scripts for generating DB3 can be found in `src/3_build_db3/`.

#### 3.1: Selecting the PANDORA-generated 3D-models

- For selecting the PANDORA-generated 3D-models, run: `sbatch 1_copy_3Dmodels_from_db2.sh`
  - PANDORA generates 20 PDB structures per cases. They are ranked based on the global energy of the complex. The first 5 PDB in this ranking contain the most plausible structures.
  - For now, only the first structure is being used. The script `copy_3Dmodels_from_db2.py` is written in a way that it will be possible to select more than 1 structure in the future.
  - Run `python copy_3Dmodels_from_db2.py --help` for more information on how the script works.

#### 3.2: Aligning structures

- For aligning the structures, run: `sbatch 4_align_pdb.sh`.
  - It aligns every structures to one template.
  - Add `--help` to see additional information.

### 4: DB4

DB4 is the collection of HDF5 files with 3D-grids or graphs containing the featurized complexes. DB3 and DB2 are used the generation of DB4. It contains TBD data points, for both 3D-grids and graphs.

#### 4.1: 3D-grids

[DeepRank](https://github.com/DeepRank/deeprank) software package was used for generating and featurizing 3D-grids. Please refer to [deeprank documentation](https://deeprank.readthedocs.io/en/latest/?badge=latest) for in-depth details about how to install the package, and its classes/methods used parameters. The 3D-grids HDF5 files generated via DeepRank can be found TBD. The scripts for processing the PDB files of the pMHC complexes into 3D-grids can be found in `src/4_build_db4/DeepRank`. 

- First populate the `features_input_folder`, by running: `sbatch 1_populate_features_input_folder.sh`.
  - The way DeepRank feature generator works for now requires all PSSM and PDB files to be in the same folder. This script creates symlinks for every `db2_selected_models` PSSM and PDB files into the `feature_input_folder`.
  - Run `python populate_features_input_folder.py --help` for more information
- Then, after having successfully installed DeepRank, you can generate the 3D-grids and store them into HDF5 files by running: `sbatch 2_generate_features.sh`.
  - Note that the path to the CSV with the targets needs to be changed in `threshold_classification.py`, line 15.
  - In `src/4_build_db4/DeepRank` you can find additional features not present in DeepRank, like the anchor feature, the Desolvation Energy or the skipgram sequence encoding.

#### 4.2: Graphs

[DeepRank2](https://github.com/DeepRank/deeprank2) software package was used for generating and featurizing graphs. Please refer to [deeprank2 documentation](https://deeprank2.readthedocs.io/en/latest/?badge=latest) for in-depth details about how to install the package, and its classes/methods used parameters. The graphs HDF5 files generated via DeepRank2 can be found TDB. The scripts for processing the PDB files of the pMHC complexes into graphs can be found in `src/4_build_db4/DeepRank2`.

- After having successfully installed DeepRank2, you can generate the graphs and store them into HDF5 files by running: `sbatch 1_generate_features.sh`. All the parameters are set at the beginning of `1_generate_features.py`.
- For adding targets into the generated HDF5 files (e.g., alleles' clusters) reading them in from CSV files, you can run: `sbatch add_targets.sh`.
- For more details about the other scripts in the folder, see [here](https://github.com/DeepRank/3D-Vac/blob/paper/src/4_build_db4/DeepRank2/README.md). 

### 5: Training

#### 5.1: CNN

[DeepRank](https://github.com/DeepRank/deeprank) software package was used for this scope. Please refer to [deeprank documentation](https://deeprank.readthedocs.io/en/latest/?badge=latest) for in-depth details about how to install the package, and its classes/methods used parameters. The pre-trained model and outputs can be found TBD. The scripts for training the CNN can be found in `src/5_train_models/DeepRank`.

- First split DB4 (the 3D-grids) into train, validation and test 10 times for shuffled and clustered CNN dataset: `sbatch 1_split_h5.sh`.
  - To generate the clustered dataset, add `--cluster` argument.
  - Add `--help` for more information.
- Then you can train the CNN on shuffled, peptide-clustered and allele-clustered sets by running: `sbatch 2_submit_2exp_training.sh`.
  - Note that we included only the alleles' clustering experiments only in the paper. 
- Generate metrics for the best CNN model with: `sbatch 3_submit_performances.sh`. 
  - This script runs cnn_performances.py, which is a custom made script had to be written to obtain metrics from DeepRank's best model.
  - This step generates metrics on test dataset (clustered and shuffled) from the best model.
  - Add `--help` for more info.
  - Add `--cluster` to generate metrics for the clustered model

#### 5.2: GNN

#### 5.3: EGNN

#### 5.4: SSL

TBD