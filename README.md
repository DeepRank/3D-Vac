# 3D-Vac
Repository of the eScience Center and Radboudumc "Personalized cancer vaccine design through 3D modelling boosted geometric learning" collaboartive project (OEC 2021).

Note that the 3D pMHC models (which are input for both deeprank and deeprank-gnn-2) have been generated using [PANDORA](https://github.com/X-lab-3D/PANDORA).

## Repository structure

Taking inspiration from [Cookiecutter Data Science](https://drivendata.github.io/cookiecutter-data-science/#starting-a-new-project) the following is the agreed directory structure:

```
├── LICENSE
│
├── README.md           <- The top-level README for developers using this project.
│
├── data
│   │
│   ├── hdf5
│   │   ├── cnn         <- hdf5 files generated with old version of deeprank, for cnns training
│   │   └── gnn         <- hdf5 files generated with new version of deeprank, for gnns training
│   │
│   ├── 3d_models       <- aligned 3d models, output of pandora
│   │
│   ├── external
│   │   ├── processed   <- csv files ready for the modelling
│   │   └── unprocessed <- BA and EL text data (csv)
│   │
│   └── pssm
│       ├── mapped
│       ├── unmapped
│       └── blast_dbs   <- blast databases and MSAs to generate them
│
├── docs                <- For useful documentation
│
├── models              <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks           <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── references          <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports             <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures         <- Generated graphics and figures to be used in reporting
│
├── requirements.txt    <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── .gitignore
│
├──src                             <- Source code for use in this project.
│   |   
│   ├── __init__.py                <- Makes src a Python module
│   │
│   ├── 0_build_db1
│   │   ├── 1_generate_db1_I.sh      <- Main scripts to generate db1 for MHC-I
│   │   │ 
│   │   ├── 1_generate_db1_II.sh     <- Main scripts to generate db1 for MHC-II
│   │   │
│   │   ├── generate_db1_II.py 
│   │   │
│   │   ├── generate_db1_I.py
│   │   │
│   │   ├── cluster_peptides_drb10101.sh 
│   │   │
│   │   ├── cluster_peptides.py
│   │   │
│   │   ├── db1_to_db2_path.py     <- Contains function to assign db2 folder per each case. The rest is obsolete and has been replaced by generate_db1_II.sh + generate_db1_subset.py
│   │   │
│   │   └── generate_db1_subset.py <- Generates a subset csv starting from a db1 csv
│   │
│   ├── 1_build_db2
│   │   ├── 1_build_db2_I.sh         <- Main script to generate db2 for MHC-I
│   │   │
│   │   ├── 1_build_db2_II.sh        <- Main script to generate db2 for MHC-II
│   │   │
│   │   ├── build_db2.py           <- Code to generate a db2. Can be used either for MHC-I and -II, and has to be submitted with a bash script
│   │   │
│   │   ├── clean_outputs.py       <- cleaning script to be run after generating db2
│   │   │
│   │   ├── 2_clean_outputs.sh       <- Runs clean_output.py. NOTE: To be manually run after the modelling.
│   │   │
│   │   ├── allocate_nodes.py      <- Decides how many nodes to allocate and starts modelling jobs
│   │   ├── allocate_nodes.sh      <- Runs allocate_nodes.py. It is run by build_db2_*.sh
│   │   │
│   │   ├── get_unmodelled_cases.py <- Gets how many of the total cases in the db provided have been modelled, and how many still need to be modelled.
│   │   │
│   │   ├── get_unmodelled_cases.sh <- get_unmodelled_cases.py. It is run by build_db2_*.sh
│   │   │
│   │   ├── modelling_job.py <- Actual 3D modelling job containing the PANDORA Wrapper. It is submitted in parallel accross multiple nodes by allocate_nodes.py
│   │   │
│   │   └── modelling_job.sh <- Runs modelling_job.py. It is run by build_db2_*.sh
│   │
│   ├── tools
│   │   ├── clip_C_domain_mhcII.py <- (obsolete) script to clip away the C-domain from all MHC-II generate models
│   │   └── run_single_case.py     <- Utility to run only one 3D modelling in case one or few are missing
│   │
│   └── visualization              <- Scripts to create exploratory and results oriented visualizations
│       └── visualize.py
└──
```

## Databases

In the code and in the repo we often refer to numbered databases. Data can refer to either pMHC class I complexes, or pMHC class II complexes. The suffix is -I in the first case and -II in the latter. 

- **DB1** - Sequences of pMHC and their experimental Binding Affinities (BAs). These data are input of [PANDORA](https://github.com/X-lab-3D/PANDORA).
- **DB2** - Structural 3D models for pMHC in DB1. These data are output of PANDORA.
- **DB3** - PSSMs for pMHC. These data are derived from BLAST database.
- **DB4** - Interface grids for CNNs (deeprank) or interface graphs for GNNs (deeprankcore) in the form of hdf5 files. DB3 and DB2 are used the generation of DB4.

## How to run the pipeline:
### Scripts Guidelines:
In general, the folders are ordered per ste number (0, 1, 2, etc.). Every folder contains both `.py` and `.sh` scripts that do not need to be manually submitted. The only scripts that need to be submitted, and eventually changed depending on the experiment, are the scripts ordered by number (e.g. `1_build_db2_II.sh`). When multiple scripts have the same number, they refer to the same job but for different experiments / mhc-class / mode. (e.g. `1_build_db2_I.sh` and `1_build_db2_II.sh` ), so only one should be run depending on the expeirment.

If you perform a new experiment, please use a new `.sh` script and write a comment in it explaining what it does (i.e. what it does differently from the other identical scripts, like "Generates db2 only for HLA-C").

### Step 0: Preparing the binding affinity targets
#### 0.1: Building DB1 for MHC-I based on MHCFlurry dataset
DB1 contains all sequences of pMHC-I (DB1-I) and pMHC-II (DB1-II) and their experimental Binding Affinities (BAs).
```
python src/0_build_db1/generate_db1_I.py --source-csv curated_training_data.no_additional_ms.csv --P 9 --allele HLA-A*02:01 --output-csv BA_pMHCI.csv
```
* Inputs: MHCFlurry dataset csv filename in `data/external/unprocessed`.
* Outputs: DB1 in 'path-to-destination.csv'.
* Run `python src/0_build_db1/generate_db1_I.py --help` for more details on how to filter for specific allele and peptide length.

#### 0.2: Clustering the peptides based on their sequence similarity
```
python src/build_db1/cluster_peptides --file BA_pMHCI.csv --clusters 10
```
* Inputs: generated db1 in `data/external/processed`.
* Output: a .pkl file in `data/external/processed` containing the clusters.
* Run `python src/0_build_db1/cluster_peptides --help` for more details on which matrix to use and have info on the format of the pkl file.
* Visualize the cluster sequence logo as well as the proportion of positive/negative with the `exploration/draw_clusters.ipynb` script.

### Step 1: Generating pdb structures
#### 1.1: Building DB2 from DB1
DB2 contains structural 3D models (output of Pandora, pdb structures) for: pMHC-I in DB1-I (DB2-I), and pMHC-II in DB1-II (DB2-II). DB1-I and DB1-II are input of Pandora. 
The following example line, that generates DB2-I, is used in `build_db2_I.sh`:
```
python src/1_build_db2/build_db2.py -i BA_pMHCI.csv --running-time 02
```
* Inputs: generated db1 in `data/external/processed`.
* Output: models in the `models` folder.
* Run `python src/1_build_db2/build_db2.py --help` for more details on how the script works.
Output folder structure (after cleaning with clean_outputs.sh):

```
│── <target_id>_<template_id>
│   │
│   ├── <template_id>.pdb           Template pdb file used for the modelling
│   │
│   ├── molpdf_DOPE.tsv             Ranking all models by molpdf and DOPE modeller's scoring functions
│   │
│   ├── <target_id>.BL00??0001.pdb  Final models
│   │
│   ├── modeller.log                Printing log file generated by MODELLER, describing modelling steps, or any issues arose along modelling
│   │
│   ├── *.ali                       Alignment file between template(s) and target used for modelling
│   │
│   ├── contacts_*.list             Contact restraints
│   │
│   ├── MyLoop.py                   MODELLER script to set loop modelling parameters for the peptide
│   │
│   ├── cmd_modeller_ini.py         MODELLER script to generate an initial model to extract restraints from
│   │
│   ├── cmd_modeller.py             MODELLER script to set the main modelling parameters
│   │
│   ├── *.ini                       Initial model generated placing the target atoms at the same coordinate as the template's atoms. This preceeds the IL model.
│   │
│   ├── *IL*.pdb                    Initial loop model based on the .ini model. Might be marked as best model when the target is identicatl to a template
│
└──
```

### Step 2: Generating db3
#### 2.1: Selecting which PANDORA-generated 3D-models to use
```
sbatch 1_copy_3Dmodels_from_db2.sh
```
* PANDORA generates 20 pdb structures per cases. They are ranked based on the global energy of the complex.
* The first 5 pdb in this ranking contain the most plausible structure.
* For now, only the first structure is being used. The script `src/2_build_db3/symlink_targets_from_db2.py` is written in a way that it will be possible to select more than 1 structure in the future.
* Run `python src/2_build_db3/copy_3Dmodels_from_db2.py --help` for more information on how the script works.

#### 2.2: Aligning structures
```
sbatch 2_align_pdb.sh
```
* Aligns every structures to one template.
* Add `--help` to see additional information.

#### 2.3: Build PSSM for M chain (MHC protein) and pseudo-PSSM encoding for the P chain (peptide)
##### 2.3.1: Build the blast database
* Make sure `blast` is installed. The easiest way to install it is with conda:
  ```
  conda install -c bioconda blast
  ```
  This might not work on Snellius. In that case, download and extract the package from https://ftp.ncbi.nlm.nih.gov/blast/executables/LATEST/, add the `bin` folder to your `PATH`. Make sure also `psiblast` is in your PATH and callable by terminal.
* Copy the human MHC sequence fasta file from `/<PANDORA_installation_folder>/Databases/default/mhcseqs/hla_prot.fasta` into `data/pssm/blast_dbs/`.
  
Run: 
```
sbatch 3_build_blastdb.sh
```

##### 2.3.2: Calculate raw PSSM for M chain:
```
sbatch 4_create_raw_pssm.sh
```
* Run `python src/2_build_db3/create_raw_pssm.py --help` for more information.

##### 2.3.3: Map generated raw PSSM to the PDB:
```
sbatch 5_map_pssm2pdb.sh
```
* Mapping raw PSSM to the pdb alleviate problems such as gaps in sequences.
* Only mapped PSSM for the M chain are used to generate the PSSM db3 feature.

##### 2.3.4: Generate fake PSSM for the peptide
```
sbatch 6_peptide2onehot.sh
```
* Run `python src/2_build_db3/peptide2onehot.py --help` for more information.


### Step 3: Generating db4
#### Step 3.1: Populating the features_input_folder.
```
python src/3_build_db4/populate_features_input_folder.py
```
* The way DeepRank feature generator works for now requires all .pssm and .pdb files to be in the same folder.
* This script creates symlinks for every `db2_selected_models` .pssm and .pdb files into the feature_input_folder
* Run `python src/3_build_db4/populate_features_input_folder.py --help` for more information

#### Step 3.2: building db4
```
python src/3_build_db4/generate_features.py
```
* Build db4 output files into h5out (the path is hardcoded)
* The list of features and targets can be modified inside the file. More information available on https://deeprank.readthedocs.io/en/latest/tutorial2_dataGeneration.html

### Step 4: Training MLP and CNN models
#### Step 4.1: Split db4 into train, validation and test 10 times for shuffled and clustered CNN dataset
```
python src/4_train_models/CNN/split_h5.py
```
* To generate the clustered dataset, add `--cluster` argument.
* Add `--help` for more information.

#### Step 4.2: Perform 10 fold cross-validated CNN training on shuffled and clustered dataset
```
python src/4_train_models/CNN/I/classification/struct/cnn_baseline.py -o cnn_test
```
* Add `--cluster` to train CNN on clustered dataset.
* Add `--help` for detailed information and available parameters.

#### Step 4.3: Generate metrics for best CNN model
```
python src/4/train_models/CNN/I/classification/struct/cnn_performances.py -o cnn_test
```
* Custom made script had to be written to obtain metrics from DeepRank's best model. This problem is not present with MLP.
* For a fair comparison between CNN and MLP, only best models are used.
* This step generates metrics on test dataset (clustered and shuffled) from the best model. 
* Add `--help` for more info.
* Add `--cluster` to generate metrics for the clustered model

#### Step 4.4: Perform 10 fold cross-validated MLP training on shuffled and clustered dataset
```
python src/4/train_models/CNN/I/classification/seq/mlp_baseline.py -o mlp_test
```
* Add `--help` for more info.
* Add `--cluster` for clustered dataset.

### GNNs
- Generate features graphs in the form of .hdf5 files. Run `src/features/pdb_to_hdf5_gnns.py`
- Combine multiple .hdf5 files into one. Run `src/features/combine_hdf5.py`
