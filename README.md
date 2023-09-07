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
DB1 is a text file containing pMHC-I (DB1-I) and pMHC-II (DB1-II) peptide sequences, MHC allele names and their experimental Binding Affinities (BAs).

### Step 1: Generating pdb structures
DB2 contains 3D models (output of Pandora, pdb structures) for: pMHC-I in DB1-I (DB2-I), and pMHC-II in DB1-II (DB2-II). DB1-I and DB1-II are used as input for Pandora. 

All the scripts needed to generate the 3D models are in this folder.

### Step 2: Generating db3
DB3 is the collection of data needed to generate hdf5 files, namely selected 3D models and their PSSMs.
PSSM features have not been used in the final version of the project, so only steps 1 and 4 are necessary.

### Step 3: Generating db4
DB4 is the collection of HDF5 files with 3D-grids or grpahs containing the featurized complexes.

### Step 4: Training MLP and CNN models

### Exploration
#### draw_cluster_motifs.ipynb
* Enables visualization of sequence motifs in clusters of peptides generated using `src/0_build_db1/cluster_peptides.py`.
* Gives the number of **unique** peptides as well as the distribution of binders/non binders for each cluster.

#### draw_grid.py
* Create a .vmd file to visualize the grid at the interface of a given case id in hdf5 file.
* Run `src/exploration/draw_grid.py --help` for more information.

#### explore_class_seq_xvalidation.ipynb
* Visualize performances of the MLP on clustered and shuffled dataset.
* Open the file for instructions.

#### explore_class_struct_xvalidaiton.ipynb
* Visualize performances of the CNN on clustered and shuffled dataset.
* Open the file for instructions.

#### explore_best_models.ipynb
* Plots metrics from CNN and MLP best models.
* Open the notebook file for instructions.

### GNNs
- Generate features graphs in the form of .hdf5 files. Run `src/features/pdb_to_hdf5_gnns.py`
- Combine multiple .hdf5 files into one. Run `src/features/combine_hdf5.py`
