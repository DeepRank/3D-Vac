# 3D-Vac
Repository of the eScience Center and RadbouduMC "Personalized cancer vaccine design through 3D modelling boosted geometric learning" collaboartive project (OEC 2021).

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
├──src                 <- Source code for use in this project.
│   |   
│   ├── __init__.py     <- Makes src a Python module
│   │
│   ├── 0_build_db1
│   │   ├── generate_db1_I.sh <- Main scripts to generate db1 for MHC-I
│   │   │ 
│   │   ├── generate_db1_II.sh <- Main scripts to generate db1 for MHC-II
│   │   │
│   │   ├── generate_db1_II.py 
│   │   │
│   │   ├── generate_db1_I.py
│   │   │
│   │   ├── cluster_peptides_drb10101.sh 
│   │   │
│   │   ├── cluster_peptides.py
│   │   │
│   │   ├── db1_to_db2_path.py <- Contains function to assign db2 folder per each case. The rest is obsolete and has been replaced by generate_db1_II.sh + generate_db1_subset.py
│   │   │
│   │   └── generate_db1_subset.py <- Generates a subset csv starting from a db1 csv
│   │
│   ├── 1_build_db2
│   │   ├── build_db2_I.sh <- Main script to generate db2 for MHC-I
│   │   │
│   │   ├── build_db2_II.sh <- Main script to generate db2 for MHC-II
│   │   │
│   │   ├── build_db2.py <- Code to generate a db2. Can be used either for MHC-I and -II, and has to be submitted with a bash script
│   │   │
│   │   ├── clean_outputs.py <- cleaning script to be run after generating db2
│   │   │
│   │   ├── clean_outputs.sh <- Runs clean_output.py. NOTE: To be manually run after the modelling.
│   │   │
│   │   ├── allocate_nodes.py <- Decides how many nodes to allocate and starts modelling jobs
│   │   ├── allocate_nodes.sh <- Runs allocate_nodes.py. It is run by build_db2_*.sh
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
│   │   └── run_single_case.py <- Utility to run only one 3D modelling in case one or few are missing
│   │
│   └── visualization   <- Scripts to create exploratory and results oriented visualizations
│       └── visualize.py
└──
```

## How to run the pipeline for the pilot dataset:
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
```
python src/1_build_db2/build_db2.py -i BA_pMHCI.csv --running-time 02
```
* Inputs: generated db1 in `data/external/processed`.
* Output: models in the `models` folder.
* Run `python src/1_build_db2/build_db2.py --help` for more details on how the script works.

### GNNs
- Generate features graphs in the form of .hdf5 files. Run `src/features/pdb_to_hdf5_gnns.py`
- Combine multiple .hdf5 files into one. Run `src/features/combine_hdf5.py`
