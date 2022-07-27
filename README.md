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
|   ├── __init__.py     <- Makes src a Python module
│   │
│   ├── data            <- Scripts to download or generate data
│   │   ├── make_dataset.py
│   │   └── utils.py
│   │
│   ├── features        <- Scripts to turn raw data into features for modeling
│   │   ├── build_features.py
│   │   └── utils.py
│   │
│   ├── models          <- Scripts to train models and then use trained models to make
│   │   │                 predictions
│   │   ├── predict_model.py
│   │   ├── train_model.py
│   │   └── utils.py
|   |
|   ├── pilot_study     <- Structure available in the README of the folder
│   │
│   ├── tools
│   │   └── other_utils.py
│   │
│   └── visualization   <- Scripts to create exploratory and results oriented visualizations
│       └── visualize.py
└──
```

## How to run the pipeline for the pilot dataset:
### Step 0: Preparing the binding affinity targets
#### 0.1: Building DB1 for MHC-I based on MHCFlurry dataset
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
```
python src/1_build_db2/build_db2.py -i BA_pMHCI.csv --running-time 02
```
* Inputs: generated db1 in `data/external/processed`.
* Output: models in the `models` folder.
* Run `python src/1_build_db2/build_db2.py --help` for more details on how the script works.

### Step 2: Generating db3
#### 1.1: Building the blast database
* Make sure `blast` is installed. Download and extract the package from https://ftp.ncbi.nlm.nih.gov/blast/executables/LATEST/, add the `bin` folder to your `PATH`.
* Copy the human MHC sequence fasta file from `/PANDORA_installation_folder/PANDORA_files/data/csv_pkl_files/hla_prot.fasta` into `data/pssm/blast_dbs/`.
* Run `src/2_build_db3/PSSM/build_blastdb.sh`.

### GNNs
- Generate features graphs in the form of .hdf5 files. Run `src/features/pdb_to_hdf5_gnns.py`
- Combine multiple .hdf5 files into one. Run `src/features/combine_hdf5.py`
