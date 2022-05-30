# 3D-Vac
Repository of the eScience Center and RadbouduMC "Personalized cancer vaccine design through 3D modelling boosted geometric learning" collaboartive project (OEC 2021).

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
├── src                 <- Source code for use in this project.
│   ├── __init__.py     <- Makes src a Python module
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
│   │
│   ├── tools
│   │   └── other_utils.py
│   │
│   └── visualization   <- Scripts to create exploratory and results oriented visualizations
│       └── visualize.py
└──
```

## Workflow
Follow these steps to replicate the work
### GNNs
- Generate features graphs in the form of .hdf5 files. Run `src/features/generate_feature_graph.py`
- Combine multiple .hdf5 files into one. Run `src/features/pbd_to_hdf5_gnns.py`
