# Personalized cancer vaccine design through 3D modelling boosted geometric learning (3D-Vac)

Welcome to the repository for the collaborative project "Personalized Cancer Vaccine Design through 3D Modelling Boosted Geometric Learning," a joint effort between the eScience Center and Radboudumc, as part of the OEC 2021 initiative.

This repository hosts the code utilized in executing the experiments outlined in the paper "Improving Generalizability for MHC-I Binding Peptide Predictions through Structure-Based Geometric Deep Learning", available as a pre-print [here](https://www.biorxiv.org/content/10.1101/2023.12.04.569776v2.abstract).

*Key Notes*

- The 3D pMHC models were generated employing [PANDORA](https://github.com/X-lab-3D/PANDORA).
- Implementation and execution of CNN and GNN models were accomplished using the [DeepRank](https://github.com/DeepRank/deeprank) and [DeepRank2](https://github.com/DeepRank/deeprank2) packages, respectively.

Feel free to explore and utilize the resources provided within this repository. If you have any questions or feedback, please don't hesitate to reach out.

- [Personalized cancer vaccine design through 3D modelling boosted geometric learning (3D-Vac)](#personalized-cancer-vaccine-design-through-3d-modelling-boosted-geometric-learning-3d-vac)
  - [How to run the pipeline](#how-to-run-the-pipeline)
    - [1: Preparing the binding affinity targets](#1-preparing-the-binding-affinity-targets)
      - [1.1: Building DB1 for MHC-I based on MHCFlurry dataset](#11-building-db1-for-mhc-i-based-on-mhcflurry-dataset)
    - [1.2 (option 2) Cluster the peptides based on their sequence similarity](#12-option-2-cluster-the-peptides-based-on-their-sequence-similarity)
      - [2\_build\_db2](#2_build_db2)
    - [Step 2: Generating db3](#step-2-generating-db3)
    - [Step 3: Generating db4](#step-3-generating-db4)
      - [3\_build\_db3](#3_build_db3)
      - [4\_build\_db4](#4_build_db4)
    - [Step 4: Training MLP and CNN models](#step-4-training-mlp-and-cnn-models)
    - [GNN folder](#gnn-folder)
      - [5.1 Explore features through features\_exploration.ipynb](#51-explore-features-through-features_explorationipynb)
      - [5.2 Train a network](#52-train-a-network)
      - [5.3 Visualize results throught exp\_visualization.ipynb](#53-visualize-results-throught-exp_visualizationipynb)
  - [Exploration](#exploration)
    - [draw\_cluster\_motifs.ipynb](#draw_cluster_motifsipynb)
    - [draw\_grid.py](#draw_gridpy)
    - [explore\_class\_seq\_xvalidation.ipynb](#explore_class_seq_xvalidationipynb)
    - [explore\_class\_struct\_xvalidaiton.ipynb](#explore_class_struct_xvalidaitonipynb)
    - [explore\_best\_models.ipynb](#explore_best_modelsipynb)
    - [Tools](#tools)
      - [Gibbs cluster](#gibbs-cluster)
    - [GNNs](#gnns)

## How to run the pipeline

Within the `src/` directory, you can find organized folders labeled by step numbers (e.g., 1, 2, etc.). Each of these folders contains both `.py` and `.sh` scripts. The key scripts intended for submission to the job scheduler are the numbered `.sh` scripts (e.g., `1_generate_ids_file_BA.sh`). These scripts orchestrate the execution of the corresponding `.py` scripts, tailored for specific experiments or modes. In cases where multiple scripts share the same number, they pertain to the same job but cater to different experiments or modes.

For optimal performance, we recommend utilizing adequate resources, preferably GPUs, for running these experiments effectively. Furthermore, the `.sh` scripts provided are designed for working with a [SLURM](https://slurm.schedmd.com/overview.html) workload manager.
