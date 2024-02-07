# Personalized cancer vaccine design through 3D modelling boosted geometric learning (3D-Vac)

Welcome to the repository for the collaborative project "Personalized Cancer Vaccine Design through 3D Modelling Boosted Geometric Learning," a joint effort between the eScience Center and Radboudumc, as part of the OEC 2021 initiative.

This repository hosts the code utilized in executing the experiments outlined in the paper "Improving Generalizability for MHC-I Binding Peptide Predictions through Structure-Based Geometric Deep Learning", available as a pre-print [here](https://www.biorxiv.org/content/10.1101/2023.12.04.569776v2.abstract).

*Key Notes*

- The 3D pMHC models were generated employing [PANDORA](https://github.com/X-lab-3D/PANDORA).
- Implementation and execution of CNN and GNN models were accomplished using the [DeepRank](https://github.com/DeepRank/deeprank) and [DeepRank2](https://github.com/DeepRank/deeprank2) packages, respectively.

Feel free to explore and utilize the resources provided within this repository. If you have any questions or feedback, please don't hesitate to reach out.

## Table of contents

- [Personalized cancer vaccine design through 3D modelling boosted geometric learning (3D-Vac)](#personalized-cancer-vaccine-design-through-3d-modelling-boosted-geometric-learning-3d-vac)
  - [Table of contents](#table-of-contents)
  - [How to run the pipeline](#how-to-run-the-pipeline)
    - [1: Preparing the binding affinity targets](#1-preparing-the-binding-affinity-targets)
      - [1.1: Building DB1](#11-building-db1)
      - [1.2 Data clustering](#12-data-clustering)
    - [2: Building DB2](#2-building-db2)

## How to run the pipeline

Within the `src/` directory, you can find organized folders labeled by step numbers (e.g., 1, 2, etc.). Each of these folders contains both `.py` and `.sh` scripts. The key scripts intended for submission to the job scheduler are the numbered `.sh` scripts (e.g., `1_generate_ids_file_BA.sh`). These scripts orchestrate the execution of the corresponding `.py` scripts, tailored for specific experiments or modes. In cases where multiple scripts share the same number, they pertain to the same job but cater to different experiments or modes.

For optimal performance, we recommend utilizing adequate resources, preferably GPUs, for running these experiments effectively. Furthermore, the `.sh` scripts provided are designed for working with a [SLURM](https://slurm.schedmd.com/overview.html) workload manager.

Note that you will need to change all the paths in the scripts with your data's paths.

### 1: Preparing the binding affinity targets

The relevant scripts can be found in `src/1_build_db1/`.

#### 1.1: Building DB1

**DB1**: pMHC-I peptide sequences, MHC allele names and their experimental binding affinities (BAs). These data are input of [PANDORA](https://github.com/X-lab-3D/PANDORA).
- It contains TBD data points, quantitative measurements only, from [MHCFlurry 2.0 S3 dataset](https://data.mendeley.com/datasets/zx3kjzc3yx/3).
- Location on TBD: TBD.

Run first: 

```bash
1_generate_ids_file_BA.sh
```

And then: 

```bash
2_generate_db1.sh
```

#### 1.2 Data clustering

Data clustering is performed with `cluster_peptides.py` for the peptides and with `get_mhci_alleles_clusters.ipynb` for the MHC alleles. Note that we included only the alleles' clustering experiments only in the paper. 

For the peptides clustering, run:

```bash 
python src/1_build_db1/cluster_peptides --file BA_pMHCI.csv --clusters 10
```

* Inputs: generated DB1 in `data/external/processed`.
* Output: a .pkl file in `data/external/processed` containing the clusters.
* Run `python src/1_build_db1/cluster_peptides --help` for more details on which matrix to use and have info on the format of the pkl file.
* Visualize the cluster sequence logo as well as the proportion of positive/negative with the `exploration/draw_clusters.ipynb` script.

### 2: Building DB2

The relevant scripts can be found in `src/2_build_db2/`.

**DB2**: Structural 3D models for the pMHC complexes in DB1. These data are output of PANDORA.
- Location on TDB: TDB. This folder contains TBD .pdb models, output of PANDORA (best model for each data point).

Run:

```bash
1_build_db2.sh
```

It takes care of checking which models are missing, distributing computations accross the nodes and cleaning the incomplete outputs at the end.

`modelling_job.py` is implicitly called and it's the actual script taking care of the modelling. To change specific modelling options, like anchors restraints standard deviation, number of models, C domain etc., modify this script.
