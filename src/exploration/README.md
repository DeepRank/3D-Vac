This folder contains scripts for exploring data used in the 3D-Vac project's experiments. In particular, the folder `DeepRank2/` contains scripts for plotting the results obtained training graph/convolutional neural networks with deeprankcore package (data processing scripts are in `src/3_build_db4/DeepRank2/`, training scripts are in `src/4_train_models/DeepRank2/`). Paths specifically refer to our shared cluster on Snellius. Please refer to [deeprank2 documentation](https://deeprankcore.readthedocs.io/en/latest/?badge=latest) for in-depth details about the classes/methods used parameters.

- `data_metafeatures_exploration.ipynb`: general exploration of the data used in the experiments (~100000 data points) from the point of view of targets, peptides' length, alleles, and clusters distribution. 
- `DeepRank2/`
   - `exp_visualization.ipynb`: notebook for exploring and plotting deeprank2 GNNs/CNNs' results (e.g., loss vs epochs, AUC, MCC), for a single experiment.
   - `exps_comparison.ipynb`: notebook for comparing multiple deeprank2 experiments in terms of loss curves and metrics (e.g., AUC).
- `deeprank_deeprank2_comparison.ipynb`: notebook for comparing deeprank and deeprank2 experiments with sequence-based methods (i.e., re-trained MHCFlurry and MLP).
