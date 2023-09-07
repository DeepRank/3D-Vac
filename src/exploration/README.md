This folder contains scripts for exploring data used in the 3D-Vac project's experiments. In particular, the folder `deeprankcore/` contains scripts for plotting the results obtained training graph/convolutional neural networks with deeprankcore package (data processing scripts are in `src/3_build_db4/deeprankcore/`, training scripts are in `src/4_train_models/deeprankcore/`). Paths specifically refer to our shared cluster on Snellius. Please refer to [deeprankcore documentation](https://deeprankcore.readthedocs.io/en/latest/?badge=latest) for in-depth details about the classes/methods used parameters.

- `data_metafeatures_exploration.ipynb`: general exploration of the data used in the experiments (~100000 data points) from the point of view of targets, peptides' length, alleles, and clusters distribution. 
- `deeprankcore/`
   - `exp_visualization.ipynb`: notebook for exploring and plotting deeprankcore GNNs/CNNs' results (e.g., loss vs epochs, AUC, MCC), for a single experiment.
   - `exps_comparison.ipynb`: notebook for comparing multiple deeprankcore experiments in terms of loss curves and metrics (e.g., AUC).
- `deeprank_deeprankcore_comparison.ipynb`: notebook for comparing deeprank and deeprankcore experiments with sequence-based methods (i.e., re-trained MHCFlurry and MLP).
