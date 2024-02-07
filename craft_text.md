### Step 4: Training MLP and CNN models

### GNN folder
#### 5.1 Explore features through features_exploration.ipynb
#### 5.2 Train a network
```
sbatch training.sh
```
* Modify variables at the top of `training.py` script, if needed. 
#### 5.3 Visualize results throught exp_visualization.ipynb

## Exploration
### draw_cluster_motifs.ipynb
* Enables visualization of sequence motifs in clusters of peptides generated using `src/0_build_db1/cluster_peptides.py`.
* Gives the number of **unique** peptides as well as the distribution of binders/non binders for each cluster.

### draw_grid.py
* Create a .vmd file to visualize the grid at the interface of a given case id in hdf5 file.
* Run `src/exploration/draw_grid.py --help` for more information.

### explore_class_seq_xvalidation.ipynb
* Visualize performances of the MLP on clustered and shuffled dataset.
* Open the file for instructions.

### explore_class_struct_xvalidaiton.ipynb
* Visualize performances of the CNN on clustered and shuffled dataset.
* Open the file for instructions.

### explore_best_models.ipynb
* Plots metrics from CNN and MLP best models.
* Open the notebook file for instructions.

### Tools
#### Gibbs cluster
```
python tools/gibbs_cluster_parser.py --help
```
* This tool is used to generate clusters using Gibbs sampling based on Shannon's entropy minimization. 
* Can be found at https://services.healthtech.dtu.dk/services/GibbsCluster-2.0/
* Clusters can be generated directly on the website or by downloading the binary. Optimal parameters for MHCI and MHCII can be checked on the website too.
* Given some parameters for the gibbs cluster binary, each run generates a cluster set made of *g* number of clusters. Next lines list available scripts to explore data distribution, cluster quality of the newly generated cluster set and map clusters for each peptide in db1.
* Once clusters are generated, peptides from db1 can be mapped to their respective cluster using the script in `tools/gibbs_cluster_parser.py`.
* `/src/exploration/explore_gibbs_output.ipynb` can be used to evaluate the clustering quality (assess noisy clusters) and plot binders/non-binders distribution. 
* `/src/exploration/explore_gibbs_kld.ipynb` calculates and plots Kullback-Leibler divergence (KLD score) between clusters (one2one and one2many). This script generates figures in `/reports/figures/gibbs-cluster` showing barplots of KLD scores for one2one or one2many cluster to cluster(s) comparison.

### GNNs
- Generate features graphs in the form of .hdf5 files. Run `src/features/pdb_to_hdf5_gnns.py`
- Combine multiple .hdf5 files into one. Run `src/features/combine_hdf5.py`