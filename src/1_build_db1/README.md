### Step 1: Preparing the binding affinity targets
#### 1.1: Building DB1 for MHC-I based on MHCFlurry dataset
DB1 is a text file containing pMHC-I (DB1-I) and pMHC-II (DB1-II) peptide sequences, MHC allele names and their experimental Binding Affinities (BAs).
This step is composed of 1_generate_ids_file_BA.sh and 2_generate_db1_I.sh
#### 1.2: Clustering the peptides based on their sequence similarity
Data clustering is performed with cluster_peptides.py for the peptides and with get_mhci_alleles_clusters.ipynb for the MHC alleles.

```
python src/build_db1/cluster_peptides --file BA_pMHCI.csv --clusters 10
```
* Inputs: generated db1 in `data/external/processed`.
* Output: a .pkl file in `data/external/processed` containing the clusters.
* Run `python src/2_build_db1/cluster_peptides --help` for more details on which matrix to use and have info on the format of the pkl file.
* Visualize the cluster sequence logo as well as the proportion of positive/negative with the `exploration/draw_clusters.ipynb` script.
