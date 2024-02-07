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
