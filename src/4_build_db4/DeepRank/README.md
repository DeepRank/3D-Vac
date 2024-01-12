DB4 is the collection of HDF5 files with 3D-grids or grpahs containing the featurized complexes.

#### Step 4.1: Populating the features_input_folder.
```
sbatch 1_populate_features_input_folder.sh
```
* The way DeepRank feature generator works for now requires all .pssm and .pdb files to be in the same folder.
* This script creates symlinks for every `db2_selected_models` .pssm and .pdb files into the feature_input_folder
* Run `python src/4_build_db4/populate_features_input_folder.py --help` for more information

#### Step 4.2: building db4
```
sbatch 2_generate_features.sh
```
IMPORTANT NOTE: the path to the .csv with the targets needs to be changed in threshold_classification.py, line 15.

* Build db4 output files into h5out (the path is hardcoded)
* The list of features and targets can be modified inside the file. More information available on https://deeprank.readthedocs.io/en/latest/tutorial2_dataGeneration.html

#### Extra features
Features added on top of the default DeepRank, like the anchor feature, the Desolvation Energy or the skipgram sequence encoding, are stored in this folder as well.