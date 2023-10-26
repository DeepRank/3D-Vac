#### Step 4.1: Split db4 into train, validation and test 10 times for shuffled and clustered CNN dataset
```
sbatch split_h5.sh
```
* To generate the clustered dataset, add `--cluster` argument.
* Add `--help` for more information.

#### Step 4.2: Run CNN on shuffled, peptide-clustered and allele-clustered sets
To train the CNN adapt `submit_3exp_training.sh` and run it with sbatch.

#### Step 4.3: Generate metrics for best CNN model
```
sbatch submit_performances.sh
```
* This script runs cnn_performances.py, which is a custom made script had to be written to obtain metrics from DeepRank's best model. This problem is not present with MLP.
* For a fair comparison between CNN and MLP, only best models are used.
* This step generates metrics on test dataset (clustered and shuffled) from the best model. 
* Add `--help` for more info.
* Add `--cluster` to generate metrics for the clustered model

#### Step 4.4: Perform 10 fold cross-validated MLP training on shuffled and clustered dataset
```
python src/4/train_models/CNN/I/classification/seq/mlp_baseline.py -o mlp_test
```
* Add `--help` for more info.
* Add `--cluster` for clustered dataset.