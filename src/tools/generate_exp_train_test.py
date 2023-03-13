"""For each of the following experiments from all human binding affinity measurements (inequalities excluded):
- Randomly shuffled
- Clustered based on peptide sequences
- Clustered based on allele pseudosequences (MHCflurry 2.0 and NetMHCpan 4.1 allele encoding)
Corresponding CSV file is filtered for alleles compatible with the MHCflurry train pan allele command.
2 sets of csv are created: train_validation.csv (containing train and validation cases) as well as test.csv (containing test cases).
CSV generated ensures that the same test data is used for each experiments.
For the randomly shuffled experiment, the test cases consists of either 25% or 100 randomly selected measurements (whichever is the lower) for each allele present in the train.
"""
import pandas as pd
from mhcflurry.train_pan_allele_models_command import assign_folds

# LOAD DATA:
shuffled_csv_path = "../../data/external/processed/all_hla_pseudoseq.csv"
shuffled_df = pd.read_csv(shuffled_csv_path)
mhcflurry_path = "/projects/0/einf2380/data/external/processed/I/mhcflurry_train.csv.bz2"
mhcflurry_df = pd.read_csv(mhcflurry_path)

peptide_clustered_csv_path = "../../data/external/processed/all_hla_pseudoseq.csv"
peptide_clustered_df = pd.read_csv(peptide_clustered_csv_path)

pseudoseq_clustered_csv_path = "/projects/0/einf2380/data/external/processed/I/clusters/BA_pMHCI_human_quantitative_only_eq_alleleclusters_pseudoseq.csv"
pseudoseq_clustered_df = pd.read_csv(pseudoseq_clustered_csv_path)

# FILTER FOR SUPPORTED ALLLELES:
shuffled_df = shuffled_df.loc[
    (shuffled_df.peptide.str.len() <= 15) & (shuffled_df.peptide.str.len() >= 8)
]
shuffled_df = shuffled_df.loc[shuffled_df.allele.isin(mhcflurry_df.allele)]

peptide_clustered_df = peptide_clustered_df.loc[
    (peptide_clustered_df.peptide.str.len() <= 15) &
    (peptide_clustered_df.peptide.str.len() >= 8)
]
peptide_clustered_df = peptide_clustered_df.loc[
    peptide_clustered_df.allele.isin(mhcflurry_df.allele)
]

pseudoseq_clustered_df = pseudoseq_clustered_df.loc[
    (pseudoseq_clustered_df.peptide.str.len() <= 15) &
    (pseudoseq_clustered_df.peptide.str.len() >= 8)
]
pseudoseq_clustered_df = pseudoseq_clustered_df.loc[
    pseudoseq_clustered_df.allele.isin(mhcflurry_df.allele)
]

# GENERATE TRAIN AND TEST FOR EACH EXPERIMENT

# cluster shuffled by selecting randomly 1/4 or 100 (what's less) from each allele. Using the exact same mhcflurry function to this end:
shuffled_train_test_df_mask = assign_folds(shuffled_df, num_folds= 1, held_out_fraction= .25, held_out_max= 100)
shuffled_train_df = shuffled_df.loc[shuffled_train_test_df_mask["fold_0"]]
shuffled_test_df = shuffled_df.loc[~shuffled_train_test_df_mask["fold_0"]]

print(f"Len of shuffled train: {shuffled_train_df.shape[0]} and test: {shuffled_test_df.shape[0]}")

# for the peptide clustered, 9 clusters from the cluster_set_10 are used for train and the 4th as the test:
peptide_clustered_train_df = peptide_clustered_df.loc[peptide_clustered_df.cluster_set_10 != 3]
peptide_clustered_test_df = peptide_clustered_df.loc[peptide_clustered_df.cluster_set_10 == 3]

print(f"Len of peptide clustered train: {peptide_clustered_train_df.shape[0]} and test: {peptide_clustered_test_df.shape[0]}")

# for the pseudoseq clustered, cluster 0 represent the train and cluster 1 the test:
pseudoseq_clustered_train_df = pseudoseq_clustered_df.loc[pseudoseq_clustered_df.allele_clustering == 0]
pseudoseq_clustered_test_df = pseudoseq_clustered_df.loc[pseudoseq_clustered_df.allele_clustering == 1]

print(f"Len of pseudoseq clustered train: {pseudoseq_clustered_train_df.shape[0]} and test: {pseudoseq_clustered_test_df.shape[0]}")

# SAVE EACH CSV:
destination_folder = "/projects/0/einf2380/data/external/processed/I/experiments"

# shuffled:
shuffled_train_df.to_csv(f"{destination_folder}/BA_pMHCI_human_quantitative_only_eq_shuffled_train_validation.csv", index=False)
shuffled_test_df.to_csv(f"{destination_folder}/BA_pMHCI_human_quantitative_only_eq_shuffled_test.csv", index=False)

# peptide clustered:
peptide_clustered_train_df.to_csv(f"{destination_folder}/BA_pMHCI_human_quantitative_only_eq_peptide_clustered_train_validation.csv", index=False)
peptide_clustered_test_df.to_csv(f"{destination_folder}/BA_pMHCI_human_quantitative_only_eq_peptide_clustered_test.csv", index=False)

#pseudoseq clustered:
pseudoseq_clustered_train_df.to_csv(f"{destination_folder}/BA_pMHCI_human_quantitative_only_eq_pseudoseq_clustered_train_validation.csv", index=False)
pseudoseq_clustered_test_df.to_csv(f"{destination_folder}/BA_pMHCI_human_quantitative_only_eq_pseudoseq_clustered_test.csv", index=False)