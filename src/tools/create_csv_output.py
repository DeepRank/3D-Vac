import pandas as pd
import numpy


def from_ic50(ic50, max_ic50=50000.0):
    """
    Convert ic50s to regression targets in the range [0.0, 1.0].
    
    Parameters
    ----------
    ic50 : numpy.array of float

    Returns
    -------
    numpy.array of float

    """
    x = 1.0 - (numpy.log(numpy.maximum(ic50, 1e-12)) / numpy.log(max_ic50))
    return numpy.minimum(
        1.0,
        numpy.maximum(0.0, x))


def to_ic50(x, max_ic50=50000.0):
    """
    Convert regression targets in the range [0.0, 1.0] to ic50s in the range
    [0, 50000.0].
    
    Parameters
    ----------
    x : numpy.array of float

    Returns
    -------
    numpy.array of float
    """
    return max_ic50 ** (1.0 - x)

output_path = "/projects/0/einf2380/data/external/processed/I/BA_pMHCI_human_quantitative_only_eq_models_output.csv"
df = pd.read_csv(output_path)
print(df)

# MHC Flurry allele clustered:
mhcflurry_allele_clustered = df.loc[df.mhcflurry_allele_clustered.notnull()]
mhcflurry_allele_clustered_df = pd.concat(
    [mhcflurry_allele_clustered.ID, mhcflurry_allele_clustered.mhcflurry_allele_clustered, from_ic50(mhcflurry_allele_clustered.mhcflurry_allele_clustered)],
    axis=1,
)
mhcflurry_allele_clustered_df.to_csv(
    "/projects/0/einf2380/data/pop_paper_data/mhcflurry_outputs/mhcflurry_allele_clustered_outputs.csv", index=False, header=False
)
# MHC Flurry peptide clustered:
mhcflurry_peptide_clustered = df.loc[df.mhcflurry_peptide_clustered.notnull()]
mhcflurry_peptide_clustered_df = pd.concat(
    [mhcflurry_peptide_clustered.ID, mhcflurry_peptide_clustered.mhcflurry_peptide_clustered, from_ic50(mhcflurry_peptide_clustered.mhcflurry_peptide_clustered)],
    axis=1,
)
mhcflurry_peptide_clustered_df.to_csv(
    "/projects/0/einf2380/data/pop_paper_data/mhcflurry_outputs/mhcflurry_peptide_clustered_outputs.csv", index=False, header=False
)

# MHC Flurry allele clustered:
mhcflurry_shuffled_trained = df.loc[df.mhcflurry_shuffled_trained.notnull()]
mhcflurry_shuffled_trained_df = pd.concat(
    [mhcflurry_shuffled_trained.ID, mhcflurry_shuffled_trained.mhcflurry_shuffled_trained, from_ic50(mhcflurry_shuffled_trained.mhcflurry_shuffled_trained)],
    axis=1,
)
mhcflurry_shuffled_trained_df.to_csv(
    "/projects/0/einf2380/data/pop_paper_data/mhcflurry_outputs/mhcflurry_shuffled_outputs.csv", index=False, header=False
)

# MLP allele clustered:
mlp_allele_clustered = df.loc[df.mlp_allele_clustered.notnull()]
mlp_allele_clustered_df = pd.concat(
    [mlp_allele_clustered.ID, mlp_allele_clustered.mlp_allele_clustered, from_ic50(mlp_allele_clustered.mlp_allele_clustered)],
    axis=1,
)
mlp_allele_clustered_df.to_csv(
    "/projects/0/einf2380/data/pop_paper_data/mlp_outputs/mlp_allele_clustered_outputs.csv", index=False, header=False
)
# MLP peptide clustered:
mlp_peptide_clustered = df.loc[df.mlp_peptide_clustered.notnull()]
mlp_peptide_clustered_df = pd.concat(
    [mlp_peptide_clustered.ID, mlp_peptide_clustered.mlp_peptide_clustered,from_ic50(mlp_peptide_clustered.mlp_peptide_clustered)],
    axis=1,
)
mlp_peptide_clustered_df.to_csv(
    "/projects/0/einf2380/data/pop_paper_data/mlp_outputs/mlp_peptide_clustered_outputs.csv", index=False, header=False
)
# MLP shuffled:
mlp_shuffled = df.loc[df.mlp_shuffled.notnull()]
mlp_shuffled_df = pd.concat(
    [mlp_shuffled.ID, mlp_shuffled.mlp_shuffled, from_ic50(mlp_shuffled.mlp_shuffled)],
    axis=1,
)
mlp_shuffled_df.to_csv(
    "/projects/0/einf2380/data/pop_paper_data/mlp_outputs/mlp_shuffled_outputs.csv", index=False, header=False
)