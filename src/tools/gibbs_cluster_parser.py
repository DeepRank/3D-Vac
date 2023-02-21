import pandas as pd
import argparse
import matplotlib.pyplot as plt
import numpy as np
import math
import glob
from joblib import Parallel, delayed
import re
import pickle

arg_parser = argparse.ArgumentParser(description="""
    Script used to parse GibbsCluster server (https://services.healthtech.dtu.dk/service.php?GibbsCluster-2.0)
    output and assign gibbs_cluster to the DB1 csv file provided in --file. If the --plot argument is provided,
    only the histogram of peptide distribution is plotted without updating the csv. Generates or updates a 
    "gibbs_cluster" column (name can be changed). Expects both negative and positive server outputs from 
    the gibbs cluster software.
""")

arg_parser.add_argument("--file", "-f",
    help="Path to the DB1 csv file.",
    default="../../data/external/processed/BA_pMHCI.csv"
)
arg_parser.add_argument("--all-peptides", "-a",
    help="""
    With this argument provided, only one server output is required (--all-peptides-server-output), 
    clusters expected are those from all peptides, without distinction between binders and non binders. 
    """,
    default=False,
    action="store_true"
)
arg_parser.add_argument("--all-peptides-server-output", "-A",
    help="""
    Path to the Gibbs server output folder containing clustered peptides, without distinction 
    between binders and non binders. Use with --all-peptides argument. 
    """
)
arg_parser.add_argument("--positive-server-output", "-P",
    help="Path to the gibbs server output folder containing clustered binders.",
)

arg_parser.add_argument("--negative-server-output", "-N",
    help="Path to the gibbs server output folder containing clustered non binders.",
)
arg_parser.add_argument("--n-clusters", "-c",
    help="Number of clusters desired.",
    default=10,
    type=int
)
arg_parser.add_argument("--cluster-column-name", "-n",
    help="Name of the newly added (or updated) column for the gibbs cluster. Default gibbs_cluster.",
    default="gibbs_cluster"
)

a = arg_parser.parse_args()

# Define the file to read:
df = pd.read_csv(a.file)

if not a.all_peptides:
    print("Retrieving positive and negative clusters from output files..")
    clusters = [{}, {}] # first element is positive dict and second negative dict

    for i, f in enumerate([a.positive_server_output, a.negative_server_output]):
        file = f"{f}/res/gibbs.{a.n_clusters}g.ds.out"
        with open(file) as infile:
            print(f"READING OUTPUT FILE OF SERVER OUTPUT {f}...")
            next(infile)
            for line in infile:
                clust_name = f"{('pos', 'neg')[i]}_{line.split()[1]}"
                if not clust_name in list(clusters[i].keys()):
                    clusters[i][clust_name] = {'peptides' : [], 'cores': []}

                clusters[i][clust_name]['cores'].append(line.split()[4]) #[3] is the peptide, [4] is the core
                clusters[i][clust_name]['peptides'].append(line.split()[3])
    # Update the gibbs cluster columns:
    print("UPDATING `gibss_cluster` COLUMN...")
    for i in range(len(clusters)):
        for c in clusters[i].keys():
            for p in clusters[i][c]["peptides"]:
                df.loc[df["peptide"] == p, a.cluster_column_name] = c
            print(f"Number of peptides in cluster {c}: {len(df.loc[df[a.cluster_column_name] == c])}")
        df.to_csv(a.file, index=False)

    # Get proportional distribution of each clsusters from neg and pos:
    # Generate a pandas series of positive clusters and negative ones:
    clusters_g = df.groupby(a.cluster_column_name)

    neg_clusters = []
    neg_clusters_count = []

    pos_clusters = []
    pos_clusters_count = []

    for name, group in clusters_g:
        if "neg" in name:
            neg_clusters.append(name)
            neg_clusters_count.append(len(group))
        if "pos" in name:
            pos_clusters.append(name)
            pos_clusters_count.append(len(group))

    # Transform the arrays into np.array to facilitate processing:
    neg_clusters = np.array(neg_clusters)
    neg_clusters_count = np.array(neg_clusters_count)

    pos_clusters = np.array(pos_clusters)
    pos_clusters_count = np.array(pos_clusters_count)

    # Sort the positive and negative clusters counts. 
    neg_clusters_count_indices = neg_clusters_count.argsort()
    pos_clusters_count_indices = pos_clusters_count.argsort()

    # Concatenate clusters and clusters counts
    pos_neg_clusters = np.concatenate((pos_clusters, neg_clusters))
    pos_neg_clusters_count = np.concatenate((pos_clusters_count, neg_clusters_count))
    # Update the neg_clusters_count_indices by shifting each indice after the pos_cluster
    neg_clusters_count_indices = neg_clusters_count_indices + pos_clusters_count_indices.shape[0]

    pos_neg_clusters_count_indices = []

    # This loop populates the pos_neg_clusters_count_indices in 1-1 pairs, one for pos and one from negatives,
    # based on sorted indices obtained before concatenating positive and negatives clusters count.
    for i in range(pos_neg_clusters.shape[0]):
        if i%2==0:
            to_append = pos_clusters_count_indices[-1] # Add the last element of the array
            pos_clusters_count_indices = pos_clusters_count_indices[:-1] # Pop the array
        else:
            to_append = neg_clusters_count_indices[-1]
            neg_clusters_count_indices = neg_clusters_count_indices[:-1]
        pos_neg_clusters_count_indices.append(to_append)

    # Create the final groups containing the clusters
    pos_neg_clusters_groups = pos_neg_clusters[pos_neg_clusters_count_indices]
    pos_neg_clusters_count_groups = pos_neg_clusters_count[pos_neg_clusters_count_indices]
    colors = ["red", "green"]*math.ceil(pos_neg_clusters_groups.shape[0]/2)

    # Plot to visualize
    p_file_name = a.positive_server_output.split("/")[-1]
    n_file_name = a.negative_server_output.split("/")[-1]
    plt_name = f"POS-{p_file_name}-NEG-{n_file_name}-clusters_groups-{a.n_clusters}_clusters.png"
    plt.figure(figsize=(12,10))
    plt.xticks(fontsize=14, rotation=90)
    plt.bar(pos_neg_clusters_groups, height = pos_neg_clusters_count_groups, color = colors)
    plt.title("Peptides disitribution among groups of clusters made of binders and non-binders", fontsize=18)
    plt.xlabel("Cluster", fontsize=16)
    plt.ylabel("Peptide count", fontsize=16)
    plt.savefig(f"../../reports/figures/gibbs_clusters/{plt_name}")
    plt.close()

else:
    from mpi4py import MPI
    conn = MPI.COMM_WORLD

    size = conn.Get_size()
    rank = conn.Get_rank()

    f_path = a.all_peptides_server_output
    files = glob.glob(f"{f_path}/res/gibbs.*g.ds.out")
    clusters = {}
    f = files[rank]
    with open(f) as infile:
        filename = f.split('/')[-1]
        print(f"Reading file {filename} on task {rank}")
        cluster_set = int(re.findall(r"\d+", filename)[0]) # index of the cluster_set
        print(f"cluster_set from regexp: {cluster_set}")
        next(infile)
        for line in infile:
            clust_name = str(line.split()[1])
            if not clust_name in list(clusters.keys()):
                clusters[clust_name] = {'peptides' : [], 'cores': []}

            clusters[clust_name]['cores'].append(line.split()[4]) #[3] is the peptide, [4] is the core
            clusters[clust_name]['peptides'].append(line.split()[3])

    for c in clusters.keys():
        for p in clusters[c]["peptides"]:
            df.loc[df["pep_anch_rep"] == p, f"cluster_set_{cluster_set}"] = str(c)
    print(f"Finished populating df on {rank}")
    all_df = conn.gather(df)
    if rank==0:
        for i in range(len(all_df)):
            if i == 0:
                continue
            cluster_set = [col for col in list(all_df[i].columns) if "cluster_set" in col][0]
            print(f"Cluster set: {cluster_set}")
            all_df[0][cluster_set] = all_df[i][cluster_set]
        all_df[0].to_csv(a.file, index=False)
        print("Done!")