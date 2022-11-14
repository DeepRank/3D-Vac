import pandas as pd
import argparse
import matplotlib.pyplot as plt

arg_parser = argparse.ArgumentParser(description="""
    Script used to parse GibbsCluster server (https://services.healthtech.dtu.dk/service.php?GibbsCluster-2.0)
    output and assign gibbs_cluster to the DB1 csv file provided in --file. If the --plot argument is provided,
    only the histogram of peptide distribution is plotted without updating the csv
""")

arg_parser.add_argument("--file", "-f",
    help="Path to the DB1 csv file.",
    default="../../data/external/processed/BA_pMHCI.csv"
)
arg_parser.add_argument("--gibbs-folder", "-g",
    help="Path to the gibbs server output folder.",
    default="/home/daqop/Desktop/3D-vac-project/GibbsCluster/ServerOutput/hla0201_9mers_pos"
)
arg_parser.add_argument("--n-clusters", "-c",
    help="Number of clusters desired.",
    default=10,
    type=int
)
arg_parser.add_argument("--cluster-prefix", "-C",
    help="Prefix to the numbering of the gibbs cluster: prefix_cluster_num.",
    default="pos"
)

arg_parser.add_argument("--plot", "-p",
    help="If this argument is set, the script plots the histogram of peptide count per cluster in the DB1.",
    default=False,
    action="store_true"
)

a = arg_parser.parse_args()

# Define the file to read:
file = f"{a.gibbs_folder}/res/gibbs.{a.n_clusters}g.ds.out"
df = pd.read_csv(a.file)

clusters = {}

if not a.plot:
    with open(file) as infile:
        next(infile)
        for line in infile:

            clust_name = f"{a.cluster_prefix}_{line.split()[1]}"
            if not clust_name in list(clusters.keys()):
                clusters[clust_name] = {'peptides' : [], 'cores': []}

            clusters[clust_name]['cores'].append(line.split()[4]) #[3] is the peptide, [4] is the core
            clusters[clust_name]['peptides'].append(line.split()[3])

    for c in clusters:
        print(f"Number of unique peptides in cluster {c} : {len(clusters[c]['peptides'])}")

    # Update the gibbs cluster columns:
    for c in clusters:
        for p in clusters[c]["peptides"]:
            df.loc[df["peptide"] == p, "gibbs_cluster"] = c
        print(len(df.loc[df["gibbs_cluster"] == c]))
    df.to_csv(a.file, index=False)

else:
    clusters_set = set(df["gibbs_cluster"])
    clusters_count = []
    for c in clusters_set:
        print(c)