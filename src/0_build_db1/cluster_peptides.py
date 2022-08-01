import argparse
import numpy as np
import os.path
import pickle
import scipy.cluster.hierarchy as sch
import time
from Bio.Align import substitution_matrices
from joblib import Parallel, delayed
from math import ceil
import pandas as pd
import matplotlib.pyplot as plt

arg_parser = argparse.ArgumentParser(description=" \
    Cluster peptides from a .csv file. \
    Calculates evolutionary distance (with a substitution matrix) within a set of peptides. \
    Uses this distance to generate a dendogram and cluster the peptides. \
    At the end of execution, dumps the clusters into clusters.pkl file. \
    The pkl file is made of --clusters number of lists containaing {peptide,ba_value} objects. \
    ")
arg_parser.add_argument(
    "--file","-f",
    help="Name of the DB1 file.",
    required=True
)
arg_parser.add_argument("--clusters", "-c",
    help="Maximum number of clusters. A threshold will be calculated to reach the closest number of clusters.",
    type=int,
    default=10
)
arg_parser.add_argument("--matrix", "-m", 
    help="Matrix to use, default is the PAM250. Other matrices can be added",
    choices=["PAM250", "PAM30"],
    default="PAM250",
)
arg_parser.add_argument("--make-graphs", "-e",
    help="Creates the dendogram and the elbow graph. Default no.",
    action="store_true",
    default=False,
)
arg_parser.add_argument("--njobs", "-n",
    help="Defines the number of jobs to launch for the matrix generation. Default 1.",
    default=1,
    type=int
)
arg_parser.add_argument("--update-csv", "-u",
    help="This option allows to either add/update the `cluster` column in db1.",
    default=False,
    action="store_true"
)
arg_parser.add_argument("--peptides-length", "-l",
    help="Peptides to be clustered length",
    default=9,
    type=int
)
a = arg_parser.parse_args()

###############################################################
#TODO: keep peptide ID

def split_in_indexed_batches(samples, n_jobs):
    samples = [(i, x) for i,x in enumerate(samples)]
    batches = []
    step = ceil(len(samples)/n_jobs)
    end = 0
    for job in range(n_jobs):
        start = end
        end = start + step
        batches.append(samples[start:end])

    return batches

def get_pepts_dist(pept1, pept2, subt_matrix):
    length = min([len(pept1), len(pept2)])
    score = 0
    for l in range(length):
        try:
            score += subt_matrix[pept1[l], pept2[l]]
        except:
            score += subt_matrix[pept2[l], pept1[l]]
    return score

def get_scores(matrix, batch, peptides):
    # Calculate distance between each peptide
    score_array = [] ### [ AB, AC, AD, AE, BC, BD, BE, CD, CE, DE]
    subt_matrix = substitution_matrices.load(matrix)

    for i, pept1 in batch:
        for j in range(i+1, len(peptides)):
            pept2 = peptides[j]

            score = get_pepts_dist(pept1, pept2, subt_matrix)
            score_array.append((i, score))

    return score_array

def get_score_matrix(peptides, n_jobs, matrix):
    peptides = sorted(list(set(peptides)))

    batches = split_in_indexed_batches(peptides, n_jobs)

    arrays = Parallel(n_jobs=n_jobs, verbose=1)(delayed(get_scores)(matrix, batch, peptides) for batch in batches)
    arrays = [sorted(a, key=lambda x:x[0]) for a in arrays]

    score_array = []
    for x in arrays:
        x = [y[1] for y in x]
        score_array.extend(x)

    return score_array

def cluster_peptides(peptides, n_clusters, frag_len = 9,
                     matrix='PAM30', n_jobs=1,):
    """
    Calculates evolutionary distance (with a substitution matrix) within a set of peptides.
    Uses this distances to generate a dendrogram and cluster the pepties.


    Args:
        peptides (list): List of peptides to be clustered.
        threshold (int): Score threshold to cut the dendrogram into clusters.
        frag_len (int, optional): Length of the given peptides. Residues past this positin
            will not be scored. Shorter peptides will cause this function to crash.
            Defaults to 9.
        matrix (str, optional): Substitution matrix to calulate peptides distance.
            Defaults to 'PAM30'.
        outplot (str or None, optional): If a path or file name is provided,
            it will generate a dendrogram output plot. Defaults to None.

    Returns:
        clst_dct (dict): dictionary of the peptides clusters.

    """

    t1 = time.time()

    score_array = get_score_matrix(peptides, n_jobs, matrix)

    t2 = time.time()

    #Convert the distances in a score array
    dist_array = []
    top = max(score_array)
    for x in score_array:
        y = top + 1 - x
        dist_array.append(y)
    array = np.asarray(dist_array)

    t3 = time.time()
    #Calculate linkage between peptides (i.e. the dendrogram)
    result = sch.linkage(array, method='complete')

    t4 = time.time()

    #Plot dendrogram
    if a.make_graphs:
        plt.figure(figsize=(60, 20))
        plt.title('Peptides Hierarchical Clusterization')
        plt.xlabel('Peptides')
        plt.ylabel('Distance')
        sch.dendrogram(
            result,
            leaf_rotation=90.,  # rotates the x axis labels
            leaf_font_size=8.,  # font size for the x axis labels
            show_contracted=True,
            labels = peptides
        )

        plt.savefig(f"../../reports/figures/{filename}_dendogram_{matrix}.png", dpi=200)
        plt.clf()

    if a.make_graphs:
        last = result[:,2]
        Y = last[::-1]
        idxs = np.arange(1, len(last)+1)
        plt.plot(idxs,Y)
        plt.xlabel("Distance index")
        plt.ylabel("Distance")
        plt.title(f"Ranked distances between dendogram clusters for the {matrix} matrice")
        plt.savefig(f"../../reports/figures/elbow_{matrix}.png")
        plt.clf()
        print(f"Elbow figure saved in reports/figures/{filename}_elbow_{matrix}.png")

    t5 = time.time()
    #Produce clusters using the given threshold
    fc = sch.fcluster(result, n_clusters, criterion='maxclust') # the number inside specifies the cutoff distance for dividing clusers
    ordered_clusters = sorted(zip(fc, peptides))
    clst_dct = {}
    mtf_lst = []
    for i in set(fc):
        clst_dct['clst_%s' % (i-1)] = []
        mtf_lst.append([])


    for clst in clst_dct:
        lenght = len(clst)
        number = clst[5:lenght]
        for i in ordered_clusters:
            if str((i[0]-1)) == (number):
                clst_dct[clst].append(i[1])
        mtf_lst[(int(number))] = [[] for j in range(frag_len)]
        for i in range(frag_len):
            for frag in clst_dct[clst]:
                if len(frag) == frag_len:
                    mtf_lst[(int(number))][i].append(frag[i])
            mtf_lst[(int(number))][i] = list(set(mtf_lst[(int(number))][i]))

    t6 = time.time()

    print('Calculate distances:', t2-t1)
    print('Distances array:', t3-t2)
    print('Linkage:', t4-t3)
    print('Plot:', t5-t4)
    print('Clusters:', t6-t5)
    return clst_dct

filename = a.file.split('/')[-1].split('.')[0]
csv_path = a.file
df = pd.read_csv(csv_path) 

# peptides has to be a unique set because the dendogram is calculated for unique peptide sequences. Because peptides are 
# used as labels, different length between peptides and the actual number of clusters (unique sequences) lead to an error.
peptides = sorted(list(set(df["peptide"].tolist()))) 

clusters = cluster_peptides(
    peptides=peptides,
    matrix=a.matrix,
    n_jobs = a.njobs,
    n_clusters = a.clusters,
    frag_len = a.peptides_length
)

if a.update_csv: 
    for idx,cluster in enumerate(clusters.keys()):
        for peptide in clusters[cluster]:
            df.loc[df["peptide"] == peptide, "cluster"] = int(idx)
    df.to_csv(csv_path, index=False)

pickle.dump(clusters, open(f"../../data/external/processed/{filename}_{a.matrix}_{a.clusters}_clusters.pkl", "wb"))