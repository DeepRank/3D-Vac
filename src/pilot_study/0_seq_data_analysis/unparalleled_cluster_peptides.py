import os.path
from Bio.SubsMat import MatrixInfo
from Bio.Align import substitution_matrices
from matplotlib import pyplot as plt
import scipy.cluster.hierarchy as sch
import pickle
# from sklearn.cluster import AgglomerativeClustering
import numpy as np
import time
# import seqlogo
import argparse
import logomaker

arg_parser = argparse.ArgumentParser(description="Cluster peptides from a .csv file")
arg_parser.add_argument(
    "--file","-f",
    help="path to the .csv file",
    default="/home/lepikhovd/3D-Vac/data/binding_data/BA_pMHCI.csv"
)
arg_parser.add_argument("--threshold", "-t",
    help="Number of clusters",
    type=int,
    default=10
)
a = arg_parser.parse_args()

with open(a.file, "r") as csv_f:
    rows = [line.replace("\n", "").split(",") for line in csv_f]
    peptides = [row[2] for row in rows]

###############################################################
#TODO: keep peptide ID 
# def cluster_peptides(peptides, threshold, frag_len = 9, 
#                      matrix='PAM250',
#                      outplot = False,
#                     #  outplot = '/home/lepikhovd/peptide.jpg',
#                      elbow= True,
#                      ):
print("peptides loaded")
threshold = a.threshold
frag_len = 9
matrix = "PAM250"
outplot = False
elbow = False
# """
# Calculates evolutionary distance (with a substitution matrix) within a set of peptides.
# Uses this distances to generate a dendrogram and cluster the pepties.


# Args:
#     peptides (list): List of peptides to be clustered.
#     threshold (int): Score threshold to cut the dendrogram into clusters.
#     frag_len (int, optional): Length of the given peptides. Residues past this positin
#         will not be scored. Shorter peptides will cause this function to crash.
#         Defaults to 9.
#     matrix (str, optional): Substitution matrix to calulate peptides distance. 
#         Defaults to 'PAM30'.
#     outplot (str or None, optional): If a path or file name is provided,
#         it will generate a dendrogram output plot. Defaults to None.

# Returns:
#     clst_dct (dict): dictionary of the peptides clusters.

# """

t1 = time.time()
peptides = sorted(list(set(peptides)))
# cluster = AgglomerativeClustering(n_clusters=threshold)

# Calculate distance between each peptide
# score_array = [] ### [ AB, AC, AD, AE, BC, BD, BE, CD, CE, DE]
score_array = pickle.load(open("./PAM250_scores.pkl", "rb")) ### [ AB, AC, AD, AE, BC, BD, BE, CD, CE, DE]
# subt_matrix = substitution_matrices.load(matrix)
# for i, frag in enumerate(peptides):
#     for j in range(i+1, len(peptides)):
#         score = 0
#         for l in range(frag_len):
#             try:
#                 score += subt_matrix[peptides[i][l], peptides[j][l]]
#             except:
#                 score += subt_matrix[peptides[j][l], peptides[i][l]]
#         score_array.append(score)
t2 = time.time()
print('Calculate distances:', t2-t1)

#Convert the distances in a score array
dist_array = []
top = max(score_array)    
for x in score_array:
    y = top + 1 - x
    dist_array.append(y)
array = np.asarray(dist_array)

t3 = time.time()
print('Distances array:', t3-t2)
result = sch.linkage(array, method="complete")
# cluster = AgglomerativeClustering(n_clusters=threshold, linkage="complete",compute_full_tree=False)

t4 = time.time()
print('Linkage:', t4-t3)

#Plot dendrogram
if outplot:
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
    
    #X = []
    #Y = []
    #ratios = []
    
        #X.append(x)
        #Y.append(y)
        #ratios.append(y/x)
        
    #plt.plot(X, Y, 'ro', ratios, 'b-')
    plt.axhline(threshold, color='r')
    #plt.xlim(5, 30)
    #plt.ylim(0, 400)
    plt.show()
    plt.savefig(f"/home/lepikhovd/dendogram_{matrix}.png", dpi=200)
    plt.close()

t5 = time.time()
print('Plot:', t5-t4)
#Produce clusters using the given threshold
fc = sch.fcluster(result, threshold, criterion='maxclust') # the number inside specifies the cutoff distance for dividing clusers
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
            mtf_lst[(int(number))][i].append(frag[i])
        mtf_lst[(int(number))][i] = list(set(mtf_lst[(int(number))][i]))
pickle.dump(clst_dct, open("./clusters.pkl", "wb"))

t6 = time.time()
print('Clusters:', t6-t5)

#make the elbow figure:
if elbow:
    last = result[:,2]
    Y = last[::-1]
    idxs = np.arange(1, len(last)+1)
    plt.plot(idxs,Y)
    plt.xlabel("Distance index")
    plt.ylabel("Distance")
    plt.title(f"Ranked distances between dendogram clusters for the {matrix} matrice")
    plt.savefig(f"/home/lepikhovd/elbow_{matrix}.png")
# 