from Bio.Align import substitution_matrices
from matplotlib import pyplot as plt
import scipy.cluster.hierarchy as sch
import pickle
import numpy as np
import time
import argparse

arg_parser = argparse.ArgumentParser(description=" \
    Cluster peptides from a .csv file. \
    Calculates evolutionary distance (with a substitution matrix) within a set of peptides. \
    Uses this distance to generate a dendogram and cluster the peptides. \
    At the end of execution, dumps the clusters into clusters.pkl file. \
    ")
arg_parser.add_argument(
    "--file","-f",
    help="Path to the .csv file.",
    default="/home/lepikhovd/binding_data/BA_pMHCI.csv"
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
arg_parser.add_argument("--save-matrix", "-s",
    help="If set, new pickle file for the matrix distance is generated. If not set, last generated matrix is used.",
    action="store_true",
    default=False
)
arg_parser.add_argument("--make-graphs", "-e",
    help="Creates the dendogram and the elbow graph. Default no.",
    action="store_true",
    default=False,
)
a = arg_parser.parse_args()

with open(a.file, "r") as csv_f:
    rows = [line.replace("\n", "").split(",") for line in csv_f]
    peptides = [row[2] for row in rows]

print("peptides loaded")
threshold = a.clusters
frag_len = 9
matrix = "PAM250"
outplot = a.make_graphs
elbow = a.make_graphs

peptides = sorted(list(set(peptides)))

# Calculate distance between each peptide
score_array = [] ### [ AB, AC, AD, AE, BC, BD, BE, CD, CE, DE]
if a.save_matrix == False:
    score_array = pickle.load(open("./PAM250_scores.pkl", "rb")) ### [ AB, AC, AD, AE, BC, BD, BE, CD, CE, DE]
    t2=time.time()
    print("Matrix loaded from pkl file")
else:
    t1 = time.time()
    subt_matrix = substitution_matrices.load(matrix)
    print("Building matrix...")
    for i, frag in enumerate(peptides):
        for j in range(i+1, len(peptides)):
            score = 0
            for l in range(frag_len):
                try:
                    score += subt_matrix[peptides[i][l], peptides[j][l]]
                except:
                    score += subt_matrix[peptides[j][l], peptides[i][l]]
            score_array.append(score)
    t2 = time.time()
    pickle.dump(score_array, open("./distance_scores.pkl", "wb"))
    print("Matrix built, pkl file saved.")
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
    
    plt.axhline(threshold, color='r')
    plt.show()
    plt.savefig(f"/home/lepikhovd/dendogram_{matrix}.png", dpi=200)
    plt.close()
    t5 = time.time()
    print('Plot:', t5-t4)
else:
    t5 = time.time()
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
    print(f"Elbow figure saved in /home/lepikhovd/elbow_{matrix}.png")