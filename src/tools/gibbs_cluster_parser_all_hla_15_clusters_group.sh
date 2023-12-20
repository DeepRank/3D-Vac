#!/bin/bash
#SBATCH --partition thin
#SBATCH --time 1:00:00
#SBATCH --nodes 1
#SBATCH --ntasks 15
#SBATCH -c 4
#SBATCH --output /projects/0/einf2380/data/modelling_logs/I/gibbs_cluster/parsing_cluster_job-%J.out

# This is an example script on how to use gibbs_cluster_parser with a node

module load 2021
module load foss/2021a

source activate deeprank

srun python -u gibbs_cluster_parser.py \
	--all-peptides \
	--all-peptides-server-output /home/lepikhovd/softwares/gibbscluster-2.0/run/all_hla_peptides_clusters_1_15_with_trash_1876625 \
	--file ../../data/external/processed/all_hla.csv \
