#!/bin/bash
#removed --partition thin
#removed --array=0-9
#removed --gpus 1
#removed --nodes 1
#removed --ntasks 1
#removed --cpus-per-task 1
#removed --time 00:01:00
#removed -o /projects/0/einf2380/data/training_logs/II/cnn_classification_struct_cuda-%J.out
#removed -e /projects/0/einf2380/data/training_logs/II/cnn_classification_struct_cuda-%J.err
#removed --mem-per-gpu 39440

conda activate 3D-Vac

for i in {0..9}
do
    sbatch -o /projects/0/einf2380/data/training_logs/II/cnn_classification_struct_cuda-%J_s$i.out \
    -e /projects/0/einf2380/data/training_logs/II/cnn_classification_struct_cuda-%J_s$i.err \
    train_single_fold.sh $i 0
done


# for i in {0..9}
# do
#     sbatch -o /projects/0/einf2380/data/training_logs/I/cnn_classification_struct_4conv-%J_c$i.out \
#     -e /projects/0/einf2380/data/training_logs/I/cnn_classification_struct_4conv-%J_c$i.err \
#     train_single_fold.sh $i
# done