#!/bin/bash


#AlleleClustered

for i in $(seq 1 5); do
  job_name_prefix='AlleleClustered'
  job_name="MLP_${job_name_prefix}_${i}"
  
  # Submit the job with the specified job name
  sbatch --job-name=$job_name train_MLP_single_fold.sh $job_name_prefix $i
done


#Shuffled

for i in $(seq 1 5); do
  job_name_prefix='Shuffled'
  job_name="MLP_${job_name_prefix}_${i}"
  
  # Submit the job with the specified job name
  sbatch --job-name=$job_name train_MLP_single_fold.sh $job_name_prefix $i
done