#!/bin/bash

experiments=('xray' 'pandora_1k' 'pandora_5k' 'pandora_10k' 'pandora_all' 'xray_pandora_all')

for data_choice in "${experiments[@]}"; do
    for i in $(seq 1 5); do
        job_name="3DSSL_${data_choice}_${i}"
        # Submit the job with the specified job name
        sbatch --job-name=$job_name run_ssl_single_fold.sh $data_choice $i
    done
done