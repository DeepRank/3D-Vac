python generate_db1_I.py \
    --source-csv /projects/0/einf2380/data/external/unprocessed/pMHCI/curated_training_data.no_additional_ms.csv \
    --output-csv /projects/0/einf2380/data/external/processed/I/BA_pMHCI_HLA0201.csv \
    --peptide-length 9 \
    --allele HLA-A*02:01 \
    --prefix BA \
    --measurement_type BA 

# Inputs: MHCFlurry dataset csv filename in `data/external/unprocessed`.
# Outputs: DB1 in 'path-to-destination.csv'.
# Run `python src/0_build_db1/generate_db1_I.py --help` for more details on how to filter for specific allele and peptide length.