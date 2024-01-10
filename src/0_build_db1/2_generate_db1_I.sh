python generate_db1_I.py \
    --source-csv /projects/0/einf2380/data/external/processed/I/BA_ids_pMHCI_human_nonhuman.csv \
    --output-csv /projects/0/einf2380/data/external/processed/I/BA_pMHCI_human_quantitative_only_eq.csv \
    --allele HLA \
    --prefix BA \
    --measurement-type BA 

# Inputs: MHCFlurry dataset csv filename in `data/external/unprocessed`.
# Outputs: DB1 in 'path-to-destination.csv'.
# Run `python src/0_build_db1/generate_db1_I.py --help` for more details on how to filter for specific allele and peptide length.