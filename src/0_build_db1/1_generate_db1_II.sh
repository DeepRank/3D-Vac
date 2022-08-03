# Generate full BA database
python generate_db1_II.py -d ../../data/external/processed/IDs_BA_MHCII.csv
# Generate HLA-DRB1*01:01 pilot study database
python generate_db1_subset.py -i ../../data/external/processed/IDs_BA_MHCII.csv -o ../../data/external/processed/IDs_BA_DRB0101_MHCII.csv -a HLA-DRB1*01:01
# Generate HLA-DRB1*01:01 on 15-mers pilot study database
python generate_db1_subset.py -i ../../data/external/processed/IDs_BA_MHCII.csv -o ../../data/external/processed/IDs_BA_DRB0101_MHCII_15mers.csv -a HLA-DRB1*01:01 -l 15