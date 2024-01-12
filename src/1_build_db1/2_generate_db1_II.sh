# Generate full BA database
python generate_db1_II.py -o /projects/0/einf2380/data/external/processed/II/IDs_BA_MHCII.csv
# Generate subset without auplicated labels
python generate_db1_subset.py -i /projects/0/einf2380/data/external/processed/II/IDs_BA_MHCII.csv -o /projects/0/einf2380/data/external/processed/II/IDs_BA_MHCII_noduplicates.csv -d remove -H
# Generate HLA-DRB1*01:01 pilot study database and without auplicated labels
python generate_db1_subset.py -i /projects/0/einf2380/data/external/processed/II/IDs_BA_MHCII.csv -o /projects/0/einf2380/data/external/processed/II/IDs_BA_DRB0101_MHCII.csv -a HLA-DRB1*01:01 -d remove -H 
# Generate HLA-DRB1*01:01 on 15-mers pilot study database and without auplicated labels
python generate_db1_subset.py -i /projects/0/einf2380/data/external/processed/II/IDs_BA_MHCII.csv -o /projects/0/einf2380/data/external/processed/II/IDs_BA_DRB0101_MHCII_15mers.csv -a HLA-DRB1*01:01 -l 15 -d remove -H