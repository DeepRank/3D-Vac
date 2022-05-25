import h5py
h5path = '/projects/0/einf2380/data/pMHCI/features_output_folder/000_hla_a_02_01_9_length_peptide.hdf5'
db = h5py.File(h5path, 'r')
# for model in db.keys():
#     print(model)
    # for atom in db[model]['features']['Edesolv']