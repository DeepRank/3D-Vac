import h5py

data = {}
for filename in filenames:
    f = h5py.File('hdf5/'+filename, 'r')
    data[filename.split('.')[0]] = f
    
