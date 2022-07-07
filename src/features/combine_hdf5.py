# This script takes as input different .hdf5 files and combines them into one .hdf5 file

import glob
import h5py
import os 

# Recursive function
def add_hdf5(input_hdf5, key, output_hdf5):
	if key in output_hdf5:
		return
	if type(input_hdf5[key]) == h5py.Group:
		out_group = output_hdf5.require_group(key)
		for child_key in input_hdf5[key]:
			add_hdf5(input_hdf5[key], child_key, out_group)
		for key, value in input_hdf5[key].attrs.items():
			out_group.attrs[key] = value
	elif type(input_hdf5[key]) == h5py.Dataset:
		output_hdf5.create_dataset(key, data=input_hdf5[key][()])
	else:
		raise TypeError(type(input_hdf5[key]))

# combines all hdf5 from a folder into a single hdf5 file
def folder2hdf5(folder_location, output_file_name):
	# find all hdf5 files in folder
	hdf5_files = glob.glob('%s/*.hdf5' % folder_location)
	# Create an output hdf5 file
	output_hdf5 = h5py.File('%s/%s' % (folder_location, output_file_name), 'w')
	# loop over all folders
	for input_hdf5path in hdf5_files:
		input_hdf5 = h5py.File(input_hdf5path, 'r')
		for entry_id in input_hdf5.keys():
			add_hdf5(input_hdf5, entry_id, output_hdf5)
		os.remove(input_hdf5path)

if __name__ == "__main__":
	folder2hdf5(folder_location='example_output_folder', output_file_name='example_file.hdf5')
