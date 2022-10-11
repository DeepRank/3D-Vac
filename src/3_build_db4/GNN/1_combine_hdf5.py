# This script takes as input different .hdf5 files and combines them into one .hdf5 file

import glob
import h5py
import os 

def add_hdf5(input_hdf5, key, output_hdf5):
	'''
	Recursive function that takes as input:

	.hdf5 input file path, generated using 0_generate_hdf5.py

	single key of .hdf5 file

	.hdf5 output file path, in which all .hdf5 files data will be stored
	'''
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


if __name__ == "__main__":

	####### please modify here #######
	run_day = '13072022'
	project_folder = '/projects/0/einf2380/'
	data = 'pMHCI'
	task = 'BA'
	resolution = 'residue' # either 'residue' or 'atomic'
	##################################

	output_folder = f'{project_folder}data/{data}/features_output_folder/GNN/{resolution}/{run_day}'
	output_file_name=f'{resolution}.hdf5'

	# find all hdf5 files in folder
	hdf5_files = glob.glob(f'{output_folder}/*.hdf5')
	# Create an output hdf5 file
	output_hdf5 = h5py.File(f'{output_folder}/{output_file_name}', 'w')
	# loop over all folders
	for input_hdf5path in hdf5_files:
		input_hdf5 = h5py.File(input_hdf5path, 'r')
		for entry_id in input_hdf5.keys():
			add_hdf5(input_hdf5, entry_id, output_hdf5)
		os.remove(input_hdf5path)