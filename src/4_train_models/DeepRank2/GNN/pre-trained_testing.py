# Script for processing and testing PDB files using a pre-trained deeprank2 model

import os
import glob
import sys
import logging
from deeprank2.query import QueryCollection
import pandas as pd
from deeprank2.trainer import Trainer
from deeprank2.dataset import GraphDataset
from pmhc_gnn import NaiveGNN1
from deeprank2.utils.exporters import HDF5OutputExporter

####### Editable parameters #######
## Query
# path to the pdb(s) to be processed with deeprank2
# note that all the pdbs contained in pdb_input_path will be processed
pdb_input_path = 'test_data/pdb'
pretrained_model_path = "exp_100k_std_transf_bs64_naivegnn1_wloss_all_data_0_231006.pth.tar"
output_path = 'test_data/test1'
resolution = 'residue' # either 'residue' or 'atomic'
interface_distance_cutoff = 15  # max distance in Å between two interacting residues/atoms of two proteins
chain_id1 = 'M'
chain_id2 = 'P'
cpu_count = None # with None will take the cpus available on the machine
feature_modules = ['components',
		   'contact',
		   'exposure',
		   'irc',
		   'surfacearea']
combine_output = False
grid_settings = None
grid_map_method = None
# grid_settings = GridSettings( # None if you don't want grids
# 	# the number of points on the x, y, z edges of the cube
# 	points_counts = [35, 30, 30],
# 	# x, y, z sizes of the box in Å
# 	sizes = [1.0, 1.0, 1.0])
# grid_map_method = MapMethod.GAUSSIAN # None if you don't want grids
## Trainer
net = NaiveGNN1
cuda = False
ngpu = 0
##################################

def data_processing():

	pdb_files = glob.glob(os.path.join(pdb_input_path, '*.pdb'))

	queries = QueryCollection()
	_log.info(f'Adding {len(pdb_files)} queries to the query collection ...')
	count = 0
	for i in range(len(pdb_files)):
		queries.add(
			PPIQuery(
				pdb_path = pdb_files[i], 
				chain_id1 = chain_id1,
				chain_id2 = chain_id2,
				distance_cutoff = interface_distance_cutoff))
		count +=1
		if count % 10000 == 0:
			_log.info(f'{count} queries added to the collection.')

	_log.info(f'Queries ready to be processed.\n')
	queries.process(
		f'{output_path}/{resolution}',
		feature_modules = feature_modules,
		cpu_count = cpu_count,
		combine_output = combine_output,
		grid_settings = grid_settings,
		grid_map_method = grid_map_method)
	
	_log.info(f'The queries processing is done. The generated hdf5 files are in {output_path}.')

def run_pre_trained():

	dataset_test = GraphDataset(
		hdf5_path = glob.glob(os.path.join(output_path, '*.hdf5')), 
		train = False,
		train_data = pretrained_model_path
	)

	_log.info('Processed data loaded and ready to be tested.')

	trainer = Trainer(
		neuralnet = net,
		dataset_test = dataset_test,
        pretrained_model = pretrained_model_path,
        cuda = cuda,
        ngpu = ngpu,
		output_exporters = [HDF5OutputExporter(output_path)])

	trainer.test()


if __name__ == "__main__":

	if resolution == 'atomic':
		from deeprank2.query import ProteinProteinInterfaceAtomicQuery as PPIQuery
	else:
		from deeprank2.query import ProteinProteinInterfaceResidueQuery as PPIQuery

	if not os.path.exists(output_path):
		os.makedirs(output_path)
	else:
		if len(os.listdir(output_path)) > 0:
			sys.exit(f'{output_path} already exists and is not empty, please remove the existing files or change output path!')

	# Loggers
	_log = logging.getLogger('')
	_log.setLevel(logging.INFO)

	fh = logging.FileHandler(os.path.join(output_path, 'pre-trained_testing.log'))
	sh = logging.StreamHandler(sys.stdout)
	fh.setLevel(logging.INFO)
	sh.setLevel(logging.INFO)
	formatter_fh = logging.Formatter('[%(asctime)s] - %(name)s - %(message)s',
								datefmt='%a, %d %b %Y %H:%M:%S')
	fh.setFormatter(formatter_fh)
	_log.addHandler(fh)
	_log.addHandler(sh)

	_log.info('\nScript running has started ...')
	data_processing()
	run_pre_trained()
