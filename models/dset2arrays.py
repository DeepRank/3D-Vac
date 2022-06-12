import numpy as np
import torch
import h5py

def get_edges(hdf5, sample_key):
	edge_index = torch.tensor(np.array(hdf5[sample_key]['edge_index']))
	edge_index = torch.nan_to_num(edge_index.long())
	return edge_index[edge_index[:,0] != edge_index[:,1]]

def get_edge_values(hdf5, sample_key, features = ['coulomb', 'covalent', 'dist', 'vanderwaals']):
	edge_keys = hdf5[sample_key]['edge_data'].keys()
	edge_values = [torch.tensor(np.array( hdf5[sample_key]['edge_data'][key]))
					for key in features]
	for vi in range(len(edge_values)):
		if len(edge_values[vi].shape) == 1:
			edge_values[vi] = edge_values[vi].unsqueeze(1)
	edge_values = torch.cat(edge_values,1)
	edge_index = torch.tensor(np.array(hdf5[sample_key]['edge_index']))
	edge_values = edge_values[edge_index[:,0] != edge_index[:,1]]
	return torch.nan_to_num(edge_values.float())
	
def get_node_values(hdf5, sample_key, features=['bsa', 'depth', 'hb_acceptors', 'hb_donors', 'hse', 'ic', 'polarity', 'pos', 'pssm', 'sasa', 'size', 'type']):
	node_keys = hdf5[sample_key]['node_data'].keys()
	node_values = [torch.tensor(np.array( hdf5[sample_key]['node_data'][key]))
					for key in features]
	# Add features that node is peptide or mhc:
	#node_values += [torch.tensor([int(str(i).split(' ')[-2] == 'P') 
	#						 for i in list(hdf5[sample_key]['nodes'])])]
	for vi in range(len(node_values)):
		if len(node_values[vi].shape) == 1:
			node_values[vi] = node_values[vi].unsqueeze(1)
	node_values = torch.cat(node_values,1)
	return torch.nan_to_num(node_values.float())

def getBinaryLabel(hdf5, sample_key):
	label = torch.tensor(np.array([hdf5[sample_key]['score']['binary']][0]))
	return label.long()

def getBALabel(hdf5, sample_key):
	label = torch.tensor(np.array([hdf5[sample_key]['score']['BA']][0]))
	return label.long()

def getClusterLabel(hdf5, sample_key):
	label = torch.tensor(int(np.array(hdf5[sample_key]['score']['cluster'])))
	return label

def genPdb(positions):
	spherelist = []
	for i in positions:
		spherelist += [SPHERE,   float(i[0]), float(i[1]), float(i[2])]
	return spherelist

if __name__ == "__main__":
	# Define location hdf5
	filename = '/home/daniel/binding_project/residueDataset/residueDataset.hdf5'
	# Read hdf5
	f = h5py.File(filename, 'r')
	# all the samples within a hdf5
	samples = [i for i in f.keys()]
	# Take one as example
	sample = samples[0]
	a = get_edges(f, sample)
	b = get_edge_values(f, sample)
	c = get_node_values(f, sample)[:,12:15]
	print(getBinaryLabel(f, sample))

