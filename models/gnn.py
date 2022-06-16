from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import torch.nn.functional as F
from gnn_architecture import *
from dset2arrays import *
import torch.nn as nn
import pylab as plt
import random
import torch


# Number of cores the program is using
torch.set_num_threads(20)

##############################
#         CONSTANTS          #
##############################
MESSAGE_VECTOR_LENGTH   = 32 #
NMB_HIDDED_ATTRIBUTES   = 32 #
NMB_EDGE_PROJECTION     = 32 #
NMB_OUPUT_FEATURES      =  2 #
NMB_MLP_NEURONS         = 32 #
NMB_GNN_LAYERS          =  3 #
##############################

##############################################
# GNN for binding prediction                 #
##############################################
class Binding_GNN(PreprocessGNN):
	def __init__(self, nmb_edge_attr, nmb_node_attr, nmb_output_features, 
				 nmb_hidden_attr, message_vector_length, nmb_mlp_neurons, 
				 nmb_gnn_layers, nmb_edge_projection, act_fn = nn.SiLU()):
		super(Binding_GNN, self).__init__(nmb_edge_attr, nmb_node_attr, 
				 nmb_hidden_attr, nmb_mlp_neurons, nmb_edge_projection, 
				 nmb_gnn_layers, nmb_output_features, message_vector_length, act_fn)
		
	# Run over all layers, and return the ouput vectors
	def forward(self, edges, edge_attr, node_attr):
		representations = self.runThroughNetwork(edges, edge_attr, node_attr, with_output_attention=False)
		# Get for each node the scores how likely they think it will bind or not bind
		# Sum for all elements:
		representations = representations.sum(0)
		return representations

# Run a sample though the network, and return the output and true label
def runNetwork(hdf5, samples, sample, gnn, edge_features, node_features):
	# Get the edges, edge_attributes and node-attributes
	edges = get_edges(hdf5, samples[sample]).T
	edge_attr = get_edge_values(hdf5, samples[sample], edge_features)
	node_attr = get_node_values(hdf5, samples[sample], node_features)
	label = getBinaryLabel(hdf5, samples[sample])
	gnn_output = gnn(edges, edge_attr, node_attr)
	return gnn_output, label

# Train a single epoch
def train_epoch(hdf5, train_samples, gnn, optimizer, edge_features, node_features):
	# The loss function, cross entropy
	loss_func = nn.CrossEntropyLoss()
	# A variable to keep track of the total loss
	tloss = 0
	labels = []
	predictions = []
	# Shuffle the list of sample-ids (strings)
	random.shuffle(train_samples)
	gnn.train()
	# Go over all samples:
	for sample in range(len(train_samples)):
		gnn_output, label = runNetwork(hdf5, train_samples, sample, gnn, edge_features, node_features)
		loss = loss_func(gnn_output, label)
		loss.backward()
		optimizer.step()
		optimizer.zero_grad()
		tloss += loss.item()
		predictions += [int(gnn_output.argmax())]
		labels += [int(label)]
	conf_matrix = confusion_matrix(labels, predictions)
	accuracy = accuracy_score(labels, predictions)
	return tloss/len(train_samples), accuracy, conf_matrix

# Validate the model
def validate(hdf5, validation_samples, gnn, edge_features, node_features):
	with torch.no_grad():
		gnn.eval()
		# The loss function, cross entropy
		loss_func = nn.CrossEntropyLoss()
		# Keep track of the predictions for the statistics
		predictions = []
		labels = []
		tloss = 0
		for sample in range(len(validation_samples)):
			gnn_output, label = runNetwork(hdf5, validation_samples, sample, gnn, edge_features, node_features)
			loss = loss_func(gnn_output, label)
			tloss += loss.item()
			predictions += [int(gnn_output.argmax())]
			labels += [int(label)]
		conf_matrix = confusion_matrix(labels, predictions)
		accuracy = accuracy_score(labels, predictions)
		gnn.train()
		return tloss/len(validation_samples), accuracy, conf_matrix

# Split the dataset randomly
def split_dataset_randomly(dataset, percentages=[0.8, 0.2, .1], shuffle=True):
	# Ensure the sum of percentages equals 1
	percentages = [i/sum(percentages)for i in percentages]
	# Shuffle the dataset if needed
	if shuffle:
		random.shuffle(dataset)
	outputs = []
	for i, n in enumerate(percentages):
		before = int(sum(percentages[:i]) * len(dataset))
		after = int(sum(percentages[:i+1]) * len(dataset))
		outputs += [dataset[before:after]]
	return outputs
	
# Split the dataset randomly
def split_dataset_cluster(dataset):
	pass

if __name__ == '__main__':

	# Name for saving plots and trained networks
	network_name = 'simple_preprocessing'
	
	# Get the dataset
	hdf5_location = '/home/daniel/binding_project/residueDataset/rd.hdf5'
	hdf5 = h5py.File(hdf5_location, 'r')
	# all the samples within a hdf5
	samples = [i for i in hdf5.keys()]

	# Used features
	# all edge features:  ['coulomb', 'covalent', 'dist', 'vanderwaals']
	edge_features = ['coulomb', 'covalent', 'dist', 'vanderwaals']
	# All node features: ['bsa', 'depth', 'hb_acceptors', 'hb_donors', 'hse', 'ic', 'polarity', 'pos', 'pssm', 'sasa', 'size', 'type']
	node_features = ['bsa', 'depth', 'hb_acceptors', 'hb_donors', 'hse', 'ic', 'pssm', 'polarity', 'sasa', 'size', 'type'] 
	
	# Calculate how many inputs the node and edge features have
	NMB_EDGE_ATTIBUTES  = get_edge_values(hdf5, samples[0], edge_features).shape[1]
	NMB_NODE_ATTRIBUTES = get_node_values(hdf5, samples[0], node_features).shape[1]

	# Init GNN
	gnn = Binding_GNN(nmb_edge_attr=NMB_EDGE_ATTIBUTES, nmb_node_attr=NMB_NODE_ATTRIBUTES, 
			  nmb_output_features=NMB_OUPUT_FEATURES, nmb_gnn_layers=NMB_GNN_LAYERS,
			  message_vector_length=MESSAGE_VECTOR_LENGTH, nmb_mlp_neurons=NMB_MLP_NEURONS,
			  nmb_hidden_attr=NMB_HIDDED_ATTRIBUTES, nmb_edge_projection=NMB_EDGE_PROJECTION)

	# Create optimizer, without weigh_decoy the model explodes
	optimizer = torch.optim.Adam(gnn.parameters(), weight_decay=1e-5)
	
	# Variables that keep track of the lowest validation loss and higest accuracy score
	lowestVal = 10e6
	highestAcc = -1
	
	# Shuffle samples
	random.shuffle(samples)
	
	# Get 80% for training and 20% for validation (we forget about test here)
	train, val, test = split_dataset_randomly(samples, [90, 10, 0])
	
	# Logs for the train and validation losses
	init_train_loss, init_train_accuracy, init_train_confusion = \
			validate(hdf5, train, gnn, edge_features, node_features)
	init_val_loss, init_val_accuracy, init_val_confusion = \
			validate(hdf5, val, gnn, edge_features, node_features)
	train_log = [init_train_loss]
	val_log = [init_val_loss]
	train_acc_log = [init_train_accuracy]
	val_acc_log = [init_val_accuracy]
	print('\n\ninit train loss:', init_train_loss)
	print('init train accuracy:', init_train_accuracy)
	print('init val loss:', init_val_loss)
	print('init val accuracy:', init_val_accuracy)
	print('\n'*2)
	# Loop n epochs
	total_epochs = 1000
	
	for epoch in range(total_epochs):
		# Caluclate 		
		train_loss, train_accuracy, train_confusion = train_epoch(hdf5, train, gnn, 
														optimizer, edge_features, node_features)
		val_loss, val_accuracy, val_confusion = validate(hdf5, val, gnn, edge_features, node_features)
		
		if val_loss < lowestVal:
			lowestVal = val_loss
			torch.save(gnn, '%s.pt' % network_name)
		if val_accuracy > highestAcc:
			highestAcc = val_accuracy
		# Fill the logs
		train_log += [train_loss]
		val_log += [val_loss]
		train_acc_log += [train_accuracy]
		val_acc_log += [val_accuracy]
		# Sometimes save the plot
		if (epoch) % 2 == 0:
			plt.plot(train_log, label='train-loss')
			plt.plot(val_log, label='validation-loss')
			plt.xlabel('epochs')
			plt.ylabel('loss')
			plt.legend(loc='upper right')
			plt.savefig('plots/loss_%s.png' % network_name,  dpi=300)
			plt.clf()
			plt.plot(train_acc_log, label='train-accuracy')
			plt.plot(val_acc_log, label='validation-accuracy')
			plt.xlabel('epochs')
			plt.ylabel('accuracy')
			plt.legend(loc='lower right')
			plt.savefig('plots/accuracy_%s.png' % network_name, dpi=300)
			plt.clf()
		print('####################################################################')
		print('\n\nepoch %i:\t' % epoch)
		print('\tTrain loss:\t', train_loss)
		print('\tTrain accuracy:\t',train_accuracy)
		print('\tConfusion train:\n', train_confusion, '\n')
		print('\tValidation loss:\t', val_loss)
		print('\tValidation accuracy:\t', val_accuracy)
		print('\tConfusion validation:\n', val_confusion)
		print('\n\tHigest validation accuracy:\t', highestAcc)
		#print('\nMultiplier:\t', gnn.multiplier.weight.item())
		
		
		


		
