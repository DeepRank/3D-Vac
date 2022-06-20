# This script defines various classes that will be used to implement a first experimental GNN (Binding_GNN)

import torch.nn as nn
import torch

#################################################################################
# Single GNN-layer                                                              #
# Problem specific specialized GNN classes, can make use of these more general  #
# layers. (see examples below)                                                  #
# These layers can be used for both "normal" and equivariant GNN's              #
#################################################################################
class GNN_layer(nn.Module):
	def __init__(self, nmb_edge_projection, nmb_hidden_attr, nmb_output_features, 
				 	   message_vector_length, nmb_mlp_neurons, act_fn = nn.SiLU(), 
				 	   is_last_layer=True, is_equivariant=False):
		super(GNN_layer, self).__init__()
		# The MLP that takes in atom-pairs and creates the Mij's
		self.edge_mlp = nn.Sequential(
			nn.Linear(nmb_edge_projection + nmb_hidden_attr*2, nmb_mlp_neurons),
			act_fn,
			nn.Linear(nmb_mlp_neurons, message_vector_length),
			act_fn)

		# The node-MLP, creates a new node-representation given the Mi's
		self.node_mlp = nn.Sequential(
			nn.BatchNorm1d(message_vector_length + nmb_hidden_attr),
			nn.Linear(message_vector_length + nmb_hidden_attr, nmb_mlp_neurons),
			act_fn,
			nn.Linear(nmb_mlp_neurons, nmb_mlp_neurons),
			act_fn,
			nn.Linear(nmb_mlp_neurons, nmb_hidden_attr),
			)

		# Only last layer have attention and output modules
		if is_last_layer:
			# attention mlp, this to create a final single value prediction given all nodes from the peptide only
			self.attention_mlp = nn.Sequential(
				nn.Linear(nmb_hidden_attr, nmb_mlp_neurons),
				act_fn,
				nn.Linear(nmb_mlp_neurons, 1),
				nn.Sigmoid(),
				)

			# Create the output vector per node we are interested in
			self.output_mlp = nn.Sequential(
				nn.Linear(nmb_hidden_attr, nmb_mlp_neurons),
				act_fn,
				nn.Linear(nmb_mlp_neurons, nmb_output_features),
				)

		# Equivariant graph networks can output vectors based on input coordinates
		if is_equivariant:
			# To calculate vector magnitude, 
			self.vector_magnitude = nn.Sequential(
				nn.Linear(nmb_hidden_attr, nmb_mlp_neurons),
				act_fn,
				nn.Linear(nmb_mlp_neurons, 1),
				)

	# MLP that takes in the node-attributes of nodes (source + target), the edge attributes 
	# and node attributes in order to create a 'message vector'between those nodes
	def edge_model(self, edge_attr, hidden_features_source, hidden_features_target):
		cat = torch.cat([edge_attr, hidden_features_source, hidden_features_target], dim=1)
		output = self.edge_mlp(cat)
		return output

	# A function that updates the node-attributes. Assumed that submessages are already summed
	def node_model(self, summed_edge_message, hidden_features):
		cat = torch.cat([summed_edge_message, hidden_features], dim=1)
		output = self.node_mlp(cat)
		return hidden_features + output

	# Sums the individual sub-messages (multiple per node) into singel message vector per node
	def sum_messages(self, edges, messages, nmb_nodes):
		row, col = edges
		summed_messages_shape = (nmb_nodes, messages.size(1))
		result = messages.new_full(summed_messages_shape, 0)  # Init empty result tensor.
		row = row.unsqueeze(-1).expand(-1, messages.size(1))
		result.scatter_add_(0, row, messages)
		return result

	# Runs the GNN
	# steps is number of times it exanges info with neighbors, should be removed in future
	def update_nodes(self, edges, edge_attr, hidden_features, steps=1):
		row, col = edges # a single edge is defined as the index of atom1 and the index of atom2
		h = hidden_features # shortening the variable name
		for step in range(steps):
			node_pair_messages = self.edge_model(edge_attr, h[row], h[col]) # get all atom-pair messages
			messages = self.sum_messages(edges, node_pair_messages, len(h)) # sum all messages per node to single message vector
			h = self.node_model(messages, h) # Use the messages to update the node-attributes
		return h

	# output, every node creates a prediction + an estimate how sure it is of its prediction. Only done by last 'GNN layer'
	def output(self, hidden_features, get_attention=True):
		output = torch.tanh(self.output_mlp(hidden_features))
		if get_attention:
			return output, self.attention_mlp(hidden_features)
		return output

	# Equivariant output, v=coordinate_atom1-coordinate_atom2
	def equivariant_output(self, hidden_features, v, normalize=False):
		output = self.vector_magnitude(hidden_features)
		if normalize:
			v = v.norm(dim=1)
		return output * v

################################################################
# GNN with simple linear projection for edges and nodes        #
################################################################
class SimpleGNN(nn.Module):
	def __init__(self, nmb_edge_attr, nmb_node_attr, nmb_hidden_attr, 
				 nmb_mlp_neurons, nmb_edge_projection, nmb_gnn_layers, 
				 nmb_output_features, message_vector_length, act_fn = nn.SiLU()):
		super(SimpleGNN, self).__init__()

		# Linear projection of features to desired dimensions
		self.project_edge = nn.Sequential(
								nn.Linear(nmb_edge_attr, nmb_edge_projection),
								act_fn,
								)

		# Project the node_attributes to the same size as the hidden vector
		self.project_node = nn.Sequential(
								nn.Linear(nmb_node_attr, nmb_hidden_attr),
								)

		self.modlist  = nn.ModuleList(
						[GNN_layer(nmb_edge_projection, nmb_hidden_attr, nmb_output_features, 
						 message_vector_length, nmb_mlp_neurons,
						 is_last_layer=(gnn_layer==(nmb_gnn_layers-1)))
						 for gnn_layer in range(nmb_gnn_layers)])

	# always use this function before running the GNN layers
	def preprocess(self, edge_attr, node_attr):
		edge_attr = self.project_edge(edge_attr)
		hidden_features = self.project_node(node_attr)
		return edge_attr, hidden_features

	# Runs data through layers
	def runThroughNetwork(self, edges,edge_attr, node_attr, with_output_attention=False):
		edge_attr, node_attr = self.preprocess(edge_attr, node_attr)
		for layer in self.modlist:
			node_attr = layer.update_nodes(edges, edge_attr, node_attr)
		representations = self.modlist[-1].output(node_attr, with_output_attention)
		return representations	

################################################################
# GNN superclass, which habdles the shared preprocessing steps #
# All GNN's will inherit from this class                       #
################################################################
class PreprocessGNN(SimpleGNN):
	def __init__(self, nmb_edge_attr, nmb_node_attr, nmb_hidden_attr, 
				 nmb_mlp_neurons, nmb_edge_projection, nmb_gnn_layers, 
				 nmb_output_features, message_vector_length, act_fn = nn.SiLU()):
		super(PreprocessGNN, self).__init__(nmb_edge_attr, nmb_node_attr, 
				 nmb_hidden_attr, nmb_mlp_neurons, nmb_edge_projection, 
				 nmb_gnn_layers, nmb_output_features, message_vector_length, act_fn)

		# Since edge_atributes go into every layer, it might be betetr to learn
		# a better/smarter representation of them first
		self.project_edge = nn.Sequential(
								nn.BatchNorm1d(nmb_edge_attr),
								nn.Linear(nmb_edge_attr, nmb_mlp_neurons),
								nn.BatchNorm1d(nmb_mlp_neurons),
								act_fn,
								nn.Linear(nmb_mlp_neurons, nmb_edge_projection),
								act_fn,
								)

		# Project the node_attributes to the same size as the hidden vector
		self.project_node = nn.Sequential(
								nn.BatchNorm1d(nmb_node_attr),
								nn.Linear(nmb_node_attr, nmb_mlp_neurons),
								nn.BatchNorm1d(nmb_mlp_neurons),
								act_fn,
								nn.Linear(nmb_mlp_neurons, nmb_hidden_attr),
								act_fn,
								)

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