import torch.multiprocessing as mp
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import warnings
import torch
import glob
import time
import os

import pylab as plt

# for REPRODUCIBILITY
warnings.filterwarnings("ignore")

# The Rotation module with all learnable parameters
# Matrix from: https://en.wikipedia.org/wiki/Rotation_matrix
class Rotation(nn.Module):
	
	def __init__(self, coordinates, mean='calculate'):
		super(Rotation, self).__init__()
		self.mean = coordinates.mean(1).reshape(-1,1,3) if mean=='calculate' else mean
		self.coordinates = coordinates.clone() - self.mean
		self.template = 0
		self.initParameters()
		
	# Get the rotation matrix using the learned angles
	def _getMatrix(self):
		# initialize the rotation matrices
		rotationMatrix = torch.zeros(self.coordinates.shape[0],3,3)
		# Apply the cos and sin to the parameters
		cosa = self.ar.cos()
		cosb = self.br.cos()
		cosy = self.yr.cos()
		sina = self.ar.sin()
		sinb = self.br.sin()
		siny = self.yr.sin()
		# Fill the matrices
		rotationMatrix[:,0,1] =  cosa*sinb*siny - sina*cosy
		rotationMatrix[:,0,2] =  cosa*sinb*cosy + sina*siny
		rotationMatrix[:,1,1] =  sina*sinb*siny + cosa*cosy
		rotationMatrix[:,1,2] =  sina*sinb*cosy - cosa*siny
		rotationMatrix[:,0,0] =  cosa*cosb
		rotationMatrix[:,1,0] =  sina*cosb
		rotationMatrix[:,2,1] =  cosb*siny
		rotationMatrix[:,2,2] =  cosb*cosy
		rotationMatrix[:,2,0] = -sinb
		return rotationMatrix

	# Get the rotated coordinates
	def forward(self):
		# Creates the rotation matrix
		rotationMatrix = self._getMatrix()
		# Apply the matrix
		newCoordinates = torch.matmul(self.coordinates, rotationMatrix)
		newCoordinates = newCoordinates + self.bias
		return newCoordinates
	
	# Returns all learnable parameters for optimizer
	def parameters(self):
		return [self.ar, self.br, self.yr, self.bias]
	
	def initParameters(self):
		nmbr = self.coordinates.shape[0]
		self.ar   = torch.zeros(nmbr) * 6.28 
		self.br   = torch.zeros(nmbr) * 6.28
		self.yr   = torch.zeros(nmbr) * 6.28
		self.bias = torch.zeros(nmbr, 1, 3) # Random translation in xyz directions
		
		self.ar.requires_grad_()  # This enables gradients for these parameters
		self.br.requires_grad_()  # This enables gradients for these parameters
		self.yr.requires_grad_()  # This enables gradients for these parameters
		self.bias.requires_grad_()# This enables gradients for these parameters
		

# Class to process pdbs
class PDB2dataset():
	
	def __init__(self,pdbInpFolder, pdbRotatedFolder, residues, chain='A', cores=-1): 
		try: os.mkdir(pdbRotatedFolder)
		except: pass
		self.chain = chain
		self.residues = [int(i) for i in residues]
		self.pdbRotatedFolder = pdbRotatedFolder
		self.pdbInpFolder = pdbInpFolder
		self.pdbs = glob.glob(pdbInpFolder)
		# self.pdbs = self.pdbs[0:5];
		self.pdbs.append("/projects/0/einf2380/data/pMHCI/models/alignment/alignment_template.pdb")
		self.pdbIds = [pdb for pdb in self.pdbs]
		self.cores = cores
		print('Start loading data')
		t0 = time.time()
		coordinates, self.indiWeights = self.getData()
		print('Retrieved data, time: %.2f seconds' % (time.time()-t0))
		#coordinates, self.indiWeights = torch.rand(10000, 100, 3), 1
		self.rotator = Rotation(coordinates)
		
	# Extract all atom xyz coordinates
	def _extractPdbAtoms(self, fileName):
		return [[float(line[30:38]),float(line[38:46]),float(line[46:54])] \
				for line in open(fileName).read().split('\n') \
						if line.startswith('ATOM ') or line.startswith('HETATM ')]
	
	# Extract all residue sequence numbers and their C-alfa xyz coordinates
	def _extractPdbCa(self, fileName):
		return [[int(line[22:26]),[float(line[30:38]),float(line[38:46]),float(line[46:54])]] \
				for line in open(fileName).read().split('\n') \
						if line.startswith('ATOM ') and line[13:15]=='CA' and line[21]==self.chain and int(line[22:26]) in self.residues]
	
	# Parse the PDBs to obtain the data needed for calculating the rotation matrices
	# TODO make it poolalble by processing one file to tensor object and then cncatenate all tensors...
	def getData(self):
		pool = mp.Pool(self.cores)
		#pdb_db    = [self._extractPdbCa(pdb) for pdb in self.pdbs]
		pdb_db = pool.map(self._extractPdbCa, self.pdbs)
		#pdb_db    = [[atom for atom in pdb if atom[0] in self.residues] for pdb in pdb_db]
		trainData = torch.zeros(len(self.pdbs), len(self.residues), 3)
		weights   = torch.zeros(len(self.pdbs), len(self.residues), 1)
		for i, pdb in enumerate(pdb_db):
			for atom in pdb:
				ind = self.residues.index(atom[0])
				trainData[i, ind] = torch.Tensor(atom[1])
				weights[i, ind] = 1
		return trainData, weights 
	
	# Rotates a single pdb
	def _rotate(self, pdbIndex):
		with torch.no_grad():
			pdbFile = self.pdbs[pdbIndex]
			coordinates = torch.Tensor(self._extractPdbAtoms(pdbFile)).reshape(1, -1, 3)
			newRotator = Rotation(coordinates, self.rotator.mean[pdbIndex])
			newRotator.ar   = self.rotator.ar[pdbIndex]
			newRotator.br   = self.rotator.br[pdbIndex]
			newRotator.yr   = self.rotator.yr[pdbIndex]
			newRotator.bias = self.rotator.bias[pdbIndex] + \
								self.rotator.mean[self.rotator.template]
			newCts = newRotator().squeeze()
			
			atomIndex = 0
			pdb = open(pdbFile).read().split('\n')[:-1]
			write = open(pdbFile, 'w')
			for line in pdb:
				if line.startswith('ATOM') or line.startswith('HETATM '):
					stringCoord = ['%.3f' % newCts[atomIndex, 0], \
									'%.3f' % newCts[atomIndex, 1], \
									 '%.3f' % newCts[atomIndex, 2]] 
					stringCoord = ''.join([(' '*(8-len(i)))+i for i in stringCoord])
					line = line[:30] + stringCoord + line[54:]
					atomIndex += 1
				write.write(line + '\n')					
			write.close()
	
	# Rotate all, or a selection of pdbs
	def rotateAll(self, nmbr=-1):
		t0 = time.time()
		nmbr = min(nmbr, len(self.pdbs)) if nmbr != -1 else len(self.pdbs)
		self.rotator.ar = self.rotator.ar.detach()
		self.rotator.br = self.rotator.br.detach()
		self.rotator.yr = self.rotator.yr.detach()
		self.rotator.bias = self.rotator.bias.detach()
		self.rotator.mean = self.rotator.mean.detach()
		self.rotator.coordinates = self.rotator.coordinates.detach()
		# pool = mp.Pool()
		# pool.map(self._rotate, range(nmbr))
		for pdbIndex in range(nmbr):
			self._rotate(pdbIndex)
		# _ = [self._rotate(pdbIndex) for pdbIndex in range(nmbr)] 
		print('Rotating and saving data took: %.2f seconds' % (time.time()-t0))
		
	# Train the function
	def train(self, templatePdb=-1):
		# print('whatdoes', templatePdb)
		tempInd = 0 if templatePdb == -1 else self.pdbIds.index(templatePdb)
		optimizer  = torch.optim.Adam(self.rotator.parameters(), lr=.8) # Create optimizer
		self.rotator.template = tempInd
		t0 = time.time()
		bestLoss = torch.zeros(len(self.rotator.mean)) + 10e6 # for keeping track of lowest loss
		label = self.rotator.coordinates[tempInd:tempInd+1]
		ar = self.rotator.ar.clone()
		br = self.rotator.br.clone()
		yr = self.rotator.yr.clone()
		bias = self.rotator.bias.clone()
		changeLog = 0
		for epoch in range(0, 10000000):
			optimizer.zero_grad() # Clear old gradients
			newCoordinates = self.rotator() # get rotated coordinates
			loss = F.mse_loss(newCoordinates, label, reduction='none') * self.indiWeights
			sampleLoss = loss.mean(1).mean(1)
			# Check if new minimum has been found
			log = (sampleLoss+0.001) < bestLoss
			if torch.all(torch.logical_not(log)):
				changeLog += 1
			else:
				changeLog  = 0
			bestLoss[log] = sampleLoss[log].detach()
			if changeLog > 25:
				break
			ar[log] = self.rotator.ar[log].clone()
			br[log] = self.rotator.br[log].clone()
			yr[log] = self.rotator.yr[log].clone()
			bias[log] = self.rotator.bias[log].clone()
			sampleLoss.mean().backward() # Calculate gradients
			optimizer.step() # Update learnable parameters
			#if epoch % 60 == 0:
			#	self.rotator.bias.data *= 0
		print('Epochs:', epoch, ', train-time: %.2f seconds!' % (time.time()-t0))
		self.rotator.ar, self.rotator.br, self.rotator.yr, self.rotator.bias = ar, br, yr, bias
		
def align(pdbInpFolder, residues, pdbRotatedFolder='modelsR', chain='M', template = -1, nmbr= -1):
	dataProcessor = PDB2dataset(pdbInpFolder, pdbRotatedFolder, residues, chain, 128)	
	dataProcessor.train(template)
	dataProcessor.rotateAll(nmbr = nmbr)


def extractPdbCa(fileName, chain):
		return [[float(line[30:38]),float(line[38:46]),float(line[46:54])] \
				for line in open(fileName).read().split('\n') \
						if line.startswith('ATOM ') and line[13:15]=='CA' and line[21]==chain]

# generate the
align('/projects/0/einf2380/data/pMHCI/pssm_mapped/BA/*/*/pdb/*.pdb', range(100), 
	template = '/projects/0/einf2380/data/pMHCI/models/alignment/alignment_template.pdb', 
	pdbRotatedFolder='/projects/0/einf2380/data/pMHCI/models/pssm_mapped/BA',
)




