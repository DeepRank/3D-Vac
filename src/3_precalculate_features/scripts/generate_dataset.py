#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 15:40:53 2021

@author: Dario
"""

import sys
sys.path.append('./')
from mpi4py import MPI
from deeprank.generate import *


comm = MPI.COMM_WORLD
rank = comm.Get_rank()


#targets = ['T29','T30', 'T32', 'T35', 'T37', 'T39', 'T40',
#           'T41', 'T46', 'T47', 'T50', 'T53', 'T54']

#target = targets[rank]
target = '10'

#for target in targets:
'''
try:
    os.mkdir('%s_native' %target)
except FileExistsError:
    pass
os.system('cp /projects/0/deeprank/CAPRI_scoreset/%s/pdb/%s_0_conv.pdb ./%s_native' %(target, target, target))

try:
    os.mkdir('%s_pssm' %target)
except:
    pass
os.system('cp /projects/0/deeprank/CAPRI_scoreset/%s/pssm/%s_0_conv.?.pdb.pssm ./%s_pssm' %(target, target, target))
'''

pdb_source = ['./pdb/mixed']
#pdb_native = ['./t35_native']
#pssm_source= './Edesolv/test_pssm'
h5out = './hdf5/test.hdf5'
database = DataGenerator(pdb_source=pdb_source,
                            #pdb_native=pdb_native,
                            #pssm_source=pssm_source,
                            chain1='M',
                            chain2='P',
                            compute_features=['deeprank.features.AtomicFeature',
                                              'deeprank.features.BSA',
                                              'deeprank.features.Edesolv'],
                                              hdf5=h5out,
                            mpi_comm=comm)

database.create_database(prog_bar=False)
