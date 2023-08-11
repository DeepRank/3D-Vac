from deeprank.generate import *
#from sequence_feature import SequenceFeature
import os

h5file = '/home/dmarz/temp/test.hdf5'
os.popen(f'rm {h5file}').read()
os.popen(f'cp /projects/0/einf2380/data/pMHCI/features_output_folder/CNN/hla_a_02_01_9_length_peptide/000_hla_a_02_01_9_length_peptide.hdf5 {h5file}').read()


database = DataGenerator(compute_features=['sequence_feature'],
                         hdf5=h5file, chain1='M', chain2='P')

database.add_feature()
database.map_features()