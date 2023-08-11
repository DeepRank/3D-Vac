import sys
import h5py
import argparse
import os.path as path
import os
import sys
sys.path.append(path.abspath("../../../../"))
from CNN.CNN_models import *
from CNN.NeuralNet import NeuralNet
#from deeprank.learn.modelGenerator import *

def get_cnn_outputs(test_hdf5, architecture, pretrained_model, outdir, with_cuda = True):
    # try:
    #     os.makedirs(outdir)
    # except FileExistsError:
    #     pass

    model = NeuralNet(test_hdf5,
        model = architecture,
        cuda = bool(with_cuda),
        ngpu = (0,1)[with_cuda],
        outdir = outdir,
        pretrained_model=pretrained_model,
    )

    model.test()

#Shuffled
indir = '/projects/0/einf2380/data/pMHCI/trained_models/CNN/shuffled_noPSSM_sumFeat_bn/CnnClass4ConvKS3Lin128ChannExpand/'
get_cnn_outputs(
    test_hdf5= '/projects/0/einf2380/data/pMHCI/features_output_folder/CNN/splits_HLA_quant_shuffled/shuffled/0/test.hdf5',
    pretrained_model = indir + 'best_valid_model.pth.tar',
    architecture = CnnClassGroupConv,
    outdir = indir
    )

#Peptides
indir = '/projects/0/einf2380/data/pMHCI/trained_models/CNN/clustPept_noPSSM_sumFeat_bn/CnnClass4ConvKS3Lin128ChannExpand/'
get_cnn_outputs(
    test_hdf5= '/projects/0/einf2380/data/pMHCI/features_output_folder/CNN/splits_HLA_quantitative_gibbs_clust_10_3/clustered/0/test.hdf5',
    pretrained_model = indir +  'best_valid_model.pth.tar',
    architecture = CnnClassGroupConv,
    outdir = indir
    )

#Alleles
indir = '/projects/0/einf2380/data/pMHCI/trained_models/CNN/clustAllele_noPSSM_sumFeat_bn/CnnClass4ConvKS3Lin128ChannExpand/'
get_cnn_outputs(
    test_hdf5= '/projects/0/einf2380/data/pMHCI/features_output_folder/CNN/splits_HLA_quant_allele_clusters/clustered/0/test.hdf5',
    pretrained_model = indir + 'best_valid_model.pth.tar',
    architecture = CnnClassGroupConv,
    outdir = indir
    )