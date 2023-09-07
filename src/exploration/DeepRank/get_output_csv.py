import h5py 
# import argparse

# arg_parser.add_argument("--input-path", "-i",
#     help="Input hdf5 file",
#     required=True,
# )

# arg_parser.add_argument("--keys-path", "-k",
#     help="hdf5 file with keys",
#     required=True,
# )

# arg_parser.add_argument("--output-path", "-o",
#     help="Output csv file",
#     required=True,
# )

# args = arg_parser.parse_args()

def get_output_csv(input_path, keys_path, output_path):
    outputs_f = h5py.File(input_path)
    keys_f = h5py.File(keys_path)

    with open(output_path, 'w') as outfile:
        outfile.write('KEY,OUTPUT_0,OUTPUT_1,TARGET\n')
        for key, output, target in zip(keys_f.keys(), outputs_f['epoch_0014/test']['outputs'], outputs_f['epoch_0014/test']['targets']):
            #print(key, output)
            outfile.write(f'{key},{output[0]:.3f},{output[1]:.3f},{target}\n')
            
if __name__=='__main__':
    
    models = [('/projects/0/einf2380/data/pMHCI/trained_models/CNN/shuffled_noPSSM_sumFeat_bn/CnnClass4ConvKS3Lin128ChannExpand/metrics.hdf5',
               '/projects/0/einf2380/data/pMHCI/features_output_folder/CNN/splits_HLA_quant_shuffled/shuffled/0/test.hdf5',
               '/projects/0/einf2380/data/pop_paper_data/CNN_outputs/shuffled_cnn_outputs.csv'),
              
              ('/projects/0/einf2380/data/pMHCI/trained_models/CNN/clustPept_noPSSM_sumFeat_bn/CnnClass4ConvKS3Lin128ChannExpand/metrics.hdf5',
               '/projects/0/einf2380/data/pMHCI/features_output_folder/CNN/splits_HLA_quantitative_gibbs_clust_10_3/clustered/0/test.hdf5',
               '/projects/0/einf2380/data/pop_paper_data/CNN_outputs/peptide_cnn_outputs.csv'),
              
              ('/projects/0/einf2380/data/pMHCI/trained_models/CNN/clustAllele_noPSSM_sumFeat_bn/CnnClass4ConvKS3Lin128ChannExpand/metrics.hdf5',
               '/projects/0/einf2380/data/pMHCI/features_output_folder/CNN/splits_HLA_quant_allele_clusters/clustered/0/test.hdf5',
               '/projects/0/einf2380/data/pop_paper_data/CNN_outputs/allele_cnn_outputs.csv')]
    
    for model in models:
        get_output_csv(*model)
        
