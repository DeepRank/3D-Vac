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
    test_keys_f = h5py.File(keys_path + '/test.hdf5')
    valid_keys_f = h5py.File(keys_path + '/valid.hdf5')

    last_epoch = sorted([x for x in outputs_f.keys() if x.startswith('epoch')])[-1]
    print(last_epoch)
    
    with open(output_path, 'w') as outfile:
        outfile.write('KEY,PHASE,OUTPUT_0,OUTPUT_1,TARGET\n')
        for key, output, target in zip(valid_keys_f.keys(), outputs_f[f'{last_epoch}/valid']['outputs'], outputs_f[f'{last_epoch}/valid']['targets']):
            #print(key, output)
            outfile.write(f'{key},validation,{output[0]:.3f},{output[1]:.3f},{target}\n')
        for key, output, target in zip(test_keys_f.keys(), outputs_f[f'{last_epoch}/test']['outputs'], outputs_f[f'{last_epoch}/test']['targets']):
            #print(key, output)
            outfile.write(f'{key},testing,{output[0]:.3f},{output[1]:.3f},{target}\n')
            
if __name__=='__main__':
    
    # models = [('/projects/0/einf2380/data/pMHCI/trained_models/CNN/shuffled_noPSSM_sumFeat_bn/CnnClass4ConvKS3Lin128ChannExpand/metrics.hdf5',
    #            '/projects/0/einf2380/data/pMHCI/features_output_folder/CNN/splits_HLA_quant_shuffled/shuffled/0',
    #            '/projects/0/einf2380/data/pop_paper_data/cnn_outputs/shuffled_cnn_outputs.csv'),
              
    #           ('/projects/0/einf2380/data/pMHCI/trained_models/CNN/clustPept_noPSSM_sumFeat_bn/CnnClass4ConvKS3Lin128ChannExpand/metrics.hdf5',
    #            '/projects/0/einf2380/data/pMHCI/features_output_folder/CNN/splits_HLA_quantitative_gibbs_clust_10_3/clustered/0',
    #            '/projects/0/einf2380/data/pop_paper_data/cnn_outputs/peptide_cnn_outputs.csv'),
              
    #           ('/projects/0/einf2380/data/pMHCI/trained_models/CNN/clustAllele_noPSSM_sumFeat_bn/CnnClass4ConvKS3Lin128ChannExpand/metrics.hdf5',
    #            '/projects/0/einf2380/data/pMHCI/features_output_folder/CNN/splits_HLA_quant_allele_clusters/clustered/0',
    #            '/projects/0/einf2380/data/pop_paper_data/cnn_outputs/allele_cnn_outputs.csv')]
    models = []
    for fold in range(1,6):
        models.append((f'/projects/0/einf2380/data/pMHCI/trained_models/CNN/shuffled_Cnn/CnnClass4ConvKS3Lin128ChannExpand/{fold}/metrics.hdf5',
                f'/projects/0/einf2380/data/pMHCI/features_output_folder/CNN/CrossValidation/Shuffled/{fold}',
                f'/projects/0/einf2380/data/pop_paper_data/cnn_outputs/shuffled_crossval/shuffled_cnn_outputs_fold_{fold}.csv'))
        
        models.append((f'/projects/0/einf2380/data/pMHCI/trained_models/CNN//clustAllele_Cnn/CnnClass4ConvKS3Lin128ChannExpand/{fold}/metrics.hdf5',
                f'/projects/0/einf2380/data/pMHCI/features_output_folder/CNN/CrossValidation/AlleleClustered/{fold}',
                f'/projects/0/einf2380/data/pop_paper_data/cnn_outputs/allele_crossval/allele_cnn_outputs_fold_{fold}.csv'))
    
    for model in models:
        get_output_csv(*model)
        
