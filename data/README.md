Structure of data in the repo: TBD

Structure of the data on Snellius (at `/projects/0/einf2380/data/pMHCI/`):

```
models                          -> DB2 (output of Pandora, .pdb files)
├── alignment
│   └── alignment_template.pdb      -> template for the align.py script, used to define optimized x,y,z coordinates of pdb files
├── BA -> models with a continuous value of binding affinity as a label
│   ├── x_y			                -> folder containing model number x to model number y
│   │   ├── BA_x_template	      -> folder containing structures of the model
│   │   │   ├── 3MRC.pdb
│   │   │   ├── BA_x.ali
│   │   │   ├── BA_x.BL*.pdb	  -> 20 stable structures generated by modeller
│   │   │   ├── BA_x.ini
│   │   │   ├── cmd_modeller_ini.py
│   │   │   ├── cmd_modeller.py
│   │   │   ├── contacts_BA_x.list
│   │   │   ├── modeller.log
│   │   │   ├── molpdf_DOPE.tsv	-> ranking of generated structures
│   │   │   └── MyLoop.py
│   │   └── ...
│   └──...			
├── EL				                  -> models with (0, 1) binding label with the same structure as BA
│
pssm_mapped			                -> DB3, (pssm mapped files and best Pandora's models, both inputs for deeprank-gnn-2)
├── BA
│   ├── x_y
│   │   ├── BA_x_template	      -> folder structure required by DeepRank/PSSMgen to map pssm
│   │   │   ├── pdb		          -> contains symlink to best model
│   │   │   ├── pssm		        -> contains 2 mapped .pssm files for chain M (the MHC) and chain P (peptide)
│   │   │   └── pssm_raw	      -> contains a symlink to pssm_raw/hla_a_02_01/hla.pssm and used for the mapping to the pdb structure in the ./pdb folder
│   │   ├── ...	 
│   └── ...
├──EL
│
pssm_raw			                  -> contains the raw pssm for each allele
├── hla_a_02_01			            -> HLA-A*02:01 is the allele used for the pilot study
│   ├── aligment.fasta		      -> multiple sequence aligment of all HLA, unused for now
│   ├── fasta			              -> folder containing the sequence of the allele
│   ├── hla.pssm
│   ├── Human_MHC_data.fasta.pdb
│   ├── Human_MHC_data.fasta.phr
│   ├── Human_MHC_data.fasta.pin
│   ├── ...
│   ├── human_mhc.fasta		      -> file containing sequences of all human alleles
│   └── pssm_raw		            -> folder required by DeepRank/PSSMgen to generate the raw pssm
│
features_input_folder		        -> folder structure required by the DeepRank CNN data generator
├── pdb				                  -> symlinks to all pdb files in DB3
├── pssm			                  -> symlinks to all pssm files in DB3
│
features_output_folder		      -> output of the feature generation
```