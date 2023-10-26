DB3 is the collection of data needed to generate hdf5 files, namely selected 3D models and their PSSMs.
PSSM features have not been used in the final version of the project, so only steps 1 and 4 are necessary.

Step by step explanation follows here.

#### 1: Selecting which PANDORA-generated 3D-models to use
```
sbatch 1_copy_3Dmodels_from_db2.sh
```
* PANDORA generates 20 pdb structures per cases. They are ranked based on the global energy of the complex.
* The first 5 pdb in this ranking contain the most plausible structure.
* For now, only the first structure is being used. The script `src/2_build_db3/copy_3Dmodels_from_db2.py` is written in a way that it will be possible to select more than 1 structure in the future.
* Run `python src/2_build_db3/copy_3Dmodels_from_db2.py --help` for more information on how the script works.

#### 2: Aligning structures
```
sbatch 4_align_pdb.sh
```
* Aligns every structures to one template.
* Add `--help` to see additional information.

#### 3 (Optional): Build PSSM for M chain (MHC protein) and pseudo-PSSM encoding for the P chain (peptide)
##### 3.1: Build the blast database

A BLAST database is required to build PSSMs. `src/2_build_db3/2_build_blastdb.sh` takes care of building it. Mind that, depending on the selected original fasta database (e.g. all unirpot, all HLAs from IMGT or all MHCs from IMGT, etc.) will greatly impact the diversity of the BLAST database and the final PSSMs.


* Make sure `blast` is installed.
  The conda installation does not work on Snellius (on other systems should be ok). In that case, download and extract the package from https://ftp.ncbi.nlm.nih.gov/blast/executables/LATEST/, add the `bin` folder to your `PATH`. Make sure also `psiblast` is in your PATH and callable by terminal.
* Copy the human MHC sequence fasta file from `/<PANDORA_installation_folder>/Databases/default/mhcseqs/hla_prot.fasta` into `data/pssm/blast_dbs/`.
  

Run: 
```
sbatch 2_build_blastdb.sh
```

##### 3.2: Calculate raw PSSM for M chain:
```
sbatch 3_create_raw_pssm_I.sh
```
* Run `python src/2_build_db3/create_raw_pssm.py --help` for more information.

##### 3.3: Generate fake PSSM for the peptide
```
sbatch 5_peptide2onehot.sh
```
* Run `python src/2_build_db3/peptide2onehot.py --help` for more information.

##### 3.4: Map generated raw PSSM to the PDB:
```
sbatch 6_map_pssm2pdb.sh
```
* Mapping raw PSSM to the pdb alleviate problems such as gaps in sequences.
* Only mapped PSSM for the M chain are used to generate the PSSM db3 feature.



