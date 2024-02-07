More details about how to generate DB3 can be found [here](https://github.com/DeepRank/3D-Vac?tab=readme-ov-file#3-db3). 

## 3.3 (Optional): Build PSSM for M chain (MHC protein) and pseudo-PSSM encoding for the P chain (peptide)
### 3.3.1: Build the blast database

A BLAST database is required to build PSSMs. `2_build_blastdb.sh` takes care of building it. Mind that, depending on the selected original fasta database (e.g. all unirpot, all HLAs from IMGT or all MHCs from IMGT, etc.) will greatly impact the diversity of the BLAST database and the final PSSMs.

* Make sure `blast` is installed.
  The conda installation does not work on Snellius (on other systems should be ok). In that case, download and extract the package from https://ftp.ncbi.nlm.nih.gov/blast/executables/LATEST/, add the `bin` folder to your `PATH`. Make sure also `psiblast` is in your PATH and callable by terminal.
* Copy the human MHC sequence fasta file from `/<PANDORA_installation_folder>/Databases/default/mhcseqs/hla_prot.fasta` into `data/pssm/blast_dbs/`.
  

Run: 
```bash
sbatch 2_build_blastdb.sh
```

### 3.3.2: Calculate raw PSSM for M chain

```bash
sbatch 3_create_raw_pssm_I.sh
```

* Run `python create_raw_pssm.py --help` for more information.

### 3.3.3: Generate fake PSSM for the peptide

```bash
sbatch 5_peptide2onehot.sh
```

* Run `python peptide2onehot.py --help` for more information.

### 3.3.4: Map generated raw PSSM to the PDB

```bash
sbatch 6_map_pssm2pdb.sh
```
* Mapping raw PSSM to the pdb alleviate problems such as gaps in sequences.
* Only mapped PSSM for the M chain are used to generate the PSSM db3 feature.
