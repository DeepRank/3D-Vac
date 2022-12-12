import argparse
from inspect import trace
import pandas as pd
import os
import traceback
import subprocess
from Bio import SeqIO


arg_parser = argparse.ArgumentParser(
    description="""
    Finds all unique alleles in db1, copies the fasta sequences corresponding to these alleles from IPD-IMGT fasta file.
    This fasta file was used to build the blast database and is the union of these two fasta's: 
    /projects/0/einf2380/softwares/PANDORA_database/data/csv_pkl_files/Human_MHC_data.fasta and
    /projects/0/einf2380/softwares/PANDORA_database/data/csv_pkl_files/NonHuman_MHC_data.fasta\n
    With these fasta files we can build a PSSM per unqiue MHC allele, this script is called in the step: '3_create_raw_pssm'
    right before using the tool PSSMgen to actually generate the PSSMs
    """
)
arg_parser.add_argument("--output-folder", "-o",
    help="""
    Name of the ouput folder which will also be used as a working dir for PSSMgen 
    Default for MHCI /projects/0/einf2380/data/pMHCI/pssm_raw/all_mhc.
    """
)
arg_parser.add_argument("--input-file-db1", "-i",
    help="""
    Input file (db1), from which to parse all the unique MHC alleles. 
    Default for MHCI /projects/0/einf2380/data/external/processed/I/BA_pMHCI_all.csv
    """,
)
arg_parser.add_argument("--input-fasta", "-f",
    help="""
    Name of the input fasta file from which the matching MHC allele entries will be retrieved. \n
    Default: /projects/0/einf2380/data/blast_dbs/all_mhc/all_mhc_prot.fasta
    """,
    default="/projects/0/einf2380/data/blast_dbs/all_mhc/all_mhc_prot.fasta",
)
arg_parser.add_argument("--mhc-class", "-m",
    help="""
    MHC class
    """,
    choices=["I", "II"],
)

def find_alleles(db1: str):
    """ find unique alleles in db1

    Args:
        db1 (str): path of db1 containing all pMHC cases

    Returns:
        _type_: list of unique alleles found in db1
    """    
    df = pd.read_csv(db1, header=0, sep=',')
    unique_alleles = df['allele'].unique()
    allele_list = unique_alleles.tolist()
    return allele_list

def find_fasta_entries(fasta_db: str, allele_list: list):
    """find the fasta entries corresponding to allele_list

    Args:
        alleles (list): unique allele ids from db1

    Returns:
        fasta_entries (list): fasta entries
    """
    n_alleles = len(allele_list)
    fasta_entries = []
    with open(fasta_db) as handle:
        for record in SeqIO.parse(handle, "fasta"):
            rec_id = record.id
            # remove the last part of fasta id which is redundant info
            if record.id.count(":") > 1:
                rec_id = ':'.join(record.id.split(':')[:-1])
            # check if we find a matching allele and append fasta entry if true
            if rec_id in allele_list:
                fasta_entries.append(f'>{rec_id}\n{record.seq}')
                allele_list.pop(allele_list.index(rec_id))

    if n_alleles != len(fasta_entries):
        print(f"Not all alleles were found, check your input fasta for missing alleles and run again\n \
            Missing alleles:{allele_list}")

    return fasta_entries

def write_output_fasta(fasta_entries: list, output_folder: str):
    """write new fasta files, one for each in fasta_entries

    Args:
        fasta_entries (list): all fasta entries matching with unique alleles in db1
        output_folder (str): output folder which is the working dir for PSSMgen
    """    
    if not os.path.exists(output_folder):
        raise Exception("Please check your ouput folder does not exist")
    else: 
        fasta_output = os.path.join(output_folder, 'fasta')
        if not os.path.exists(fasta_output):
            try:
                subprocess.run(f'mkdir {fasta_output}', shell=True, check=True)
            except Exception as e:
                print(e)
                print(traceback.format_exc())

    for entry in fasta_entries:
        filename = entry.split('\n')[0].split('>')[1]
        try:
            with open(os.path.join(fasta_output, f'{filename}.fasta'), 'w') as fasta_out:
                fasta_out.write(entry)
        except Exception as e:
            print(e)
            print(traceback)

if __name__ == "__main__":
    args = arg_parser.parse_args()
    db1 = args.input_file_db1
    fasta_db = args.input_fasta
    output_folder = args.output_folder

    allele_list = find_alleles(db1)
    fasta_entries = find_fasta_entries(fasta_db, allele_list)
    write_output_fasta(fasta_entries, output_folder)

