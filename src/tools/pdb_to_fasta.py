from Bio import SeqIO
import sys

'''Prints as fasta the given chain of a pdb.
usage: python pdb_to_fasta.py inpdb outfasta chain
'''
def pdb_to_fasta(inpdb, outfasta, chain):
    """Prints as fasta the given chain of a pdb.

    Args:
        inpdb (str): input pdb
        outfasta (str): output fasta
        chain (str): chain ID
    """    
    model_id = outfasta.split('/')[-1].split('.')[0]
    with open(inpdb, 'r') as pdb_file:
        for record in SeqIO.parse(pdb_file, 'pdb-atom'):
            chain_id = record.id.split(':')[1]
            if chain_id == chain:
                with open(outfasta, 'w') as fasta_file:
                    fasta_file.write('>' + record.id.replace('????', model_id) + '\n')
                    fasta_file.write(str(record.seq) + '\n')


if __name__=='__main__':
    inpdb = sys.argv[1]
    outfasta = sys.argv[2]
    chain = sys.argv[3]