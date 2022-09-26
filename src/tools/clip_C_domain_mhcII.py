import os
import sys
from glob import glob
from Bio.PDB import PDBParser, PDBIO
#from mpi4py import MPI
import pickle
from joblib import Parallel, delayed

def clip_C_domain(infile, MHC_class='II'):
    print('Clipping file: ', infile)
    try:
        p = PDBParser()
        S = p.get_structure('structure',infile)
    except:
        print('WARNING: file %s empty' %infile)
        return None
    if MHC_class == 'I':
        for chain in S.get_chains():
            M_end = 187
            if chain.id == 'M':
                if [x for x in chain.get_residues()][-1].id == ((' ', M_end-1, ' ')):
                        print('Chain M already cut for ' + infile)
                else:
                    for id in [(' ', x, ' ') for x in range(M_end,500)]:
                        try:
                            chain.detach_child(id)
                        except KeyError:
                            #print('Break at ', id)
                            break

    elif MHC_class == 'II':
        M_end = 86
        N_end = 95
        for chain in S.get_chains():
            if chain.id == 'M':
                if [x for x in chain.get_residues()][-1].id == ((' ', M_end-1, ' ')):
                    print('Chain M already cut for ' + infile)
                else:
                    for id in [(' ', x, ' ') for x in range(M_end,500)]:
                        try:
                            chain.detach_child(id)
                        except KeyError:
                            #print('Break at ', id)
                            break
            elif chain.id =='N':
                if [x for x in chain.get_residues()][-1].id == ((' ', N_end-1, ' ')):
                    print('Chain N already cut for ' + infile)
                else:
                    for id in [(' ', x, ' ') for x in range(N_end,500)]:
                        try:
                            chain.detach_child(id)
                        except KeyError:
                            #print('Break at ', id)
                            break

    io = PDBIO()
    io.set_structure(S)
    io.save(infile, preserve_atom_numbering = True)

def check_cut_file(infile, MHC_class='II'):
    print('Checking file: ', infile)
    p = PDBParser()
    S = p.get_structure('structure',infile)
    if MHC_class == 'I':
        M_end = 182
        if chain.id == 'M':
            if [x for x in chain.get_residues()][-1].id == ((' ', M_end-1, ' ')):
                return False
            else:
                return True

    elif MHC_class == 'II':
        M_end = 82
        N_end = 95
        for chain in S.get_chains():
            if chain.id == 'M':
                if [x for x in chain.get_residues()][-1].id == ((' ', M_end-1, ' ')):
                    return False
                else:
                    return True

            elif chain.id =='N':
                if [x for x in chain.get_residues()][-1].id == ((' ', N_end-1, ' ')):
                    return False
                else:
                    return True

def check_cut_folder(folders):
    to_cut = []
    for folder in folders:
        case = folder.split('/')[-1][:-5]
        #print(case)
        try:
            for f in os.listdir(folder):
                if f.startswith(case + '.BL00') and f.endswith('.pdb'):
                    #print(file)
                    C = check_cut_file(folder+'/'+f)
                    if C == True:
                        to_cut.append(folder)
                        break
        except FileNotFoundError:
            raise Exception('%s, %s' %(folder, case))
    return to_cut

def clip_files(folders):
    for folder in folders:
        case = folder.split('/')[-1][:-5]
        #print(case)
        for f in os.listdir(folder):
            if (f.startswith(case + '.BL00') or f.startswith(case + '.IL00')) and f.endswith('.pdb'):
            #print(file)
                clip_C_domain(folder+'/'+f)

def split_folders(folders, size):
    #comm = MPI.COMM_WORLD
    #rank = comm.Get_rank()
    #size = comm.Get_size()
    cut_folders = []
    for rank in range(size):
        step = int(len(folders)/size)
        start = int(rank*step)
        end = int((rank+1)*step)

        if rank != size-1:
            cut_folders.append(folders[start:end])
        else:
            cut_folders.append(folders[start:])

    return cut_folders

folders = glob(f"/projects/0/einf2380/data/pMHCII/3D_models/BA/*/*")
size = int(sys.argv[1])
#job = int(sys.argv[2])
#size = 32
#job = 0

cut_folders = split_folders(folders, size)
#job_folders = cut_folders[job]
#job_folders = split_folders(job_folders, size)

#to_cut = []
#to_cut = Parallel(n_jobs = size, verbose = 1)(delayed(check_cut_folder)(cut_fol) for cut_fol in cut_folders)

#with open('./to_cut_%i.pkl' %job, 'wb') as outpkl:
#    pickle.dump(to_cut, outpkl)

Parallel(n_jobs = size, verbose = 1)(delayed(clip_files)(cut_fol) for cut_fol in cut_folders)



# for folder in to_cut:
#     case = folder.split('/')[-1][:-5]
#     #print(case)
#     for f in os.listdir(folder):
#         if f.startswith(case + '.BL00') and f.endswith('.pdb'):
#            #print(file)
#            clip_C_domain(folder+'/'+f)
