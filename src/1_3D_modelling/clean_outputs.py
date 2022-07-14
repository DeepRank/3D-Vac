import os
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

output_dir = '/projects/0/einf2380/data/pMHCI/models/temp'

folders = os.listdir(output_dir)
step = int(len(folders)/size)
start = int(rank*step)
end = int((rank+1)*step)

if rank != size-1:
    cut_folders = folders[start:end]
else:
    cut_folders = folders[start:]

for case in cut_folders:
    case_path = output_dir + '/%s' %case
    case_id = "_".join(case.split('_')[0:2])#.split('Target')[1]
    os.system('rm %s/%s.DL*' %(case_path, case_id))
    os.system('rm %s/%s.B99990001.pdb' %(case_path, case_id))
    os.system('rm %s/%s.V99990001' %(case_path, case_id))
    os.system('rm %s/%s.D00000001' %(case_path, case_id))
    os.system('rm %s/results_%s.pkl' %(case_path, case))
    os.system('rm -r %s/__pycache__/' %case_path)
    os.system('rm %s/*.lrsr' %case_path)
    os.system('rm %s/*.rsr' %case_path)
    os.system('rm %s/*.sch' %case_path)
    os.system('rm %s/modeller_ini.log' %case_path)
