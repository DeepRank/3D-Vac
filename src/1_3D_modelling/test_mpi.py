import os
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

output_dir = '/mnt/csb/Dario/3d_epipred/models'

folders = os.listdir(output_dir)
step = int(len(folders)/size)
start = int(rank*step)
end = int((rank+1)*step)

print('TYPES: ' + str(type(rank)) + ', ' + str(type(size))+ ', ' + str(type(step))+ ', ' + str(type(start))+ ', ' + str(type(end)))

if rank != size-1:
    cut_folders = folders[start:end]
    print('NORMAL STEP')
else:
    print('LAST STEP')
    cut_folders = folders[start:]


print('TOT LEN: %i' %len(folders))
print('RANK: %i' %rank)
print('SIZE: %i' %size)
print('STEP: %i' %step)
print('START: %i' %start)
print('END: %i' %end)
print('LEN CUT FOLDERS: %i' %len(cut_folders))
