# Snellius

## Basic info

- The Dutch National supercomputer hosted at SURF.
- A cluster of heterogeneous nodes built by Lenovo, containing predominantly AMD technology.
- Peak performance of 14 petaflop/s.
- To login via the terminal from a white-listed IP: `ssh user@snellius.surf.nl`. Secure Shell is an encrypted network protocol for transferring data across insecure networks, and is widely used to connect to clusters. An SSH Key uses asymmetric cryptography to authenticate your identity. SSH keys are
composed of two parts: the public key which, as the name suggest can be distributed publicly, and the private key which must remain private.
- If you encounter the need to display graphical interfaces from the remote system on your local screen, X11 port forwarding/tunneling facilitates this seamlessly and securely. If you use `ssh -X remotemachine` the remote machine is treated as an untrusted client. So your local client sends a command to the remote machine and receives the graphical output. If your command violates some security settings you'll receive an error instead. But if you use `ssh -Y remotemachine` the remote machine is treated as a trusted client. You can check the options ForwardX11 and ForwardX11Trusted in your `/etc/ssh/ssh_config` file (after login to Snellius).
- From abroad or not white-listed-IP, there is a separate login server: doornode.surfsara.nl (thus using `ssh user@doornode.surfsara.nl`). Use your usual login and password, and select 'Snellius'. Please note that you cannot copy files or use X11 when using the door node. See also [here](https://servicedesk.surfsara.nl/wiki/pages/viewpage.action?pageId=30660265) for more info.
- You can find the blocked ip-address when logging in to Snellius using `ssh -v [login]@snellius.surf.nl`.
- To disconnect type either `logout` or `exit` in the terminal window, and then enter. 
- To go to shared 3dVac folder: `cd /projects/0/einf2380/`.
- The interactive nodes only accept (GSI-)ssh connections from known, white-listed IP, ranges.
- type `accinfo` on terminal to get general infos.

## Local development using SSH

- For MAC users: after having installed macFUSE (it allows you to extend macOS's native file handling capabilities via third-party file systems) and SSHFS (it's used to mount a remote directory by using SSH alone), see https://osxfuse.github.io/, you can use the following command to mount REMOTE_PATH (folder on cluster) to LOCAL_PATH (your local machine):
```
sshfs -p 22 user@server:REMOTE_PATH LOCAL_PATH -o auto_cache,reconnect,defer_permissions,noappledouble,negative_vncache,volname=VOL_NAME
```
- Now you can see the contents of REMOTE_PATH on your LOCAL_PATH, and using them to run scripts *locally*.
- To unmount use `diskutil umount force LOCAL_PATH`.
- Note that if you try to read files to your local machine from Snellius, each file will be downloaded from the remote every time. If you try to populate in bulk a list with all certain files names, for example, if they are thousands you can get stucked. Run the script from Snellius instead. 

## Remote development locally using SSH

- The [Visual Studio Code Remote - SSH extension](https://code.visualstudio.com/docs/remote/ssh) allows you to open a remote folder on any remote machine. Once connected to a server, you can interact with files and folders anywhere on the remote filesystem. The local system requirement is to have a supported OpenSSH compatible SSH client, while the remote system requirement is to have a running SSH server on RHEL 7+ (in Snellius case).

### Conda on Snellius

For installing conda on your Snellius home folder:
- From [here](https://www.anaconda.com/products/distribution), run `wget https://repo.anaconda.com/archive/Anaconda3-2022.05-Linux-x86_64.sh`, or the most updated version. 
- Run the .sh script just downloaded with `bash Anaconda3-2022.05-Linux-x86_64.sh`.
- Activate the base conda env with `source .bashrc`.

Now you can create conda envs. To activate a conda env run `source activate env_name`.

## Creating and running jobs

### The batch system

- It's a tool to distribute computational tasks over the available nodes.
- You prepare a set of commands (a 'job') and store them in the job script file, that the computer will execute later.
- When and where a job is executed is determined by the job scheduler.

### The job scheduler

- When you submit a job, it enters a job queue. A scheduler reads the job requirements from all submitted job scripts and determines when, and on which nodes these jobs should run.
- For very small tasks, it’s okay to run the script from the entry node, with no job scheduler. In the 3D-Vac case, an example could be to run the preprocessing part, until we have 7000 files.

### Writing a job script

- A job script generally consists of the following parts.
  1. Specification of the job requirements for the batch system (number of nodes, expected runtime, etc).
  2. Loading of modules needed to run your application. To see which modules are available, use `modul avail` or to find all possible modules and extensions `module spider`. To load a module, use `
module load module_name`. [Here](https://servicedesk.surf.nl/wiki/display/WIKI/Software+on+Snellius+and+Lisa) you can find modules' names.
  3. Preparing input data (e.g. copying input data to from your home to scratch, preprocessing). The scratch disk provides temporary storage that is much faster than the home file system, because scratch is local and on newer clusters is often on SSD, while /home is usually on hard disks and needs to be accessed over the network which every single other job running on the cluster is also trying to read or write to it. In case you have 1 or 2 tiny input files, it may be acceptable to read these directly from the home file system. If you have many files and/or very large files, however, this will put too high a load on the home file system - and slow down your job considerably. The solution is to copy your input data to the local scratch disk of each node before starting your application. However, note that if you're reading thousands of files in once, then copying them from /home to scratch at the start of the job doesn't necessarily help much, because you're still reading them once from /home, and now also writing them to scratch and then reading them back in again from there. Indeed, disks and file systems aren't set up to read or write several small files. So, if loading the data is the bottleneck, then it would probably help more to combine it all into a single large file (for example packing the data into a ZIP or TAR file, and use Python's built-in zipfile or tarfileto access it). In this case, using the scratch disk will also help, because the copy will do a big sequential read from /home, which is relatively fast, and then all the browsing through the data and opening individual files can be done on scratch. See [here](https://servicedesk.surf.nl/wiki/display/WIKI/Copy+input+data+to+scratch) for more details. 
  4. [Executing your program](https://servicedesk.surf.nl/wiki/display/WIKI/Executing+your+program).
  5. Aggregating output data (e.g. post-processing, copying data from scratch to your home). After your job finishes, if you have used the local scratch disk don't forget to [copy your results back to your home directory](https://servicedesk.surf.nl/wiki/display/WIKI/Copy+output+data+from+scratch), otherwise they will be lost.

### Script example

Common options that can be passed to the SLURM batch system, either on the `sbatch` command line, or as an `#SBATCH` attribute:

- -N [value]
- -n [value]
- -c [value]
- --tasks-per-node=[value]
- --cpus-per-task=[value]
- --mem=[value]
- --constraint=[value
- -t [HH:MM:SS] or [MM:SS] or [minutes]
- -p gpuGPU
- --requeue

All available SBATCH flags can be seen typing `sbatch --help` on the terminal. 

#### Single node, serial program example

```
#!/bin/bash
#Set job requirements
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --partition=thin
#SBATCH --time=01:00:00
 
#Execute program located in $HOME
$HOME/my_serial_program
```

In particular ... 

`#!/bin/bash` defines the interpreter for the job script with a shebang. Here we use Bash.

`#SBATCH --nodes=1` defines the number of nodes you need. 

`#SBATCH --ntasks=1` sets the number of tasks per node. Tasks are separate programs that can run on different nodes, and communicate via the network, usually via Message Passing Interface (MPI). In SLURM (and MPI, where this comes from), a program consists of one or more tasks, each of which has a number of threads. Threads within a task are the kind of threads you get with Python's `multithreading`, and in this case the processes you get with `multiprocessing` are also considered threads. All threads of the same task need to run on the same node, because they can only communicate with each other locally (using shared memory or pipes). When allocating resources, SLURM takes that restriction into account. If you have a single task you won't be able to use more than one node.

`#SBATCH --cpus-per-task=32` the default will allocate 1 cpu processor (or simply "core") per task. If you would like to advise the Slurm controller that job steps will require a different number of cpu processors per task, use this option. Note that if your code is single-threaded, then you're only ever going to use a single core, independently of how many are available.

`#SBATCH --partition=thin` if you don't specify a partition, your job will be scheduled on the normal partition by default. The available compute partitions that batch jobs can be submitted to can be listed by issuing `sinfo` command. To get a summary of the available partitions, use `sinfo -s`. The available partitions of Snellius are described [here](https://servicedesk.surfsara.nl/wiki/display/WIKI/Snellius+usage+and+accounting).

`#SBATCH --time=01:00:00` sets the duration for which the nodes remain allocated to you (known as the wall clock time).

`$HOME/my_serial_program` runs a single instance of a serial (i.e. non-parallel) program, taking a single input file as an argument. 

### Final job script

```
#!/bin/bash
#Set job requirements
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=4
#SBATCH --partition=fat
#SBATCH --time=01:00:00
#SBATCH --job-name give_a_name_to_the_job
#SBATCH -o /projects/0/einf2380/folder_name-%J.out

#Loading modules
module load Python/3.9.5-GCCcore-10.3.0
#MPI case (multi-node job)
module load mpicopy
module load OpenMPI/4.1.1-GCC-10.3.0
mpicopy $HOME/input_dir

#Copy dir with input files
cp -r $HOME/input_dir "$TMPDIR"

#Create output directory on scratch
mkdir "$TMPDIR"/output_dir

# Activate conda environment:
source /home/username/.bashrc
source activate env_name

#Execute a Python program located in $HOME, that takes an input directory and an output directory as arguments.
srun python -u $HOME/my_program.py "$TMPDIR"/input_dir "$TMPDIR"/output_dir
#MPI case
mpiexec my_program "$TMPDIR"/input_dir "$TMPDIR"/output_dir

#Copy output directory from scratch to home
cp -r "$TMPDIR"/output_dir $HOME
```

### Submitting a job

- `srun` is used to submit a job for execution in real time. You can use `srun` to start an interactive job or an MPI program (more about it [here](https://servicedesk.surf.nl/wiki/display/WIKI/Interactive+jobs)). `srun` is interactive and blocking (you get the result in your terminal and you cannot write other commands until it is finished). In this case the script is immediately executed on the remote host.
- `sbatch` is used to submit a job script for later execution. `sbatch` is batch processing and non-blocking (results are written to a file and you can submit other commands right away). In this case the job is handled by Slurm: you can disconnect, kill your terminal, etc. with no consequence. The job is no longer linked to a running process.
- You typically use `sbatch` to submit a job and `srun` in the submission script to create job steps as Slurm calls them. `srun` is used to launch the processes. If your program is a parallel MPI program, `srun` takes care of creating all the MPI processes. If not, `srun` will run your program as many times as specified by the `--ntasks` option. Unless otherwise specified, `srun` inherits by default the pertinent options of the `sbatch`.
- For some applications, you want to submit the same job script repeatedly. In these situations, you can submit an [array job](https://servicedesk.surf.nl/wiki/display/WIKI/Array+jobs).
- A feature that is available to `sbatch` and not to `srun` is job arrays.
- All the parameters `--ntasks`, `--nodes`, `--cpus-per-task`, `--ntasks-per-node` have the same meaning in both commands.
- To submit the job described in my_job.sh, use `sbatch my_job.sh`.
- Upon submission, the system will report the job ID that has been assigned to the job.
- If you want to cancel a job in the queue, use `scancel [jobid]`.
- In the batch system, the job output is written to a text file: `slurm-[jobid].out`. Make sure to check this file after your job has finished to see if it ran correctly.
- After you have submitted a job, it ends up in the job queue. You can inspect the queue with `squeue -u [username]` or `squeue -j [jobid]`.


### How to run efficient jobs

The three most common reasons for a job to run inefficiently are:

1. The file systems are not used efficiently.
2. A number of nodes are reserved for a job, but the job is only running on one node.
3. Only a single core is being used on each node.

This is a waste of resources, but more importantly, it is a waste of your CPU budget, because you are paying for all the cores reserved by your job script.

The solutions are:

(1) If you use many/large input files, copy them from the home file system to scratch before starting your program. You may consider compressing them first (using tar). Write intermediate results only to scratch. If you have many/large output files, write to scratch and then copy them to the home file system. Again, consider compressing them first.

(2 and 3) To use all nodes and cores, use [appropriate parallelization](https://servicedesk.surf.nl/wiki/display/WIKI/Methods+of+parallelization). An essential step is also to verify that your job runs on all nodes and cores the way you intended: it is easy to make a mistake and the difference in running on all cores of a node or just one may be just a single character in your job script.

To verify that your job is using all nodes and cores, you first need the ID of your job. The job ID is shown when you submit a job using sbatch, or you may find it in the queue using `squeue -u [username]`. Then, you can use the ssh command to login to any of the nodes where your job is running `ssh [node_hostname]`. Once on the node, you can use the "top" unix command to show the processes running on the node and their cpu utilisation. The output you expect depends on the type of parallelisation you have used. In case you are running a multithreaded program on a 16 core node, you may expect to see a single process that is using >> 100% CPU (ideally close to 1600%, but in practice this is usually less). For more information on how to interpret the output, see `man top`.
