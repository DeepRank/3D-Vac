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

## Remote development locally using SSH

- The [Visual Studio Code Remote - SSH extension](https://code.visualstudio.com/docs/remote/ssh) allows you to open a remote folder on any remote machine. Once connected to a server, you can interact with files and folders anywhere on the remote filesystem. The local system requirement is to have a supported OpenSSH compatible SSH client, while the remote system requirement is to have a running SSH server on RHEL 7+ (in Snellius case).

## Creating and running jobs

### The batch system

- It's a tool to distribute computational tasks over the available nodes.
- You prepare a set of commands (a 'job') and store them in the job script file, that the computer will execute later.
- When and where a job is executed is determined by the job scheduler.

### The job scheduler

- When you submit a job, it enters a job queue. A scheduler reads the job requirements from all submitted job scripts and determines when, and on which nodes these jobs should run.

### Writing a job script

- A job script generally consists of the following parts.
  1. Specification of the job requirements for the batch system (number of nodes, expected runtime, etc).
  2. Loading of modules needed to run your application. To see which modules are available, use `modul avail` or to find all possible modules and extensions `module spider`. To load a module, use `
module load module_name`. [Here](https://servicedesk.surf.nl/wiki/display/WIKI/Software+on+Snellius+and+Lisa) you can find modules' names.
  3. Preparing input data (e.g. copying input data to from your home to scratch, preprocessing). The scratch disk provides temporary storage that is much faster than the home file system. In case you have 1 or 2 tiny input files, it may be acceptable to read these directly from the home file system. If you have many files and/or very large files, however, this will put too high a load on the home file system - and slow down your job considerably. 
The solution is to copy your input data to the local scratch disk of each node before starting your application. See [here](https://servicedesk.surf.nl/wiki/display/WIKI/Copy+input+data+to+scratch) for more details. 
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

`#SBATCH --ntasks=1` sets the number of tasks per node. For example, MPI programs will use this number to determine how many processes to run per node.

`#SBATCH --cpus-per-task=32` the default will allocate 1 cpu per task. If you would like to advise the Slurm controller that job steps will require a different number of processors per task, use this option.

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

#Execute a Python program located in $HOME, that takes an input directory and an output directory as arguments.
srun python $HOME/my_program.py "$TMPDIR"/input_dir "$TMPDIR"/output_dir
#MPI case
mpiexec my_program "$TMPDIR"/input_dir "$TMPDIR"/output_dir

#Copy output directory from scratch to home
cp -r "$TMPDIR"/output_dir $HOME
```

### Submitting a job

- To submit the job described in my_job.sh, use `sbatch my_job.sh`.
- Upon submission, the system will report the job ID that has been assigned to the job.
- If you want to cancel a job in the queue, use `scancel [jobid]`.
- In the batch system, the job output is written to a text file: `slurm-[jobid].out`. Make sure to check this file after your job has finished to see if it ran correctly.
- After you have submitted a job, it ends up in the job queue. You can inspect the queue with `squeue -u [username]` or `squeue -j [jobid]`.
- For some applications, you want to submit the same job script repeatedly. In these situations, you can submit an [array job](https://servicedesk.surf.nl/wiki/display/WIKI/Array+jobs).
- you can use `srun` to start an interactive job or an MPI program (more about it [here](https://servicedesk.surf.nl/wiki/display/WIKI/Interactive+jobs)).
