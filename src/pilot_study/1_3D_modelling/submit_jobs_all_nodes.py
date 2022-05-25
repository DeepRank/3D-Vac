#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 30 19:39:49 2021

@author: Dario Marzella
"""
import os
from math import ceil

## Usage: python submit_jobs_all_nodes.py
## Args: n_cores, batch dimension, batch id

# Get unmodelled cases
os.system('qsub -q all.q@surprise.umcn.nl -N get_unmodelled_cases octarine_run_get_unmodelled_cases.sh')

# (nodename, n_cores, cases_per_hour)
nodes = [("all.q@narrativum.umcn.nl",30, 9), ("all.q@plinc.umcn.nl", 30, 5)]#, ("all.q@noggo.umcn.nl", 39, 8), #Noggo has here 1 core less than real
         #("all.q@surprise.umcn.nl", 32, 7)],
         #("dev.q", 48, 5)]


#Cases_per_hour = 9
Hours = 24
max_running_time_per_job = 3
#Cases_per_timespan = Cases_per_hour * Hours
Start = 0
End = Start
jobnames = []
for i in range(len(nodes)):
    # Define Nodename
    if '@' in nodes[i][0]:
        nodename = nodes[i][0].split('@')[1].split('.')[0]
    else:
        nodename='fm'
    
    #Define Number of cases to run per each job
    job_cases_threshold = nodes[i][1]*(max_running_time_per_job*nodes[i][2])
    tot_cases = nodes[i][1]*(Hours*nodes[i][2])
    Start = End
    End = Start+(tot_cases) # Start + (n_cores*(hours * cases/hour))
    print('START: %i, END: %i' %(Start, End))
    n_sub_batches = ceil(tot_cases/job_cases_threshold)
    cases_per_sub_batch = tot_cases/n_sub_batches
    subjob_end = Start
    for x in range(n_sub_batches):
        jobname = 'job_%s_%i' %(nodename,x)
        subjob_start = subjob_end
        subjob_end = Start + (cases_per_sub_batch*(x+1))
        if subjob_end > End:
            subjob_end = End
        if x == 0:
            command = 'qsub -q %s -N %s -hold_jid get_unmodelled_cases -pe smp %i octarine_run_wrapper.sh %i %i %i' %(
            nodes[i][0], jobname, nodes[i][1], nodes[i][1], subjob_start, subjob_end)
        else:
            command = 'qsub -q %s -N %s -hold_jid get_unmodelled_cases,%s -pe smp %i octarine_run_wrapper.sh %i %i %i' %(
            nodes[i][0], jobname, 'job_%s_%i' %(nodename,x-1), nodes[i][1], nodes[i][1], subjob_start, subjob_end)
        

    #jobname = 'job_%s' %nodename
    #command = 'qsub -q %s -N %s -hold_jid get_unmodelled_cases -pe smp %i octarine_run_wrapper.sh %i %i %i' %(
    #        nodes[i][0], jobname, nodes[i][1], nodes[i][1], S, End)
        print(command)
        os.system(command)
    
        jobnames.append(jobname)
    
#Clean the outputs
os.system('qsub -q all.q@surprise.umcn.nl -hold_jid %s -pe smp 32 octarine_run_clean_outputs.sh' %(',').join(jobnames))

'''
os.system('qsub -q %s octarine_run_wrapper.sh 40 160 1')
os.system('qsub -q %s octarine_run_wrapper.sh 64 256 2')
os.system('qsub -q %s octarine_run_wrapper.sh 32 128 3')
'''
