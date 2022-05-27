#!/bin/bash
#$ -l h_rt=00:10:00
#$ -cwd
#$ -V
#$ -p -5

python test_sleep.py
