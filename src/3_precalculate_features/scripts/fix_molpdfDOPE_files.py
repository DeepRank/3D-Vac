#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 16:10:46 2021

@author: Dario Marzella
"""

import os

mod_dir = '/mnt/csb/Dario/3d_epipred/models'
score_file = 'molpdf_DOPE.tsv'

for folder in os.listdir(mod_dir):
    if score_file in os.listdir(mod_dir + '/' + folder):
        print(folder)
        with open(mod_dir + '/' + folder + '/' + score_file, 'r') as infile:
            with open(mod_dir + '/' + folder + '/' + '_molpdf_DOPE.tsv', 'w') as outfile:
                for line in infile:
                    outfile.write(line.replace('matched_', ''))
                    #print(line.replace('matched_', ''))
                    
                    
                    
        os.popen('mv %s/%s/_molpdf_DOPE.tsv %s/%s/molpdf_DOPE.tsv' %(mod_dir, folder, mod_dir, folder)).read()