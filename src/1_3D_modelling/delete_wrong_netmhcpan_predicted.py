# -*- coding: utf-8 -*-

import pickle
import os

with open('../bugged_anchors.pkl', 'rb') as inpkl:
    to_print = pickle.load(inpkl)
    
cases_dir = '/mnt/csb/Dario/3d_epipred/models'

for case in to_print:
    folder = '%s/%s' %(cases_dir, case[0])
    print(folder)
    os.popen('rm -r %s' %folder).read()