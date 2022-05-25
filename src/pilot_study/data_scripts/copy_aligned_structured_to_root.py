import argparse
from shutil import copy
import glob
import random

arg_parser = argparse.ArgumentParser(description="Copy a fraction of the files into the root directory");
arg_parser.add_argument("--file-num", "-n",
    help="number of files to copy",
    default=20,
    type=int
)
a = arg_parser.parse_args();

pdb_files = glob.glob("/projects/0/einf2380/data/pMHCI/pssm_mapped/BA/*/*/pdb/*.pdb")
picked_files = random.sample(pdb_files,a.file_num)
for f in picked_files:
    copy(f, "/home/lepikhovd/aligned/"+f.split("/")[-1])