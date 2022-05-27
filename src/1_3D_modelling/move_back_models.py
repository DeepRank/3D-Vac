import glob
import shutil


files = glob.glob("/projects/0/einf2380/data/pMHCI/models/BA/*/*");
# print(files)
for f in files:
    shutil.move(f, "/projects/0/einf2380/data/pMHCI/models/temp")

files = glob.glob("/projects/0/einf2380/data/pMHCI/models/BA/*/*");
print("number of files left:", len(files))