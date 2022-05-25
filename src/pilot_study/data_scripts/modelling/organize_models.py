import glob
import re
import math
import os
import shutil
import argparse

arg_parser = argparse.ArgumentParser(
    description="Script to move the generated models from the `temp` folder into the final location. \n \
    the destination is models/EL or models/BA. This script generates the subfolders numerated by their indexes from \n \
    1 to 1000 (default value, can be changed with --per-folder argument)"
)
arg_parser.add_argument("--per-folder", "-n",
    help="number of subfolders per folder",
    type=int,
    default=1000
)
arg_parser.add_argument("--mhc-type", "-m",
    help="specify the type, either 1 or 2",
    choices=[1,2],
    type=int,
    required=True,
)
arg_parser.add_argument("--toe", "-t",
    help="type of experiment",
    choices=["BA", "EL"],
    required=True
)
#initiate the variables
a = arg_parser.parse_args();
mhc = ("pMHCI", "pMHCII")[a.mhc_type == 2] # the type of MHC we're working on based on the argument provided
models_path = f"/projects/0/einf2380/data/{mhc}/models/temp" 
models_paths = glob.glob(f"{models_path}/*");

# retrieve the folder names, sort by IDs
models = {re.findall("(?<=_)(.[0-9]*)(?=_)",path.split("/")[-1])[0]:path for path in models_paths if path.split("/")[-1].startswith(a.toe)}
model_ids = [int(m) for m in models.keys()];
model_ids.sort();

models_dest_dir = f"/projects/0/einf2380/data/{mhc}/models/{a.toe}";
n_models = len(model_ids);
n_dirs = math.ceil(n_models/a.per_folder);
n = 0;
for i in range(n_dirs):
    # if n+n_per_folder is biger than the len, the index becomes the last element and not +a.per_folder
    last_index = (n+a.per_folder, n_models-1)[n+a.per_folder > n_models] 
    
    dest_subdir = f"{models_dest_dir}/{model_ids[n]}_{model_ids[last_index]}" #-1 here because the range() function 
    # print(dest_subdir)
    # print(models[str(model_ids[range(n,last_index)[0]])],models[str(model_ids[range(n,last_index+1)[-1]])]) #line to check if the first and last index files are correct
    # print(range(n,last_index)[0],range(n,last_index+1)[-1]) #line to check if the first and last index files are correct
    print(f"creating subfolder {dest_subdir} and populating it...")
    try:
        os.mkdir(dest_subdir);
    except OSError as error:
        print("skipping creating already created folder: ", dest_subdir, " populating anyway..");
    for k in range(n, last_index+1):
        src_path = models[str(model_ids[k])];
        shutil.move(src_path, dest_subdir)
    n = n + a.per_folder+1
        
        


