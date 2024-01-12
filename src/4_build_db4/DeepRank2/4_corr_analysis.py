import glob
import os
import h5py
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import logging
import sys


# for now we do the correlation analysis for graphs only
# we could do the same for the features mapped to the grid

####### please modify here #######
run_day = '230530'
# project_folder = '/projects/0/einf2380/'
project_folder = '/home/ccrocion/snellius_data_sample/'
data = 'pMHCI'
resolution = 'residue' # either 'residue' or 'atomic'
target_dataset = 'binary'
##################################

output_folder = f'{project_folder}data/{data}/features_output_folder/deeprank2/{resolution}/{run_day}'
hdf5_files = glob.glob(os.path.join(output_folder, '*.hdf5'))
df_path = f"{project_folder}data/{data}/features_output_folder/deeprank2/{resolution}/{run_day}/residue_pandas.feather"
images_path = os.path.join(output_folder, 'images')

# Loggers
_log = logging.getLogger('')
_log.setLevel(logging.INFO)

fh = logging.FileHandler(os.path.join(output_folder, '4_corr_analysis.log'))
sh = logging.StreamHandler(sys.stdout)
fh.setLevel(logging.INFO)
sh.setLevel(logging.INFO)
formatter_fh = logging.Formatter('[%(asctime)s] - %(name)s - %(message)s',
                               datefmt='%a, %d %b %Y %H:%M:%S')
fh.setFormatter(formatter_fh)

_log.addHandler(fh)
_log.addHandler(sh)

with h5py.File(hdf5_files[0], 'r') as hdf5:
    mol = list(hdf5.keys())[0]
    node_feat = list(hdf5[mol]["node_features"].keys())
    edge_feat = list(hdf5[mol]["edge_features"].keys())

df = pd.read_feather(df_path)
pssm_cols = [pssm_feat for pssm_feat in list(df.columns) if 'pssm' in pssm_feat]
cols_to_drop = ['id'] + pssm_cols
df.drop(columns = cols_to_drop, inplace=True)
_log.info(f"df read. Features: \n{df.columns}")

node_df = pd.DataFrame()
edge_df = pd.DataFrame()
for col in df.columns:
    for feat in node_feat:
        if feat in col:
            node_df[col] = df[col].copy()
    for feat in edge_feat:
        if feat in col:
            edge_df[col] = df[col].copy()

node_df = node_df.explode(list(node_df.columns), ignore_index=True)
edge_df = edge_df.explode(list(edge_df.columns), ignore_index=True)

_log.info("node and edge dfs created.")

plt.figure(figsize=(50, 30))
heatmap = sns.heatmap(node_df.astype('float64').corr(),
            annot=True,
            vmin=-1,
            vmax=1,
            cmap='BrBG')
heatmap.set_title('Correlation Heatmap for Node Features', fontdict={'fontsize':18}, pad=16)
plt.savefig(os.path.join(images_path, 'heatmap_node_feat.png'), dpi=300, bbox_inches='tight')

_log.info("node heatmap saved.")

plt.figure(figsize=(10, 4))
heatmap = sns.heatmap(edge_df.astype('float64').corr(),
            annot=True,
            vmin=-1,
            vmax=1,
            cmap='BrBG')
heatmap.set_title('Correlation Heatmap for Edge Features', fontdict={'fontsize':18}, pad=16)
plt.savefig(os.path.join(images_path, 'heatmap_edge_feat.png'), dpi=300, bbox_inches='tight')

_log.info("edge heatmap saved.")
