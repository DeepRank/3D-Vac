More details about how to generate DB4-graphs can be found [here](https://github.com/DeepRank/3D-Vac?tab=readme-ov-file#42-graphs). 

- `2_feat_pandas_hist.sh`: for reading in the HDF5 files containing graphs (and grids) and putting them into a Pandas Dataframe, which is saved into a FEATHER file. Histograms showing the distributions for all the features are also generated, and are saved into PNG images. 
- `3_explore_feat.sh`: for plotting histograms, means and standard deviation of the features, reading the FEATHER file generated in the above step, and saving the histograms into PNG images. 
- `4_corr_analysis.sh`: for plotting Pearson correlation heatmaps for the processed data (one heatmap for nodes features and one for edge features), reading them from the FEATHER file, and saving the heatmaps into PNG images.
