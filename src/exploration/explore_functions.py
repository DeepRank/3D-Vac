import pickle
from matplotlib import pyplot as plt
import torch
import pandas as pd
import plotly.graph_objects as go

def plot_hist_comparison(title, save=False):
    fig = go.Figure()
    x = ["Sensitivity", "Specificity", "AUC", "MCC", "Accuracy"]
    for model_path in df["model_path"]:
        fig.add_trace(go.Bar(
            x=x,
            y=df.loc[df["model_path"]==model_path,
                ["sensitivity_mean", "specificity_mean", "auc_mean", "mcc_mean", "accuracy_mean"]].values[0],
            name=model_path,
            error_y=dict(
                type="data",
                array=df.loc[df["model_path"]==model_path,
                    ["sensitivity_std", "specificity_std", "auc_std", "mcc_std", "accuracy_std"]].values[0],
            )
        ))
    fig.update_layout(
        barmode="group",
        title_text=title,
    )
    fig.show()
    if save:
        plt.savefig(save)

