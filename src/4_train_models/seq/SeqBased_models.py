from torch import nn
import torch
from mhcflurry.regression_target import to_ic50
import numpy as np
from sklearn import metrics
import pickle

# FUNCTIONS AND USEFUL STUFF
#----------------------------

# define the train function
def train_f(dataloader, model, loss_fn, optimizer, device):
    """Function used to train the model over the whole training dataset.

    Args:
        dataloader (DataLoader): Instance of the DataLoader object iterating
        over the train dataset.
        model (obj): Model to train.
        loss_fn (obj): Function used to measure the difference between the truth
        and prediction.
        optimizer (obj): Function used to perform backpropagation and update of
        weights.
    """
    model.train()
    # print(f"training epoch {e} on {rank}")
    for X,y in dataloader:
        X.to(device)
        y.to(device)
        # forward propagation
        pred = model(X)
        loss = loss_fn(pred, y)

        # backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# define the function used for evaluation
def evaluate(dataloader, model, loss_fn, device, task="classification"):
    """### CALCULATE SENSITIVITY, SPECIFICITY AND OTHER METRICS ###
    Calculate the confusion tensor (AKA matching matrix) by dividing predictions
    with true values.
    To assess performances, first calculate absolute values and adjust it to 
    the total number of expected postives, negatives:

    Args:
        dataloader (DataLoader): Dataloader containing the iterator for the 
        dataset we want to evaluate.
        model (obj): Model being evaluated.
        loss_fn (obj): Function used to measure the difference between the truth
        and prediction.
        device (string): "cpu" or torch.device("cuda")
        task (string): Evaluate either on classification or regression

    Returns:
        _type_: _description_
    """
    logits = torch.empty(0)
    #logits = []
    pred = []
    targets = []
    tpr = [] # sensitivity
    tnr = [] # specificity
    accuracies = []
    losses = []
    model.eval()
    with torch.no_grad():
        for X,y in dataloader:
            X.to(device)
            y.to(device)
            mb_logits = model(X)
            #logits.extend(mb_logits)
            logits = torch.cat((logits, mb_logits), dim=0)
            targets.extend([float(x) for x in y.tolist()])
            loss = loss_fn(model(X), y)
            losses.append(loss.float())


            if task == "regression":
                ic50_logits = to_ic50(np.array(mb_logits))
                ic50_y = to_ic50(np.array(y))
                pred = torch.tensor([int(val < 500) for val in ic50_logits], dtype=torch.float32)
                y = torch.tensor([int(val < 500) for val in ic50_y], dtype=torch.int32).to(device)
            else:
                pred = mb_logits.max(1)[1]

            confusion = pred/y # absolute values for metrics
            tot = y.shape[0] # total number of prediction
            pos = float((y == 1.).sum()) # total number of positives (truth 1)
            neg = float((y == 0.).sum()) # total number of negatives (truth 0)

            tpr.append( float((confusion == 1.).sum()/pos) ) # true positive rate = prediction 1/truth 1
            tnr.append( float(torch.isnan(confusion).sum()/neg) ) # true negative rate = predicted 0/truth 0
            accuracies.append(float((pred==y).sum()/tot))
        
    #auc = metrics.roc_auc_score(logits,targets)

    tpr = torch.tensor(tpr, device=device)
    tnr = torch.tensor(tnr, device=device)
    accuracies = torch.tensor(accuracies, device=device)
    losses = torch.tensor(losses, device=device)

    return accuracies.mean(), tpr.mean(), tnr.mean(), losses.mean()#, auc

class MlpRegBaseline(nn.Module):
    def __init__(self, input_shape = (21,82), neurons_per_layer=1000, n_output_nodes=1 ):
        """This class is the architecture of the multi-layer perceptron used both
        for classification and regression.
        Used by CNN/I/regression/seq/mlp_baseline.py and CNN/I/classification/seq/mlp_baseline.py.

        Args:
            input (int, optional): Dimensions of the flattened matrix representing the peptide. Defaults to 20*9.
            neurons_per_layer (int, optional): Neurons per layer. Defaults to 1000.
            n_output_nodes (int, optional): Number of output neurons. Defaults to 1.
        """

        super(MlpRegBaseline,self).__init__()

        # self.conv_layer = nn.Sequential(
        #     nn.BatchNorm1d(input_shape[0]),

        #     nn.Conv1d(
        #         in_channels = input_shape[0], #82
        #         out_channels = 200, 
        #         kernel_size = 1,
        #     ),
        #     nn.ReLU(),
        #     nn.MaxPool1d(
        #         kernel_size = 2,
        #         padding= 2,
        #     )

        # )

        # # input size for the linear layer:
        # self.flatten_shape = self.conv_layer(torch.randn(input_shape).unsqueeze(0)).flatten().shape[0] 
        self.n_output_nodes = n_output_nodes
        self.flatten_shape = input_shape[0] * input_shape[1]

        self.linear = nn.Sequential(
            # input layer
            nn.Flatten(),
            nn.BatchNorm1d(self.flatten_shape),

            # hidden layers
            nn.Linear(self.flatten_shape, neurons_per_layer),
            nn.ReLU(),
            nn.Dropout(),

            nn.Linear(neurons_per_layer, neurons_per_layer),
            nn.ReLU(),
            nn.Dropout(),

            nn.Linear(neurons_per_layer, neurons_per_layer),
            nn.ReLU(),
            nn.Dropout(),

            # output layer
            nn.Linear(neurons_per_layer, n_output_nodes),
            # if using BCE:
        )
    def forward(self,x):
        #torch.Size([12509, 21, 82])
        # conv_out = self.conv_layer(x)
        x = self.linear(x)
        if self.n_output_nodes == 1:
            s = nn.Sigmoid()
            x = s(x)
            x = x.squeeze(1)
        return x