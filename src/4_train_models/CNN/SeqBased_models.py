from torch import nn
import torch

# FUNCTIONS AND USEFUL STUFF
#----------------------------

# define the train function
def train_f(dataloader, model, loss_fn, optimizer):
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
        # forward propagation
        pred = model(X)
        loss = loss_fn(pred,y)

        # backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# define the function used for evaluation
def evaluate(dataloader, model, loss_fn, device):
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

    Returns:
        _type_: _description_
    """
    tpr = [] # sensitivity
    tnr = [] # specificity
    accuracies = []
    losses = []
    model.eval()
    with torch.no_grad():
        for X,y in dataloader:
            logits = model(X)
            pred = logits.max(1)[1]

            confusion = pred/y # absolute values for metrics
            tot = y.shape[0] # total number of prediction
            pos = float((y == 1.).sum()) # total number of positives (truth 1)
            neg = float((y == 0.).sum()) # total number of negatives (truth 0)

            loss = loss_fn(logits, y)

            tpr.append( float((confusion == 1.).sum()/pos) ) # true positive rate = prediction 1/truth 1
            tnr.append( float(torch.isnan(confusion).sum()/neg) ) # true negative rate = predicted 0/truth 0
            accuracies.append(float((pred==y).sum()/tot))
            losses.append(loss.float())

    tpr = torch.tensor(tpr, device=device)
    tnr = torch.tensor(tnr, device=device)
    accuracies = torch.tensor(accuracies, device=device)
    losses = torch.tensor(losses, device=device)

    return accuracies.mean(), tpr.mean(), tnr.mean(), losses.mean()

class MlpRegBaseline(nn.Module):
    def __init__(self, input = 20*9, neurons_per_layer=1000, outputs=1 ):
        """This class is the architecture of the multi-layer perceptron used both
        for classification and regression.
        Used by CNN/I/regression/seq/mlp_baseline.py and CNN/I/classification/seq/mlp_baseline.py.

        Args:
            input (int, optional): Dimensions of the flattened matrix representing the peptide. Defaults to 20*9.
            neurons_per_layer (int, optional): Neurons per layer. Defaults to 1000.
            outputs (int, optional): Output of the layer. Defaults to 1.
        """
        super(MlpRegBaseline,self).__init__()
        self.architecture = nn.Sequential(
            # input layer
            nn.Flatten(),

            # hidden layer
            nn.Linear(input, neurons_per_layer),
            nn.ReLU(),
            nn.Dropout(.25),

            nn.Linear(neurons_per_layer, neurons_per_layer),
            nn.ReLU(),
            nn.Dropout(.25),

            nn.Linear(neurons_per_layer, neurons_per_layer),
            nn.ReLU(),
            # nn.Dropout(),

            # output layer
            nn.Linear(neurons_per_layer, outputs),
        )
    def forward(self,x):
        return self.architecture(x)