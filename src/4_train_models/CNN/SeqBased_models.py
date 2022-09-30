from torch import nn

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
            nn.Dropout(),

            nn.Linear(neurons_per_layer, neurons_per_layer),
            nn.ReLU(),
            nn.Dropout(),

            nn.Linear(neurons_per_layer, neurons_per_layer),
            nn.ReLU(),
            nn.Dropout(),

            # output layer
            nn.Linear(neurons_per_layer, outputs),
        )
    def forward(self,x):
        # x = x.permute((0,2,1))
        return self.architecture(x)