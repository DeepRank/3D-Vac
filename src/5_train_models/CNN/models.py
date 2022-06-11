from torch import nn 
class MlpRegBaseline(nn.Module):
    def __init__(self, input = 20*9, neurons_per_layer=1000, outputs=1 ):
        super(MlpRegBaseline,self).__init__()
        self.architecture = nn.Sequential(
            nn.Flatten(),

            nn.Linear(input, neurons_per_layer),
            # nn.BatchNorm1d(neurons_per_layer),
            nn.Dropout(),
            nn.ReLU(),

            nn.Linear(neurons_per_layer, neurons_per_layer),
            # nn.BatchNorm1d(int(neurons_per_layer/2)),
            nn.Dropout(),
            nn.ReLU(),

            nn.Linear(neurons_per_layer, neurons_per_layer),
            # nn.BatchNorm1d(int(neurons_per_layer/4)),
            nn.Dropout(),
            nn.ReLU(),

            nn.Linear(neurons_per_layer, outputs),
        )
    def forward(self,x):
        return self.architecture(x)
