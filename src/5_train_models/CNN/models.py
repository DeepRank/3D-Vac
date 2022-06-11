from torch import nn 
class MlpRegBaseline(nn.Module):
    def __init__(self, input = 20*9, neurons_per_layer=1000, outputs=1 ):
        super(MlpRegBaseline,self).__init__()
        self.architecture = nn.Sequential(
            # input layer
            nn.Flatten(),
            
            # hidden layer
            nn.Linear(input, neurons_per_layer),
            nn.ReLU(),
            nn.Dropout(), # half of weights are droped each forward()

            nn.Linear(neurons_per_layer, neurons_per_layer),
            nn.ReLU(),

            nn.Linear(neurons_per_layer, neurons_per_layer),
            nn.ReLU(),

            # output layer
            nn.Linear(neurons_per_layer, outputs),
        )
    def forward(self,x):
        x = x.permute((0,2,1))
        return self.architecture(x)
