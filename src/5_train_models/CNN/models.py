from torch import nn 
class CnnClassificationBaseline(nn.Module):
    def __init__(self, input_shape):
        super(CnnClassificationBaseline, self).__init__()

        self.flatten_shape = 48 # output shape of flatten conv_layers

        self.conv_layers = nn.Sequential(
            nn.BatchNorm3d(input_shape[0]),

            nn.Conv3d(input_shape[0], input_shape[0], kernel_size=2),
            nn.ReLU(),
            nn.MaxPool3d((2,2,2)),

            nn.Conv3d(input_shape[0],32, kernel_size=2),
            nn.ReLU(),
            nn.MaxPool3d((2,2,2)),

            nn.Conv3d(32, 4, kernel_size=2),
            nn.ReLU(),
            nn.MaxPool3d((2,2,2))
        )

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.BatchNorm1d(self.flatten_shape),

            nn.Linear(self.flatten_shape, 100),
            nn.ReLU(),
            # nn.Dropout(),

            nn.Linear(100, 32),
            nn.ReLU(),
            # nn.Dropout(),

            nn.Linear(32, 2)
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

class MlpRegBaseline(nn.Module):
    def __init__(self, input = 20*9, neurons_per_layer=1000, outputs=1 ):
        super(MlpRegBaseline,self).__init__()
        self.architecture = nn.Sequential(
            # input layer
            nn.Flatten(),
            nn.BatchNorm1d(input),
            
            # hidden layer
            nn.Linear(input, neurons_per_layer),
            nn.ReLU(),

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
