from torch import nn
import torch

class CnnClassificationBaseline(nn.Module):
    def __init__(self, input_shape):
        """This class is the architecture for DeepRank's CNN framework.
        Made of one sequential convolutional layers array followed by
        one fully connected layers array.
        Used in CNN/I/classification/struct/cnn_baseline.py as an argument
        for the architecture class.

        Args:
            input_shape (int): Number of features (64 as of now)
        """
        super(CnnClassificationBaseline, self).__init__()

        #self.flatten_shape = 2304 # output shape of flatten conv_layers

        self.conv_layers = nn.Sequential(
            nn.BatchNorm3d(input_shape[0]),

            nn.Conv3d(input_shape[0], input_shape[0], kernel_size=2),
            nn.ReLU(),
            nn.MaxPool3d((2,2,2)),

            nn.Conv3d(input_shape[0],input_shape[0]*2, kernel_size=2),
            nn.ReLU(),
            nn.MaxPool3d((2,2,2)),

            nn.Conv3d(input_shape[0]*2, input_shape[0]*3, kernel_size=2),
            nn.ReLU(),
            nn.MaxPool3d((2,2,2))
        )

        self.flatten_shape = len(self.conv_layers(torch.rand(input_shape).unsqueeze(0)).flatten())

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.BatchNorm1d(self.flatten_shape),

            nn.Linear(self.flatten_shape, 1000),
            nn.ReLU(),
            nn.Dropout(),

            nn.Linear(1000, 1000),
            nn.ReLU(),
            nn.Dropout(),

            nn.Linear(1000, 2)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

class CnnClassification4Conv(nn.Module):
    def __init__(self, input_shape):
        """This class is the architecture for DeepRank's CNN framework.
        Made of one sequential convolutional layers array followed by
        one fully connected layers array.
        Used in CNN/I/classification/struct/cnn_baseline.py as an argument
        for the architecture class.

        Args:
            input_shape (int): Number of features (64 as of now)
        """
        super(CnnClassification4Conv, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.BatchNorm3d(input_shape[0]),

            nn.Conv3d(input_shape[0], int(input_shape[0]//2), kernel_size=1),
            nn.ReLU(),

            nn.Conv3d(int(input_shape[0]//2), int(input_shape[0]//2), kernel_size=1),
            nn.MaxPool3d((2,2,2)),
            nn.ReLU(),

            nn.Conv3d(int(input_shape[0]//2),int(input_shape[0]//2), kernel_size=2),
            nn.MaxPool3d((2,2,2)),
            nn.ReLU(),

            nn.Conv3d(int(input_shape[0]//2), int(input_shape[0]//2), kernel_size=2),
            nn.MaxPool3d((2,2,2))
            nn.ReLU(),
        )

        self.flatten_shape = len(self.conv_layers(torch.rand(input_shape).unsqueeze(0)).flatten())

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.BatchNorm1d(self.flatten_shape),

            nn.Linear(self.flatten_shape, 1000),
            nn.ReLU(),
            nn.Dropout(),

            nn.Linear(1000, 1000),
            nn.ReLU(),
            nn.Dropout(),

            nn.Linear(1000, 2)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        print(x.shape)
        x = self.fc_layers(x)
        return x


