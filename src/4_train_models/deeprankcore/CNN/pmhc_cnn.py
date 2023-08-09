import torch
from torch import nn

class CnnClass4ConvKS3Lin128ChannExpand(nn.Module):
    
    def __init__(self, num_features, box_shape):
        """This class is the architecture for DeepRank's CNN framework.
        Made of one sequential convolutional layers array followed by
        one fully connected layers array.
        Used in CNN/I/classification/struct/cnn_baseline.py as an argument
        for the architecture class.

        Args:
            num_features (int): Number of features
            box_shape (tuple): box_width, box_height, and box_depth number of points
        """
        super(CnnClass4ConvKS3Lin128ChannExpand, self).__init__()

        input_shape = torch.tensor([num_features] + list(box_shape))
    
        self.conv_layers = nn.Sequential(
            nn.BatchNorm3d(input_shape[0]),

            nn.Conv3d(input_shape[0], int(input_shape[0]//2), kernel_size=1),
            nn.BatchNorm3d(input_shape[0]//2),
            nn.ReLU(),

            nn.Conv3d(int(input_shape[0]//2), int(input_shape[0]//2), kernel_size=1),
            nn.BatchNorm3d(input_shape[0]//2),
            #nn.MaxPool3d((2,2,2)),
            nn.ReLU(),

            nn.Conv3d(int(input_shape[0]//2),int(input_shape[0]), kernel_size=3),
            nn.BatchNorm3d(input_shape[0]),
            nn.MaxPool3d((2,2,2)),
            nn.ReLU(),

            nn.Conv3d(int(input_shape[0]), int(input_shape[0]*2), kernel_size=3),
            nn.BatchNorm3d(input_shape[0]*2),
            nn.MaxPool3d((2,2,2)),
            nn.ReLU(),
        )

        self.flatten_shape = len(self.conv_layers(torch.rand(input_shape.tolist()).unsqueeze(0)).flatten())

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.BatchNorm1d(self.flatten_shape),

            nn.Linear(self.flatten_shape, 128),
            nn.ReLU(),
            nn.Dropout(),

            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(),

            nn.Linear(128, 2)
        )

    def forward(self, data):
        x = self.conv_layers(data.x)
        x = self.fc_layers(x)
        return x
