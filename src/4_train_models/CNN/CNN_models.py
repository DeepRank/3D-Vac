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
            nn.MaxPool3d((2,2,2)),
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
        x = self.fc_layers(x)
        return x

class CnnClass4ConvKS3XYZ(nn.Module):
    
    def __init__(self, input_shape):
        """This class is the architecture for DeepRank's CNN framework.
        Made of one sequential convolutional layers array followed by
        one fully connected layers array.
        Used in CNN/I/classification/struct/cnn_baseline.py as an argument
        for the architecture class.

        Args:
            input_shape (int): Number of features (64 as of now)
        """
        super(CnnClass4ConvKS3, self).__init__()
        
        input_shape = torch.tensor(input_shape)
        input_shape[0] += 3
        #input_shape[0] = input_shape[0] +3
        
        x = torch.arange(0, 1, 1/35)
        x = x.reshape(1, 35, 1, 1)
        x = x.repeat(1, 1, 30, 30)
        #x = x.reshape(1, 35)
        y = torch.arange(0, 1, 1/30)
        y = y.reshape(1, 1, 30, 1)
        y = y.repeat(1, 35, 1, 30)
        #y = y.reshape(1, 30)
        z = torch.arange(0, 1, 1/30)
        z = z.reshape(1, 1, 1, 30)
        z = z.repeat(1, 35, 30, 1)
        #z = z.reshape(1, 30)
        
        self.xyz = nn.Parameter(torch.cat([x,y,z], 0).unsqueeze(0))
    
        self.conv_layers = nn.Sequential(
            nn.BatchNorm3d(input_shape[0]),

            nn.Conv3d(input_shape[0], int(input_shape[0]//2), kernel_size=1),
            nn.BatchNorm3d(input_shape[0]//2),
            nn.ReLU(),

            nn.Conv3d(int(input_shape[0]//2), int(input_shape[0]//2), kernel_size=1),
            nn.BatchNorm3d(input_shape[0]//2),
            nn.MaxPool3d((2,2,2)),
            nn.ReLU(),

            nn.Conv3d(int(input_shape[0]//2),int(input_shape[0]//2), kernel_size=3),
            nn.BatchNorm3d(input_shape[0]//2),
            nn.MaxPool3d((2,2,2)),
            nn.ReLU(),

            nn.Conv3d(int(input_shape[0]//2), int(input_shape[0]//2), kernel_size=3),
            nn.BatchNorm3d(input_shape[0]//2),
            nn.MaxPool3d((2,2,2)),
            nn.ReLU(),
        )

        self.flatten_shape = len(self.conv_layers(torch.rand(input_shape.tolist()).unsqueeze(0)).flatten())

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
        x = torch.cat([x, self.xyz.repeat(x.shape[0],1,1,1,1)], 1)
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

class CnnReg4ConvKS3XYZ(nn.Module):
    
    def __init__(self, input_shape):
        """This class is the architecture for DeepRank's CNN framework.
        Made of one sequential convolutional layers array followed by
        one fully connected layers array.
        Used in CNN/I/classification/struct/cnn_baseline.py as an argument
        for the architecture class.

        Args:
            input_shape (int): Number of features (64 as of now)
        """
        super(CnnReg4ConvKS3, self).__init__()
        
        input_shape = torch.tensor(input_shape)
        input_shape[0] += 3
        #input_shape[0] = input_shape[0] +3
        
        x = torch.arange(0, 1, 1/35)
        x = x.reshape(1, 35, 1, 1)
        x = x.repeat(1, 1, 30, 30)
        #x = x.reshape(1, 35)
        y = torch.arange(0, 1, 1/30)
        y = y.reshape(1, 1, 30, 1)
        y = y.repeat(1, 35, 1, 30)
        #y = y.reshape(1, 30)
        z = torch.arange(0, 1, 1/30)
        z = z.reshape(1, 1, 1, 30)
        z = z.repeat(1, 35, 30, 1)
        #z = z.reshape(1, 30)
        
        self.xyz = nn.Parameter(torch.cat([x,y,z], 0).unsqueeze(0))
    
        self.conv_layers = nn.Sequential(
            nn.BatchNorm3d(input_shape[0]),

            nn.Conv3d(input_shape[0], int(input_shape[0]//2), kernel_size=1),
            nn.BatchNorm3d(input_shape[0]//2),
            nn.ReLU(),

            nn.Conv3d(int(input_shape[0]//2), int(input_shape[0]//2), kernel_size=1),
            nn.BatchNorm3d(input_shape[0]//2),
            nn.MaxPool3d((2,2,2)),
            nn.ReLU(),

            nn.Conv3d(int(input_shape[0]//2),int(input_shape[0]//2), kernel_size=3),
            nn.BatchNorm3d(input_shape[0]//2),
            nn.MaxPool3d((2,2,2)),
            nn.ReLU(),

            nn.Conv3d(int(input_shape[0]//2), int(input_shape[0]//2), kernel_size=3),
            nn.BatchNorm3d(input_shape[0]//2),
            nn.MaxPool3d((2,2,2)),
            nn.ReLU(),
        )

        self.flatten_shape = len(self.conv_layers(torch.rand(input_shape.tolist()).unsqueeze(0)).flatten())

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.BatchNorm1d(self.flatten_shape),

            nn.Linear(self.flatten_shape, 1000),
            nn.ReLU(),
            nn.Dropout(),

            nn.Linear(1000, 1000),
            nn.ReLU(),
            nn.Dropout(),

            nn.Linear(1000, 1)
        )

    def forward(self, x):
        x = torch.cat([x, self.xyz.repeat(x.shape[0],1,1,1,1)], 1)
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

class CnnClass4ConvKS3Lin128(nn.Module):
    
    def __init__(self, input_shape):
        """This class is the architecture for DeepRank's CNN framework.
        Made of one sequential convolutional layers array followed by
        one fully connected layers array.
        Used in CNN/I/classification/struct/cnn_baseline.py as an argument
        for the architecture class.

        Args:
            input_shape (int): Number of features (64 as of now)
        """
        super(CnnClass4ConvKS3Lin128, self).__init__()
        
        input_shape = torch.tensor(input_shape)
        
    
        self.conv_layers = nn.Sequential(
            nn.BatchNorm3d(input_shape[0]),

            nn.Conv3d(input_shape[0], int(input_shape[0]//2), kernel_size=1),
            nn.BatchNorm3d(input_shape[0]//2),
            nn.ReLU(),

            nn.Conv3d(int(input_shape[0]//2), int(input_shape[0]//2), kernel_size=1),
            nn.BatchNorm3d(input_shape[0]//2),
            nn.MaxPool3d((2,2,2)),
            nn.ReLU(),

            nn.Conv3d(int(input_shape[0]//2),int(input_shape[0]//2), kernel_size=3),
            nn.BatchNorm3d(input_shape[0]//2),
            nn.MaxPool3d((2,2,2)),
            nn.ReLU(),

            nn.Conv3d(int(input_shape[0]//2), int(input_shape[0]//2), kernel_size=3),
            nn.BatchNorm3d(input_shape[0]//2),
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

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

class CnnReg4ConvKS3Lin128(nn.Module):
    
    def __init__(self, input_shape):
        """This class is the architecture for DeepRank's CNN framework.
        Made of one sequential convolutional layers array followed by
        one fully connected layers array.
        Used in CNN/I/classification/struct/cnn_baseline.py as an argument
        for the architecture class.

        Args:
            input_shape (int): Number of features (64 as of now)
        """
        super(CnnReg4ConvKS3Lin128, self).__init__()
        
        input_shape = torch.tensor(input_shape)
    
        self.conv_layers = nn.Sequential(
            nn.BatchNorm3d(input_shape[0]),

            nn.Conv3d(input_shape[0], int(input_shape[0]//2), kernel_size=1),
            nn.BatchNorm3d(input_shape[0]//2),
            nn.ReLU(),

            nn.Conv3d(int(input_shape[0]//2), int(input_shape[0]//2), kernel_size=1),
            nn.BatchNorm3d(input_shape[0]//2),
            nn.MaxPool3d((2,2,2)),
            nn.ReLU(),

            nn.Conv3d(int(input_shape[0]//2),int(input_shape[0]//2), kernel_size=3),
            nn.BatchNorm3d(input_shape[0]//2),
            nn.MaxPool3d((2,2,2)),
            nn.ReLU(),

            nn.Conv3d(int(input_shape[0]//2), int(input_shape[0]//2), kernel_size=3),
            nn.BatchNorm3d(input_shape[0]//2),
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

            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x
    
class CnnClass4ConvKS3Lin128SoftMax(nn.Module):
    
    def __init__(self, input_shape):
        """This class is the architecture for DeepRank's CNN framework.
        Made of one sequential convolutional layers array followed by
        one fully connected layers array.
        Used in CNN/I/classification/struct/cnn_baseline.py as an argument
        for the architecture class.

        Args:
            input_shape (int): Number of features (64 as of now)
        """
        super(CnnClass4ConvKS3Lin128SoftMax, self).__init__()
        
        input_shape = torch.tensor(input_shape)
        
    
        self.conv_layers = nn.Sequential(
            nn.BatchNorm3d(input_shape[0]),

            nn.Conv3d(input_shape[0], int(input_shape[0]//2), kernel_size=1),
            nn.BatchNorm3d(input_shape[0]//2),
            nn.ReLU(),

            nn.Conv3d(int(input_shape[0]//2), int(input_shape[0]//2), kernel_size=1),
            nn.BatchNorm3d(input_shape[0]//2),
            nn.MaxPool3d((2,2,2)),
            nn.ReLU(),

            nn.Conv3d(int(input_shape[0]//2),int(input_shape[0]//2), kernel_size=3),
            nn.BatchNorm3d(input_shape[0]//2),
            nn.MaxPool3d((2,2,2)),
            nn.ReLU(),

            nn.Conv3d(int(input_shape[0]//2), int(input_shape[0]//2), kernel_size=3),
            nn.BatchNorm3d(input_shape[0]//2),
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

            nn.Linear(128, 2),
            nn.Softmax()
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

class CnnReg4ConvKS3Lin128Sigmoid(nn.Module):
    
    def __init__(self, input_shape):
        """This class is the architecture for DeepRank's CNN framework.
        Made of one sequential convolutional layers array followed by
        one fully connected layers array.
        Used in CNN/I/classification/struct/cnn_baseline.py as an argument
        for the architecture class.

        Args:
            input_shape (int): Number of features (64 as of now)
        """
        super(CnnReg4ConvKS3Lin128Sigmoid, self).__init__()
        
        input_shape = torch.tensor(input_shape)
    
        self.conv_layers = nn.Sequential(
            nn.BatchNorm3d(input_shape[0]),

            nn.Conv3d(input_shape[0], int(input_shape[0]//2), kernel_size=1),
            nn.BatchNorm3d(input_shape[0]//2),
            nn.ReLU(),

            nn.Conv3d(int(input_shape[0]//2), int(input_shape[0]//2), kernel_size=1),
            nn.BatchNorm3d(input_shape[0]//2),
            nn.MaxPool3d((2,2,2)),
            nn.ReLU(),

            nn.Conv3d(int(input_shape[0]//2),int(input_shape[0]//2), kernel_size=3),
            nn.BatchNorm3d(input_shape[0]//2),
            nn.MaxPool3d((2,2,2)),
            nn.ReLU(),

            nn.Conv3d(int(input_shape[0]//2), int(input_shape[0]//2), kernel_size=3),
            nn.BatchNorm3d(input_shape[0]//2),
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

            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

class CnnClass4ConvKS3Lin128ChannExpand(nn.Module):
    
    def __init__(self, input_shape):
        """This class is the architecture for DeepRank's CNN framework.
        Made of one sequential convolutional layers array followed by
        one fully connected layers array.
        Used in CNN/I/classification/struct/cnn_baseline.py as an argument
        for the architecture class.

        Args:
            input_shape (int): Number of features (64 as of now)
        """
        super(CnnClass4ConvKS3Lin128ChannExpand, self).__init__()
        
        input_shape = torch.tensor(input_shape)
        
    
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

            nn.Linear(128, 2),
            nn.Softmax()
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

class CnnReg4ConvKS3Lin128SigmoidChannExpand(nn.Module):
    
    def __init__(self, input_shape):
        """This class is the architecture for DeepRank's CNN framework.
        Made of one sequential convolutional layers array followed by
        one fully connected layers array.
        Used in CNN/I/classification/struct/cnn_baseline.py as an argument
        for the architecture class.

        Args:
            input_shape (int): Number of features (64 as of now)
        """
        super(CnnReg4ConvKS3Lin128SigmoidChannExpand, self).__init__()
        
        input_shape = torch.tensor(input_shape)
    
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

            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

class CnnClass4ConvKS3Lin128ChannExpandXYZ(nn.Module):
    
    def __init__(self, input_shape):
        """This class is the architecture for DeepRank's CNN framework.
        Made of one sequential convolutional layers array followed by
        one fully connected layers array.
        Used in CNN/I/classification/struct/cnn_baseline.py as an argument
        for the architecture class.

        Args:
            input_shape (int): Number of features (64 as of now)
        """
        super(CnnClass4ConvKS3Lin128ChannExpandXYZ, self).__init__()
        
        input_shape = torch.tensor(input_shape)
        input_shape[0] += 3
        #input_shape[0] = input_shape[0] +3
        
        x = torch.arange(0, 1, 1/35)
        x = x.reshape(1, 35, 1, 1)
        x = x.repeat(1, 1, 30, 30)
        #x = x.reshape(1, 35)
        y = torch.arange(0, 1, 1/30)
        y = y.reshape(1, 1, 30, 1)
        y = y.repeat(1, 35, 1, 30)
        #y = y.reshape(1, 30)
        z = torch.arange(0, 1, 1/30)
        z = z.reshape(1, 1, 1, 30)
        z = z.repeat(1, 35, 30, 1)
        #z = z.reshape(1, 30)

        self.xyz = nn.Parameter(torch.cat([x,y,z], 0).unsqueeze(0))
        
    
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

            nn.Linear(128, 2),
            nn.Softmax()
        )

    def forward(self, x):
        x = torch.cat([x, self.xyz.repeat(x.shape[0],1,1,1,1)], 1)
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

class CnnReg4ConvKS3Lin128SigmoidChannExpandXYZ(nn.Module):
    
    def __init__(self, input_shape):
        """This class is the architecture for DeepRank's CNN framework.
        Made of one sequential convolutional layers array followed by
        one fully connected layers array.
        Used in CNN/I/classification/struct/cnn_baseline.py as an argument
        for the architecture class.

        Args:
            input_shape (int): Number of features (64 as of now)
        """
        super(CnnReg4ConvKS3Lin128SigmoidChannExpandXYZ, self).__init__()
        
        input_shape = torch.tensor(input_shape)
        input_shape[0] += 3
        
        x = torch.arange(0, 1, 1/35)
        x = x.reshape(1, 35, 1, 1)
        x = x.repeat(1, 1, 30, 30)

        y = torch.arange(0, 1, 1/30)
        y = y.reshape(1, 1, 30, 1)
        y = y.repeat(1, 35, 1, 30)

        z = torch.arange(0, 1, 1/30)
        z = z.reshape(1, 1, 1, 30)
        z = z.repeat(1, 35, 30, 1)


        self.xyz = nn.Parameter(torch.cat([x,y,z], 0).unsqueeze(0))
    
        self.conv_layers = nn.Sequential(
            nn.BatchNorm3d(input_shape[0]),

            nn.Conv3d(input_shape[0], int(input_shape[0]//2), kernel_size=1),
            nn.BatchNorm3d(input_shape[0]//2),
            nn.ReLU(),

            nn.Conv3d(int(input_shape[0]//2), int(input_shape[0]//2), kernel_size=1),
            nn.BatchNorm3d(input_shape[0]//2),

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

            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = torch.cat([x, self.xyz.repeat(x.shape[0],1,1,1,1)], 1)
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x