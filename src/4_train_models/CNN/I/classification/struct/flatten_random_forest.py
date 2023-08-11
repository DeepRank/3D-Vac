from torchsummary import summary
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data_utils
from torch.autograd import Variable

import numpy as np
import random
import h5py
import time
import sys

from deeprank.learn import DataSet

class MLP(nn.Module):
    def __init__(self, input_shape, input_size, hidden_size, output_size, 
                 blind_ratio_voxels, blind_ratio_features):
        super(MLP, self).__init__()
        # self.bn0 = nn.BatchNorm3d(input_shape[0]),
        self.input_shape = input_shape
        self.input_size = input_size
        self.hidden_size= hidden_size
        self.output_size = output_size
        self.blind_ratio_voxels = blind_ratio_voxels
        self.blind_ratio_features = blind_ratio_features

        #Projection layers 
        self.conv1 = nn.Conv3d(self.input_size, self.input_size, kernel_size=1)
        self.bn1 = nn.BatchNorm3d(self.input_size)

        self.conv2 = nn.Conv3d(self.input_size, self.input_size, kernel_size=1)
        self.bn2 = nn.BatchNorm3d(input_size)
        
        self.pool = nn.MaxPool3d((2,2,2))
        
        self.flatten_shape = len((self.pool(torch.rand([1, 24, 35, 30, 30]))).flatten())
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(self.flatten_shape, self.hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        
        self.fc3 = nn.Linear(self.hidden_size, self.output_size)
        
        self.get_masks()
        
    def get_masks(self):
        
        # input shape: (batch size, channels, x, y, z)
        # Create mask to blind part of the features
        features_mask = torch.randperm(self.input_shape[1])[:int(self.blind_ratio_features * self.input_shape[1])]
        self.features_mask = torch.ones(self.input_shape)
        for feat in features_mask:
            self.features_mask[:,feat,:,:,:] *= 0
        # Create mask to blind part of the data
        voxels_mask = [[random.choice(range(coord)) for coord in self.input_shape[2:]] for voxel in range(int(self.blind_ratio_voxels * self.input_shape[2] * self.input_shape[3] * self.input_shape[4]))]
        self.voxels_mask = torch.ones(self.input_shape)
        for v in voxels_mask:
            self.voxels_mask[:,:,v[0],v[1],v[2]] *= 0
                
    def forward(self, x):
        x_copy = x.clone()
        # Set inputs of blinded features to 0
        x_copy*=self.features_mask
        x_copy*=self.voxels_mask
        
        out = self.conv1(x_copy)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.pool(out)
        out = self.flatten(out)
        #raise Exception((out.shape, self.flatten_shape))
        
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.fc2(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.fc3(out)
        out = self.relu(out)
        out = self.dropout(out)
        return out

# # Define the ensemble model class
# class BlindMLPEnsemble(nn.Module):
#     def __init__(self, num_models, input_shape, input_size, hidden_size, 
#                 output_size, blind_ratio_voxels, blind_ratio_features):
#         super(BlindMLPEnsemble, self).__init__()
#         self.num_models = num_models
#         self.models = nn.ModuleList()
#         self.blind_ratio_voxels = blind_ratio_voxels
#         self.blind_ratio_features = blind_ratio_features
#         self.input_size = input_size
#         for i in range(num_models):
#             # input shape: (batch size, channels, x, y, z)
#             model = MLP(self.input_size, hidden_size, output_size)
#             # Create mask to blind part of the features
#             features_mask = torch.randperm(input_shape[1])[:int(self.blind_ratio_features * input_shape[1])]
#             model.features_mask = torch.ones(input_shape)
#             for feat in features_mask:
#                 model.features_mask[:,feat,:,:,:] *= 0
#             # Create mask to blind part of the data
#             voxels_mask = [[random.choice(range(coord)) for coord in input_shape[2:]] for voxel in range(int(self.blind_ratio_voxels * input_shape[2] * input_shape[3] * input_shape[4]))]
#             model.voxels_mask = torch.ones(input_shape)
#             for v in voxels_mask:
#                 model.voxels_mask[:,:,v[0],v[1],v[2]] *= 0
#             self.models.append(model)

#     def forward(self, mini_batches):
#         outputs = []
#         targets = []
#         # skip_model = random.choice(range(len(self.models)))
#         for i, model in enumerate(self.models):
#             inputs = mini_batches[i]['feature']
#             target = mini_batches[i]['target']
#             x, target = _get_variables(inputs, target)
            
#             x_copy = x.clone()
#             # Set inputs of blinded features to 0
#             x_copy*=model.features_mask
#             x_copy*=model.voxels_mask
#             #raise Exception(x_copy.shape)
#             outputs.append(model(x_copy))
#             targets.append(target)
            
#         return torch.mean(torch.stack(outputs), dim=0)

def get_hdf5_data(hdf5_file, n_models, blind_ratio_data, train_batch_size=16):
    
    data_set = DataSet(train_database=hdf5_file,
    chain1 = "M",
    chain2 = "P",
    grid_info = (35,30,30),
    select_feature = {'AtomicDensities_ind': 'all',
        "Feature_ind": ['Edesolv', 'anch', 'SkipGram*',
        'RCD_*', 'bsa', 'charge', 'coulomb', 'vdwaals']},
    select_target = "BIN_CLASS",
    normalize_features = False, #Change back to True
    normalize_targets = False,
    pair_chain_feature = np.add,
    mapfly = False,
    tqdm = True,
    clip_features = False,
    process = True,
    )
    
    loaders = []
    for model in range(n_models):
        indices = random.sample(data_set.index_train, int(blind_ratio_data * len(data_set.index_train)))
        train_sampler = data_utils.sampler.SubsetRandomSampler(indices)
        train_loader = data_utils.DataLoader(
        data_set,
        batch_size=train_batch_size,
        sampler=train_sampler,
        pin_memory=False,
        num_workers=1,
        prefetch_factor=2,
        shuffle=False,
        drop_last=True,
        persistent_workers=False)
        loaders.append(train_loader)

    return loaders
    
def _get_variables(inputs, targets, cuda=False, pin=False, task='class'):

    """Convert the feature/target in torch.Variables.

    The format is different for regression where the targets are float
    and classification where they are int.

    Args:
        inputs (np.array): raw features
        targets (np.array): raw target values

    Returns:
        torch.Variable: features
        torch.Variable: target values
    """

    # if cuda is available
    if cuda:
        inputs = inputs.cuda(non_blocking=pin)
        targets = targets.cuda(non_blocking=pin)

    # get the variable as float by default
    inputs, targets = Variable(inputs).float(), Variable(targets).float()

    # change the targets to long for classification
    if task == 'class':
        targets = targets.long()

    return inputs, targets
    
if __name__=='__main__':
    # Define the hyperparameters
    batch_size = 16
    n_channels = 24
    input_size = 24
    hidden_size = 100
    output_size = 2
    num_models = 5
    learning_rate = 0.01
    num_epochs = 20
    blind_ratio_data = 0.2
    blind_ratio_voxels = 0.1
    blind_ratio_features = 0.3

    input_shape = [batch_size, n_channels, 35, 30, 30]
    # Load the data
    data_loaders = get_hdf5_data(
        '/projects/0/einf2380/data/pMHCI/features_output_folder/CNN/splits_HLA_quant_shuffled/shuffled/0/train.hdf5', 
        train_batch_size=batch_size, blind_ratio_data=blind_ratio_data, n_models=num_models)

    # Create the ensemble model and optimizer
    models = { i : {'model': MLP(input_shape, input_size, hidden_size, 
                            output_size, blind_ratio_voxels, blind_ratio_features), 
                            'loader': data_loaders[i],
                            'optimizer': ''} for i in range(num_models)}
    for i in models:
        models[i]['optimizer'] = optim.SGD(models[i]['model'].parameters(), lr=learning_rate)

    # model = BlindMLPEnsemble(num_models, input_shape, input_size, hidden_size, 
    #                         output_size, blind_ratio_voxels, blind_ratio_features)
    # optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    sys.stdout.flush()
    torch.autograd.set_detect_anomaly(True)
    # Train the ensemble model
    for i in models:
        print(f'Model {i}')
        for epoch in range(num_epochs):
            t0 = time.time()
        
            total_minibatches = len(models[i]['loader']) 
            n_mini = 0

            for mini_batch in models[i]['loader']:
                
                # Zero the gradients
                models[i]['optimizer'].zero_grad()
            
                n_mini += 1
                inputs = mini_batch['feature']
                targets = mini_batch['target']
                inputs, targets = _get_variables(inputs, targets)
                targets = targets.view(-1)
                print((len(targets), targets.tolist().count(1), targets.tolist().count(0)))
                raise Exception('ok')
                #mol = mini_batch['mol']
                # Forward pass
                y_pred = models[i]['model'](inputs)

                #raise Exception(y_pred.shape, targets.shape)
                # Compute the loss
                loss = nn.CrossEntropyLoss()(y_pred, targets)

                # Backward pass
                loss.backward()

                # Update the parameters
                models[i]['optimizer'].step()

            t1 = time.time()
            print('Epoch [{}/{}], Time: {:.2f}, Loss: {:.4f}'.format(epoch+1, num_epochs, t1-t0, loss.item()))
            # # Print the loss every 10 epochs
            # if (epoch + 1) % 10 == 0:
            #     print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))
        
        torch.save(models[i]['model'].state_dict(), f'/projects/0/einf2380/data/pMHCI/trained_models/Flattened_CNN/model_{i}.pth')
        break
    sys.stdout.flush()
