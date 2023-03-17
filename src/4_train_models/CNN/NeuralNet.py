#!/usr/bin/env python
import os
import sys
import time
import pickle

import h5py
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.lines as mlines
import matplotlib.transforms as mtransforms
import numpy as np
from numpy import mean
import pandas as pd
import warnings

import torch
import torch.cuda
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils
from torchsummary import summary

from deeprank.config import logger
from deeprank.learn import DataSet, rankingMetrics#, classMetrics
from torch.autograd import Variable

from . import classMetrics

matplotlib.use('agg')
torch.manual_seed(0)

class NeuralNet():

    def __init__(self, data_set, model,
                 model_type='3d', proj2d=0, task='reg',
                 optimizer='sgd',
                 learn_rate=1e-3,
                 contrastive = False,
                 class_weights = None,
                 pretrained_model=None,
                 chain1='A',
                 chain2='B',
                 cuda=False, ngpu=0,
                 plot=False,
                 save_hitrate=False,
                 save_classmetrics=False,
                 outdir='./'):
        """Train a Convolutional Neural Network for DeepRank.

        Args:
            data_set (deeprank.DataSet or list(str)): Data set used for
                training or testing.
                - deeprank.DataSet for training;
                - str or list(str), e.g. 'x.hdf5', ['x1.hdf5', 'x2.hdf5'],
                for testing when pretrained model is loaded.

            model (nn.Module): Definition of the NN to use.
                Must subclass nn.Module.
                See examples in model2d.py and model3d.py

            model_type (str): Type of model we want to use.
                Must be '2d' or '3d'.
                If we specify a 2d model, the data set is automatically
                converted to the correct format.

            proj2d (int): Defines how to slice the 3D volumetric data to generate
                2D data. Allowed values are 0, 1 and 2, which are to slice along
                the YZ, XZ or XY plane, respectively.

            task (str 'reg' or 'class'): Task to perform.
                - 'reg' for regression
                - 'class' for classification.
                The loss function, the target datatype and plot functions
                will be autmatically adjusted depending on the task.

            class_weights (Tensor): a manual rescaling weight given to
                each class. If given, has to be a Tensor of size #classes.
                Only applicable on 'class' task.

            pretrained_model (str): Saved model to be used for further
                training or testing. When using pretrained model,
                remember to set the following 'chain1' and 'chain2' for
                the new data.
            chain1 (str): first chain ID of new data when using pretrained model
            chain2 (str): second chain ID of new data when using pretrained model

            cuda (bool): Use CUDA.

            ngpu (int): number of GPU to be used.

            plot (bool): Plot the prediction results.

            save_hitrate (bool): Save and plot hit rate.

            save_classmetrics (bool): Save and plot classification metrics.
                Classification metrics include:
                - accuracy(ACC)
                - sensitivity(TPR)
                - specificity(TNR)

            outdir (str): output directory

        Raises:
            ValueError: if dataset format is not recognized
            ValueError: if task is not recognized

        Examples:
            Train models:
            >>> data_set = Dataset(...)
            >>> model = NeuralNet(data_set, cnn,
            ...                   model_type='3d', task='reg',
            ...                   plot=True, save_hitrate=True,
            ...                   outdir='./out/')
            >>> model.train(nepoch = 50, divide_trainset=0.8,
            ...             train_batch_size = 5, num_workers=0)

            Test a model on new data:
            >>> data_set = ['test01.hdf5', 'test02.hdf5']
            >>> model = NeuralNet(data_set, cnn,
            ...                   pretrained_model = './model.pth.tar',
            ...                   outdir='./out/')
            >>> model.test()
        """

        # ------------------------------------------
        # Dataset
        # ------------------------------------------

        # data set and model
        self.data_set = data_set

        self.contrastive = contrastive

        # pretrained model
        self.pretrained_model = pretrained_model

        self.class_weights = class_weights

        if isinstance(self.data_set, (str, list)) and pretrained_model is None:
            raise ValueError(
                'Argument data_set must be a DeepRankDataSet object\
                              when no pretrained model is loaded')

        # load the model
        if self.pretrained_model is not None:

            if not cuda:
                self.state = torch.load(self.pretrained_model,
                                        map_location='cpu')
            else:
                self.state = torch.load(self.pretrained_model)

            # create the dataset if required
            # but don't process it yet
            if isinstance(self.data_set, (str, list)):
                self.data_set = DataSet(self.data_set, chain1=chain1,
                                    chain2=chain2, process=False)

            # load the model and
            # change dataset parameters
            self.load_data_params()

            # process it
            self.data_set.process_dataset()

        # convert the data to 2d if necessary
        if model_type == '2d':

            self.data_set.transform = True
            self.data_set.proj2D = proj2d
            self.data_set.get_input_shape()

        # ------------------------------------------
        # CUDA
        # ------------------------------------------

        # CUDA required
        self.cuda = cuda
        self.ngpu = ngpu

        # handles GPU/CUDA
        if self.ngpu > 0:
            self.cuda = True

        if self.ngpu == 0 and self.cuda:
            self.ngpu = 1

        # ------------------------------------------
        # Regression or classification
        # ------------------------------------------

        # task to accomplish
        self.task = task

        # Set the loss functiom
        if self.task == 'class' or self.contrastive:
            self.criterion = nn.CrossEntropyLoss(weight = self.class_weights, reduction='mean')
            self._plot = self._plot_boxplot_class
            self.data_set.normalize_targets = False
        elif self.task == 'reg':
            self.criterion = nn.MSELoss(reduction='mean')
            self._plot = self._plot_scatter_hist_reg
        else:
            raise ValueError(
                f"Task {self.task} not recognized. Options are:\n\t "
                f"reg': regression \n\t 'class': classifiation\n")

        # ------------------------------------------
        # Output
        # ------------------------------------------

        # plot or not plot
        self.plot = plot

        # plot and save hitrate or not
        self.save_hitrate = save_hitrate

        # plot and save classification metrics or not
        self.save_classmetrics = save_classmetrics
        if self.save_classmetrics:
            if self.task == 'class' or self.contrastive:
                self.metricnames = ['acc', 'tpr', 'tnr', 'mcc', 'auc', 'f1']
            elif self.task == 'reg':
                self.metricnames = ['rmse']

        # output directory
        self.outdir = outdir
        if not os.path.isdir(self.outdir):
            os.mkdir(outdir)

        # ------------------------------------------
        # Network
        # ------------------------------------------

        # load the model
        if model.__name__ == 'CnnLocalPred':
            self.net = model(self.data_set.input_shape) #TODO this breaks if other network class
            self.attention = True
        else:
            self.net = model(self.data_set.input_shape)
            self.attention = False

        # print model summary
        sys.stdout.flush()
        if cuda is True:
            device = torch.device("cuda")  # PyTorch v0.4.0
        else:
            device = torch.device("cpu")
        summary(self.net.to(device),
                self.data_set.input_shape,
                device=device.type)
        # print('#*'*100)
        # print('Not printing torch summary because of errors..')
        sys.stdout.flush()

        # load parameters of pretrained model if provided
        if self.pretrained_model:
            # a prefix 'module.' is added to parameter names if
            # torch.nn.DataParallel was used
            # https://pytorch.org/docs/stable/nn.html#torch.nn.DataParallel
            if self.state['cuda']:
                for paramname in list(self.state['state_dict'].keys()):
                    paramname_new = paramname.lstrip('module.')
                    if paramname != paramname_new:
                        self.state['state_dict'][paramname_new] = \
                            self.state['state_dict'][paramname]
                        del self.state['state_dict'][paramname]
            self.load_model_params()

        # multi-gpu
        if self.ngpu > 1:
            ids = [i for i in range(self.ngpu)]
            self.net = nn.DataParallel(self.net, device_ids=ids).cuda()
        # cuda compatible
        elif self.cuda:
            self.net = self.net.cuda()

        # set the optimizer
        if optimizer == 'sgd':
            self.optimizer = optim.SGD(self.net.parameters(),
                                    lr=learn_rate, # previous 0.005
                                    momentum=0.9,
                                    weight_decay=0.001)
        elif optimizer == 'adam':
            self.optimizer = optim.Adam(self.net.parameters(),
                                        lr=learn_rate, # previous 0.001
                                        eps=1e-08,
                                        weight_decay=0)
        else:
            print('Optimizer not implemented yet, please add to the module or check your spelling')
            raise(NotImplementedError)
        print(f'Optimizer: {optimizer} with learn rate: {learn_rate}')
        
        if self.pretrained_model:
            self.load_optimizer_params()

        # ------------------------------------------
        # print
        # ------------------------------------------

        logger.info('\n')
        logger.info('=' * 40)
        logger.info('=\t Convolution Neural Network')
        logger.info(f'=\t model   : {model_type}')
        logger.info(f'=\t CNN      : {model.__name__}')

        for feat_type, feat_names in self.data_set.select_feature.items():
            logger.info(f'=\t features : {feat_type}')
            for name in feat_names:
                logger.info(f'=\t\t     {name}')
        if self.data_set.pair_chain_feature is not None:
            logger.info(f'=\t Pair     : '
                        f'{self.data_set.pair_chain_feature.__name__}')
        logger.info(f'=\t targets  : {self.data_set.select_target}')
        logger.info(f'=\t CUDA     : {str(self.cuda)}')
        if self.cuda:
            logger.info(f'=\t nGPU     : {self.ngpu}')
        logger.info('=' * 40 + '\n')

        # check if CUDA works
        if self.cuda and not torch.cuda.is_available():
            logger.error(
                f' --> CUDA not deteceted: Make sure that CUDA is installed '
                f'and that you are running on GPUs.\n'
                f' --> To turn CUDA of set cuda=False in NeuralNet.\n'
                f' --> Aborting the experiment \n\n')
            sys.exit()

    def train(self,
              nepoch=50,
              early_stop=None,
              save_fraction=1/2,
              divide_trainset=None,
              hdf5='epoch_data.hdf5',
              train_batch_size=64,
              pin_memory_cuda=True,
              preshuffle=True,
              preshuffle_seed=None,
              export_intermediate=True,
              num_workers=1,
              prefetch_factor=2,
              save_model='best',
              save_epoch='intermediate',
              hit_cutoff=None):
        """Perform a simple training of the model.

        Args:
            nepoch (int, optional): number of iterations

            divide_trainset (list, optional): the percentage assign to
                the training, validation and test set.
                Examples: [0.7, 0.2, 0.1], [0.8, 0.2], None

            hdf5 (str, optional): file to store the training results

            train_batch_size (int, optional): size of the batch

            preshuffle (bool, optional): preshuffle the dataset before
                dividing it.

            preshuffle_seed (int, optional): set random seed for preshuffle

            export_intermediate (bool, optional): export data at
                intermediate epochs.

            num_workers (int, optional): number of workers to be used to
                prepare the batch data

            save_model (str, optional): 'best' or 'all', save only the
                best model or all models.

            save_epoch (str, optional): 'intermediate' or 'all',
                save the epochs data to HDF5.

            hit_cutoff (float, optional): the cutoff used to define hit by
                comparing with docking models' target value, e.g. IRMSD value
        """
        self.hit_cutoff = hit_cutoff
        self.nepoch = nepoch

        logger.info(f'\n: Batch Size: {train_batch_size}')
        if self.cuda:
            logger.info(f': NGPU      : {self.ngpu}')

        # hdf5 support
        self.fname = os.path.join(self.outdir, hdf5)
        self.f5 = h5py.File(self.fname, 'w')

        # divide the set in train+ valid and test
        if divide_trainset is not None:
            # if divide_trainset is not None
            index_train, index_valid, index_test = self._divide_dataset(
                divide_trainset, preshuffle, preshuffle_seed)
        else:
            index_train = self.data_set.index_train
            index_valid = self.data_set.index_valid
            index_test = self.data_set.index_test

        # check if batch size is smaller than data set size
        if len(index_train) < train_batch_size:
            print("Batch size is larger than data set size, cannot create mini-batches")
            return

        logger.info(f': {len(index_train)} confs. for training')
        logger.info(f': {len(index_valid)} confs. for validation')
        logger.info(f': {len(index_test)} confs. for testing')

        # train the model
        t0 = time.time()
        if early_stop == None:
            early_stop = nepoch
        self._train(index_train, index_valid, index_test,
                    nepoch=nepoch,
                    early_stop_delta=early_stop,
                    save_fraction=save_fraction,
                    train_batch_size=train_batch_size,
                    pin_cuda = pin_memory_cuda,
                    export_intermediate=export_intermediate,
                    num_workers=num_workers,
                    prefetch_factor=prefetch_factor,
                    save_epoch=save_epoch,
                    save_model=save_model)

        self.f5.close()
        logger.info(
            f' --> Training done in {self.convertSeconds2Days(time.time()-t0)}')

        # save the model
        self.save_model(filename='last_model.pth.tar')

    @staticmethod
    def convertSeconds2Days(time):
        # input time in seconds

        time = int(time)
        day = time // (24 * 3600)
        time = time % (24 * 3600)
        hour = time // 3600
        time %= 3600
        minutes = time // 60
        time %= 60
        seconds = time
        return '%02d-%02d:%02d:%02d' % (day, hour, minutes, seconds)

    def test(self, hdf5='test_data.hdf5', hit_cutoff=None, has_target=False):
        """Test a predefined model on a new dataset.

        Args:
            hdf5 (str, optional): hdf5 file to store the test results
            hit_cutoff (float, optional): the cutoff used to define hit by
                comparing with docking models' target value, e.g. IRMSD value
            has_target(bool, optional): specify the presence (True) or absence (False) of 
                target values in the test set. No metrics can be computed if False.
            
        Examples:
            >>> # adress of the database
            >>> database = '1ak4.hdf5'
            >>> # Load the model in a new network instance
            >>> model = NeuralNet(database, cnn,
            ...                   pretrained_model='./model/model.pth.tar',
            ...                   outdir='./test/')
            >>> # test the model
            >>> model.test()
        """
        # output
        fname = os.path.join(self.outdir, hdf5)
        self.f5 = h5py.File(fname, 'w')

        # load pretrained model to get task and criterion
        self.load_nn_params()

        # load data
        index = list(range(self.data_set.__len__()))
        sampler = data_utils.sampler.SubsetRandomSampler(index)
        #sampler = torch.utils.data.distributed.DistributedSampler(self.data_set)
        loader = data_utils.DataLoader(self.data_set, sampler=sampler)

        # define the target value threshold to compute the hits if save_hitrate is True
        if self.save_hitrate and hit_cutoff is not None:
            self.hit_cutoff = hit_cutoff
            logger.info(f'Use hit cutoff {self.hit_cutoff}')

        # do test
        self.data = {}
        _, self.data['test'] = self._epoch(loader, train_model=False, has_target=has_target)

        # plot results
        if self.plot is True :
            self._plot(os.path.join(self.outdir, 'prediction.png'))
        if self.save_hitrate:
            self.plot_hit_rate(os.path.join(self.outdir + 'hitrate.png'))

        self._export_epoch_hdf5(0, self.data)
        self.f5.close()

    def save_model(self, filename='model.pth.tar'):
        """save the model to disk.

        Args:
            filename (str, optional): name of the file
        """
        filename = os.path.join(self.outdir, filename)

        state = {'state_dict': self.net.state_dict(),
                 'optimizer': self.optimizer.state_dict(),
                 'normalize_targets': self.data_set.normalize_targets,
                 'normalize_features': self.data_set.normalize_features,
                 'select_feature': self.data_set.select_feature,
                 'select_target': self.data_set.select_target,
                 'target_ordering': self.data_set.target_ordering,
                 'pair_chain_feature': self.data_set.pair_chain_feature,
                 'dict_filter': self.data_set.dict_filter,
                 'transform': self.data_set.transform,
                 'proj2D': self.data_set.proj2D,
                 'clip_features': self.data_set.clip_features,
                 'clip_factor': self.data_set.clip_factor,
                 'hit_cutoff': self.hit_cutoff,
                 'grid_info': self.data_set.grid_info,
                 'mapfly': self.data_set.mapfly,
                 'task': self.task,
                 'criterion': self.criterion,
                 'cuda': self.cuda
                 }

        if self.data_set.normalize_features:
            state['feature_mean'] = self.data_set.feature_mean
            state['feature_std'] = self.data_set.feature_std

        if self.data_set.normalize_targets:
            state['target_min'] = self.data_set.target_min
            state['target_max'] = self.data_set.target_max

        torch.save(state, filename)

    def pickle_metrics(self, filename='metrics.pkl'):
        filename = os.path.join(self.outdir, filename)
        with open(filename, 'wb') as outpkl:
            pickle.dump(self.classmetrics, outpkl)
        pass

    def load_model_params(self):
        """Get model parameters from a saved model."""
        self.net.load_state_dict(self.state['state_dict'])

    def load_optimizer_params(self):
        """Get optimizer parameters from a saved model."""
        self.optimizer.load_state_dict(self.state['optimizer'])

    def load_nn_params(self):
        """Get NeuralNet parameters from a saved model."""
        self.task = self.state['task']
        self.criterion = self.state['criterion']
        try:
            self.hit_cutoff = self.state['hit_cutoff']
        except Exception:
            logger.warning(f'No "hit_cutoff" found in {self.pretrained_model}. Please set it in function "test()" when doing benchmark"')

    def load_data_params(self):
        """Get dataset parameters from a saved model."""
        self.data_set.select_feature = self.state['select_feature']
        self.data_set.select_target = self.state['select_target']

        self.data_set.pair_chain_feature = self.state['pair_chain_feature']
        self.data_set.dict_filter = self.state['dict_filter']

        self.data_set.normalize_targets = self.state['normalize_targets']
        if self.data_set.normalize_targets:
            self.data_set.target_min = self.state['target_min']
            self.data_set.target_max = self.state['target_max']

        self.data_set.normalize_features = self.state['normalize_features']
        if self.data_set.normalize_features:
            self.data_set.feature_mean = self.state['feature_mean']
            self.data_set.feature_std = self.state['feature_std']

        self.data_set.transform = self.state['transform']
        self.data_set.proj2D = self.state['proj2D']
        self.data_set.target_ordering = self.state['target_ordering']
        self.data_set.clip_features = self.state['clip_features']
        self.data_set.clip_factor = self.state['clip_factor']
        self.data_set.mapfly = self.state['mapfly']
        self.data_set.grid_info = self.state['grid_info']

    def _divide_dataset(self, divide_set, preshuffle, preshuffle_seed):
        """Divide the data set into training, validation and test
        according to the percentage in divide_set.

        Args:
            divide_set (list(float)): percentage used for
                training/validation/test.
                Example: [0.8, 0.1, 0.1], [0.8, 0.2]
            preshuffle (bool): shuffle the dataset before dividing it
            preshuffle_seed (int, optional): set random seed

        Returns:
            list(int),list(int),list(int): Indices of the
                training/validation/test set.
        """
        # if user only provided one number
        # we assume it's the training percentage
        if not isinstance(divide_set, list):
            divide_set = [divide_set, 1. - divide_set]

        # if user provided 3 number and testset
        if len(divide_set) == 3 and self.data_set.test_database is not None:
            divide_set = [divide_set[0], 1. - divide_set[0]]
            logger.info(f'  : test data set AND test in training set detected\n'
                        f'  : Divide training set as '
                        f'{divide_set[0]} train {divide_set[1]} valid.\n'
                        f'  : Keep test set for testing')

        # preshuffle
        if preshuffle:
            if preshuffle_seed is not None and not isinstance(
                    preshuffle_seed, int):
                preshuffle_seed = int(preshuffle_seed)
            np.random.seed(preshuffle_seed)
            np.random.shuffle(self.data_set.index_train)

        # size of the subset for training
        ntrain = int(np.ceil(float(self.data_set.ntrain) * divide_set[0]))
        nvalid = int(np.floor(float(self.data_set.ntrain) * divide_set[1]))

        # indexes train and valid
        index_train = self.data_set.index_train[:ntrain]
        index_valid = self.data_set.index_train[ntrain:ntrain + nvalid]

        # index of test depending of the situation
        if len(divide_set) == 3:
            index_test = self.data_set.index_train[ntrain + nvalid:]
        else:
            index_test = self.data_set.index_test

        return index_train, index_valid, index_test

    def _train(self, index_train, index_valid, index_test,
               nepoch=50, train_batch_size=64, pin_cuda=True,
               export_intermediate=False, num_workers=1, prefetch_factor=2,
               early_stop_delta=None, save_fraction=1/2,
               save_epoch='intermediate', save_model='best'):
        """Train the model.

        Args:
            index_train (list(int)): Indices of the training set
            index_valid (list(int)): Indices of the validation set
            index_test  (list(int)): Indices of the testing set
            nepoch (int, optional): numbr of epoch
            train_batch_size (int, optional): size of the batch
            export_intermediate (bool, optional):export itnermediate data
            num_workers (int, optional): number of workers pytorch
                uses to create the batch size
            save_epoch (str,optional): 'intermediate' or 'all'
            save_model (str, optional): 'all' or 'best'

        Returns:
            torch.tensor: Parameters of the network after training
        """

        # printing options
        nprint = np.max([1, int(nepoch / 10)])

        # pin memory for cuda
        self.pin = False
        if self.cuda:
            self.pin = pin_cuda

        # create the sampler
        train_sampler = data_utils.sampler.SubsetRandomSampler(index_train)
        valid_sampler = data_utils.sampler.SubsetRandomSampler(index_valid)
        test_sampler = data_utils.sampler.SubsetRandomSampler(index_test)
        
        # get if we test as well
        self._valid_ = len(valid_sampler.indices) > 0
        self._test_ = len(test_sampler.indices) > 0
        
        # how often to measure the progress as a fraction of an epoch
        self.frac_measure = save_fraction
        self.early_stop_delta = early_stop_delta
        self.batch_size = train_batch_size
        self.train_index = index_train

        # containers for the losses
        self.losses = {'train': []}
        if self._valid_:
            self.losses['valid'] = []
        if self._test_:
            self.losses['test'] = []

        # containers for the class metrics
        if self.save_classmetrics:
            self.classmetrics = {}
            for i in self.metricnames:
                self.classmetrics[i] = {'train': []}
                if self._valid_:
                    self.classmetrics[i]['valid'] = []
                if self._test_:
                    self.classmetrics[i]['test'] = []

        #  create the loaders
        train_loader = data_utils.DataLoader(
            self.data_set,
            batch_size=train_batch_size,
            sampler=train_sampler,
            pin_memory=self.pin,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            shuffle=False,
            drop_last=True,
            persistent_workers=False)
        if self._valid_:
            self.valid_loader = data_utils.DataLoader(
                self.data_set,
                batch_size=train_batch_size,
                sampler=valid_sampler,
                pin_memory=self.pin,
                num_workers=num_workers,
                prefetch_factor=prefetch_factor,
                shuffle=False,
                drop_last=False,
                persistent_workers=False)
        if self._test_:
            test_loader = data_utils.DataLoader(
                self.data_set,
                batch_size=train_batch_size,
                sampler=test_sampler,
                pin_memory=self.pin,
                num_workers=num_workers,
                prefetch_factor=prefetch_factor,
                shuffle=False,
                drop_last=False,
                persistent_workers=False)

        # min error to keep track of the best model.
        self.min_error = {'train': float('Inf'),
                        'valid': float('Inf'),
                        'test': float('Inf')}

        epoch_method = self._epoch
        if self.contrastive:
            epoch_method = self._epoch_contrastive

        # training loop
        av_time = 0.0
        self.exec_epochs = 0
        self.data = {}
        self.best = {}
        for epoch in range(-1, nepoch):

            logger.info(f'\n: epoch {epoch:03d} / {nepoch:03d} {"-"*45}')
            t0 = time.time()

            # epoch -1 is an epoch where there is no training, just initial validation and testing
            if epoch != -1:
                # train the model
                logger.info(f"\n\t=> train the model\n")
                train_loss, self.data['train'] = epoch_method(
                    train_loader, train_model=True)
                self.losses['train'].append(train_loss)
                if self.save_classmetrics:
                    for i in self.metricnames:
                        self.classmetrics[i]['train'].append(self.data['train'][i])

            # validate the model
            if self._valid_:
                logger.info(f"\n\t=> validate the model\n")
                valid_loss, self.data['valid'] = epoch_method(
                    self.valid_loader, train_model=False)
                self.losses['valid'].append(valid_loss)
                if self.save_classmetrics:
                    for i in self.metricnames:
                        self.classmetrics[i]['valid'].append(
                            self.data['valid'][i])

            # test the model
            if self._test_:
                logger.info(f"\n\t=> test the model\n")
                test_loss, self.data['test'] = epoch_method(
                    test_loader, train_model=False)
                self.losses['test'].append(test_loss)
                if self.save_classmetrics:
                    for i in self.metricnames:
                        self.classmetrics[i]['test'].append(
                            self.data['test'][i])

            # print loss
            if epoch != -1:
                logger.info(f'\n  train loss      : {train_loss:1.3e}')
            if self._valid_:
                logger.info(f'  valid loss      : {valid_loss:1.3e}')
            if self._test_:
                logger.info(f'  test loss       : {test_loss:1.3e}')

            # timer
            elapsed = time.time() - t0
            logger.info(
                f'  epoch done in   : {self.convertSeconds2Days(elapsed)}')

            # remaining time
            if epoch != -1:
                av_time += elapsed
                nremain = nepoch - (epoch + 1)
                remaining_time = av_time / (epoch + 1) * nremain
                logger.info(f"  Estimated remaining time  : "
                    f"{time.strftime('%H:%M:%S', time.gmtime(remaining_time))}")

            # save the best model
            for mode in ['train', 'valid', 'test']:
                if mode not in self.losses or not len(self.losses[mode]):
                    continue
                if self.losses[mode][-1] < self.min_error[mode]:
                    self.save_model(
                        filename="best_{}_model.pth.tar".format(mode))
                    self.min_error[mode] = self.losses[mode][-1]
                    self.best[mode] = self.data[mode]

            # save all the model if required
            if save_model == 'all':
                self.save_model(filename="model_epoch_%04d.pth.tar" % epoch)

            # plot and save epoch
            if (export_intermediate and epoch % nprint == nprint - 1) or \
                epoch == 0 or epoch == nepoch - 1:

                if self.plot:
                    figname = os.path.join(self.outdir,
                        f"prediction_{epoch:04d}.png")
                    self._plot(figname)      
                                      
                    for i in self.metricnames:
                        self._plot_metric(i, n_epochs=epoch)
                    
                    self._plot_losses(os.path.join(self.outdir, 'losses.png'), n_epochs=epoch)

                    # if self.task == 'reg':
                    #     self._plot_scatter_reg('scatter.png')              
                            
                if self.save_hitrate:
                    figname = os.path.join(self.outdir,
                        f"hitrate_{epoch:04d}.png")
                    self.plot_hit_rate(figname)

                self._export_epoch_hdf5(epoch, self.data)

            elif save_epoch == 'all':
                self._export_epoch_hdf5(epoch, self.data)

            sys.stdout.flush()
            
            # check if early stop is needed
            if self._early_stop():
                print(f"Early stop at n epochs: {epoch}\n Validation loss has not improved for {self.early_stop_delta} epochs")
                break
            

        # make variable 'exec_epoch', if early stop was executed the x axis (with n epochs) for plots will still be acurate
        self.exec_epochs = epoch+1
        # plot the losses
        
        self._export_losses(os.path.join(self.outdir, 'losses.png'), n_epochs=epoch, plot_best_model=True)
        # plot ROC for best epoch
        index_best = self._get_best_epoch()
        if self.task=='class':
            # plot ROC for best valid
            figname = os.path.join(self.outdir,f"ROC_valid_{index_best:04d}.png")
            self._plot_roc(figname=figname, data=self.best['valid'])
            # plot ROC for test of that best valid epoch
            figname = os.path.join(self.outdir,f"ROC_test_{index_best:04d}.png")
            self._plot_roc(figname=figname, data=self.best['test'])
            # plot ROC for train of that best valid epoch
            figname = os.path.join(self.outdir,f"ROC_train_{index_best:04d}.png")
            self._plot_roc(figname=figname, data=self.best['train'])

        self._export_loss_table(self.exec_epochs)

        # plot classification metrics and export table
        if self.save_classmetrics:
            for i in self.metricnames:
                self._export_metrics(i, n_epochs=epoch, plot_best_model=True)
                self._export_metric_table(i, epoch, self.exec_epochs)

        return torch.cat([param.data.view(-1)
                          for param in self.net.parameters()], 0)
        


    def _epoch(self, data_loader, train_model, has_target=True):
        """Perform one single epoch iteration over a data loader.
        Args:
            data_loader (torch.DataLoader): DataLoader for the epoch
            train_model (bool): train the model if True or not if False
        Returns:
            float: loss of the model
            dict:  data of the epoch
        """

        # variables of the epoch
        running_loss = 0
        data = {'outputs': [], 'targets': [], 'mol': []}
        if self.save_hitrate:
            data['hit'] = None

        if self.save_classmetrics:
            for i in self.metricnames:
                data[i] = None

        n = 0
        debug_time = False
        time_learn = 0

        # set train/eval mode
        self.net.train(mode=train_model)
        torch.set_grad_enabled(train_model)

        total_minibatches = len(data_loader)
        n_mini = 0

        for mini_batch in data_loader:
        
            n_mini += 1
            logger.info(f"\t\t-> mini-batch: {n_mini} ")
            inputs = mini_batch['feature']
            targets = mini_batch['target']
            mol = mini_batch['mol']

            # transform the data
            inputs, targets = self._get_variables(inputs, targets)
            
            # starting time
            tlearn0 = time.time()

            # forward
            outputs = self.net(inputs)

            # class complains about the shape ...
            if self.task == 'class':
                targets = targets.view(-1)

            # check for the case where the network gives a single output
            # convert to dim 2 because the rest of operations expect dim 2
            if self.task == 'class' and outputs.squeeze().shape[-1] != 2:
                output_2dim = torch.zeros((outputs.shape[0], 2))
                output_2dim[:,0] = 1-outputs[:,0]
                output_2dim[:,1] = outputs[:,0]
                outputs = output_2dim
                if self.cuda:
                    outputs = outputs.cuda(non_blocking=self.pin)

            if has_target:
                # evaluate loss
                loss = self.criterion(outputs, targets)
                running_loss += (loss.data.item() * len(inputs))  # pytorch1 compatible
            n += len(inputs)

            # zero + backward + step
            if train_model:          
                self.optimizer.zero_grad()
                loss.backward()
                
                if self.attention and self.exec_epochs/self.nepoch < 0.7: # in the last 20% percent of training..
                    #self.net.attention.grad *= 0.000001 # make gradient really small so no effect of attention yet
                    self.net.attention.grad *= 0.000001 # make gradient really small so no effect of attention yet
                    
                self.optimizer.step()
            time_learn += time.time() - tlearn0
            
            if self.cuda:
                outputs = outputs.data.cpu().numpy()
                targets = targets.data.cpu().numpy()
            else:
                outputs = outputs.data.numpy()
                targets = targets.data.numpy()

            # get the outputs for export
            data['outputs'] += outputs.tolist()
            data['targets'] += targets.tolist()

            fname, molname = mol[0], mol[1]
            data['mol'] += [(f, m) for f, m in zip(fname, molname)]
            
            #  mini_batch%np.ceil(len(self.train_index)/self.batch_size)*self.frac_measure == 0 and 
            # check every few minibatches for overfitting during 'train' phases
            if train_model == True and \
                n_mini in [round(total_minibatches*self.frac_measure) * i for i in range(1, int(total_minibatches/(total_minibatches*self.frac_measure)))] and \
                n_mini != total_minibatches and \
                n_mini != 0: # dont check if this is the first epoch
                
                print(f"Checking validation score after {n_mini} of total {total_minibatches} minibatches")
                if self._valid_:
                    logger.info(f"\n\t=> validate the model\n")
                    valid_loss, self.data['valid'] = self._epoch(
                        self.valid_loader, train_model=False)
                    self.losses['valid'].append(valid_loss)

                # save best model in case it is already overfitting
                mode = 'valid'
                if self.losses[mode][-1] < self.min_error[mode]:
                    self.save_model(
                        filename="best_{}_model.pth.tar".format(mode))
                    self.min_error[mode] = self.losses[mode][-1]

                # finally reset train/eval mode
                self.net.train(mode=train_model)
                torch.set_grad_enabled(train_model)

        #### done with minibatch loop 
        
        if self.data_set.normalize_targets:
            data['outputs'] = self.data_set.backtransform_target(
                np.array(data['outputs']))  # .flatten())
            data['targets'] = self.data_set.backtransform_target(
                np.array(data['targets']))  # .flatten())
        else:
            data['outputs'] = np.array(data['outputs'])  # .flatten()
            data['targets'] = np.array(data['targets'])  # .flatten()

        # make np for export
        data['mol'] = np.array(data['mol'], dtype=object)

        # get the relevance of the ranking
        if self.save_hitrate and has_target:
            logger.info(f'Use hit cutoff {self.hit_cutoff}')
            data['hit'] = self._get_relevance(data, self.hit_cutoff)

        # get classification metrics
        if self.save_classmetrics and has_target:
            for i in self.metricnames:
                try:
                    data[i] = self._get_classmetrics(data, i)
                except Exception as e:
                    print(e)
                    print(f'metric name: {i}')
                    print(f'{data}')
                    

        # normalize the loss
        if n != 0:
            running_loss /= n
        else:
            warnings.warn(f'Empty input in data_loader {data_loader}.')

        return running_loss, data
    
    def _epoch_contrastive(self, data_loader, train_model, has_target=True):
        # variables of the epoch
        running_loss = 0
        cr_running_loss = 0
        data = {'outputs': [], 'targets': [], 'mol': []}
        if self.save_hitrate:
            data['hit'] = None

        if self.save_classmetrics:
            for i in self.metricnames:
                data[i] = None

        debug_time = False
        time_learn = 0

        # set train/eval mode
        self.net.train(mode=train_model)
        torch.set_grad_enabled(train_model)

        total_minibatches = len(data_loader)
        n_mini_batch = 0
        count_items = 0
        cr_count_items = 0
        n_training_samples = 0

        for i_mini, mini_batch in enumerate(data_loader):

            n_mini_batch += 1

            logger.info(f"\t\t-> mini-batch: {n_mini_batch} ")

            # get the data
            inputs = mini_batch['feature']
            targets = mini_batch['target']
            mol = mini_batch['mol']

            # transform the data
            inputs, targets = self._get_variables(inputs, targets)

            # starting time
            tlearn0 = time.time()

            # forward
            outputs = self.net(inputs)
            
            # class complains about the shape ...
            if self.task == 'class':
                targets = targets.view(-1)
            
            if train_model:
                
                try:
                    # Couple random pairs
                    cr_outputs = outputs.reshape(len(inputs)//2, 2)
                    cr_targets = targets.reshape(len(inputs)//2, 2)
                except (ValueError, RuntimeError):
                    # Couple random pairs
                    logger.info('Try to make the shape divisible by 2 for making pairs')
                    cr_outputs = outputs[:-1].reshape(len(inputs)//2, 2)
                    cr_targets = targets[:-1].reshape(len(inputs)//2, 2)
                    cr_inputs = inputs[:-1]
        
                # Remove the pairs that accidentially have the same label
                if self.task == 'class':
                    cr_outputs = cr_outputs[(cr_targets[:,0]!=cr_targets[:,1])]
                    cr_targets = cr_targets[(cr_targets[:,0]!=cr_targets[:,1])]
                elif self.task == 'reg':
                    # for regression make sure the pairs are different enough, labels difference >= 0.05
                    cr_outputs = cr_outputs[(abs(cr_targets[:,0] - cr_targets[:,1]) >= 0.05)]
                    cr_targets = cr_targets[(abs(cr_targets[:,0] - cr_targets[:,1]) >= 0.05)]
                
                if not cr_outputs.size or not cr_targets.size:
                    continue

                # Make predictions relative among pairs
                sm_output = F.softmax(cr_outputs, dim=1)[:,0]
                # Create the correct labels, example: labelpair [9, 1] becomes 1 and labelpair[1, 9] becomes 0
                relat_targets = (cr_targets[:,0] > cr_targets[:,1]).float()
                
                # Calculate loss and backpropagate :)
                loss = F.binary_cross_entropy(sm_output.squeeze(), relat_targets)
                self.optimizer.zero_grad()
                loss.backward()
                # if we have an attention vector 
                if self.attention and self.exec_epochs/self.nepoch < 0.7: # in the first 80% percent of training no finetuning yet
                    self.net.attention.grad *= 0.000001 # make gradient really small so effect of attention is almost zero
                # do back propagation
                self.optimizer.step()

                # calculate running loss
                cr_running_loss += loss.item()*(len(inputs)//2)
                cr_count_items += len(inputs)//2
                
                n_training_samples += len(cr_targets)*2

            # translate to 'normal' loss instead of contrastive loss
            outputs = torch.sigmoid(outputs).squeeze()
            loss = F.binary_cross_entropy(outputs, targets.squeeze().float())
            running_loss += loss.item()*len(inputs)
            count_items += len(inputs)

            # get the outputs for export
            if self.cuda:
                outputs = outputs.data.cpu().numpy()
                targets = targets.data.cpu().numpy()
            else:
                outputs = outputs.data.numpy()
                targets = targets.data.numpy()

            # outputs to a 2-dim shaped output, negatives as 0 index, positive as 1 index
            # this is needed because we interpret the performance based on binary classification with contrastive regression 

            output_2dim = np.zeros((outputs.shape[0], 2))
            output_2dim[:,0] = 1-outputs[:]
            output_2dim[:,1] = outputs[:]
            outputs = torch.Tensor(output_2dim)
            
            # also convert targets to binary
            class_label_cutoff = 1-(np.log10(max(min(500, 50_000), 1))/np.log10(50_000))
            targets = (targets > class_label_cutoff).astype('int32')

            data['outputs'] += outputs.tolist()
            data['targets'] += targets.tolist()

            fname, molname = mol[0], mol[1]
            data['mol'] += [(f, m) for f, m in zip(fname, molname)]
            
        #### done with minibatch loop 

        data['outputs'] = np.array(data['outputs']).squeeze()  # .flatten()
        data['targets'] = np.array(data['targets']).squeeze()  # .flatten()

        # make np for export
        data['mol'] = np.array(data['mol'], dtype=object)

        # get classification metrics
        if self.save_classmetrics and has_target:
            for i in self.metricnames:
                try:
                    data[i] = self._get_classmetrics(data, i)
                except Exception as e:
                    print(e)
                    print(f'metric name: {i}')
                    print(f'{data}')
                    
        # normalize the loss
        if count_items != 0:
            running_loss /= count_items
        else:
            warnings.warn(f'Empty input in data_loader {data_loader}.')
            
        if train_model:
            n_pairs = n_training_samples/2
            logger.info(f"  {n_training_samples} used for training in this epoch ({n_pairs:,} pairs)")

        return running_loss, data
            
            
    def _get_variables(self, inputs, targets):
        # xue: why not put this step to DataSet.py?
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
        if self.cuda:
            inputs = inputs.cuda(non_blocking=self.pin)
            targets = targets.cuda(non_blocking=self.pin)

        # get the variable as float by default
        inputs, targets = Variable(inputs).float(), Variable(targets).float()

        # change the targets to long for classification
        if self.task == 'class':
            targets = targets.long()

        return inputs, targets
    
    def _early_stop(self):
        """check if training should early stop based on progression of validation loss

        Returns:
            bool: true if training should stop else false 
        """
        best_loss = self.min_error['valid']
        
        # if current loss is worse than best loss measured so far
        if self.losses['valid'][-1] > best_loss:
            index_best = np.where(np.array(self.losses['valid']) == best_loss)[0][-1]
            index_current = len(self.losses['valid'])-1
            # loss is measured every frac_measure*epoch and delta loss is expressed in full epochs
            if np.floor((index_current - index_best)*self.frac_measure) >= self.early_stop_delta:
                
                format_index_best = "{:.2f}".format(index_best*self.frac_measure)
                print(f'best model was saved at {format_index_best} epochs')
                return True
            
    def _save_outputs_targets(self, data):
        # transform the output back
        if self.data_set.normalize_targets:
            data['outputs'] = self.data_set.backtransform_target(
                np.array(data['outputs']))  # .flatten())
            data['targets'] = self.data_set.backtransform_target(
                np.array(data['targets']))  # .flatten())
        else:
            data['outputs'] = np.array(data['outputs'])  # .flatten()
            data['targets'] = np.array(data['targets'])  # .flatten()
        return data

    def _export_losses(self, figname, n_epochs, plot_best_model=False):
        """Plot the losses vs the epoch.

        Args:
            figname (str): name of the file where to export the figure
        """
        self._plot_losses(figname, n_epochs, plot_best_model=plot_best_model)

        grp = self.f5.create_group('/losses/')
        grp.attrs['type'] = 'losses'
        for k, v in self.losses.items():
            grp.create_dataset(k, data=v)
    
    def _plot_losses(self, figname, n_epochs, plot_best_model=False):
        logger.info('\n --> Loss Plot')

        color_plot = ['red', 'blue', 'green']
        labels = ['Train', 'Valid', 'Test']
        
        if plot_best_model:
            index_best = ((np.where(np.array(self.losses['valid']) == self.min_error['valid'])[0][-1] \
                           +1)*self.frac_measure)-self.frac_measure

        fig, ax = plt.subplots()
        for ik, name in enumerate(self.losses):
            if n_epochs == -1 or n_epochs == 0:
                n_epochs = len(self.losses[name])

            if n_epochs == -1 or n_epochs == 0:
                if name == 'train':
                    n_epochs = len(np.array(self.losses[name]))
                else:
                    n_epochs = len(np.array(self.losses[name]))+1

            if name == 'train':
                x = np.linspace(0, n_epochs, len(self.losses[name]))
            else:
                x = np.linspace(-1, n_epochs, len(self.losses[name]))
                
            plt.plot(   x,
                        np.array(self.losses[name]),
                        c = color_plot[ik],
                        label = labels[ik])
            plt.scatter(   x,
                        np.array(self.losses[name]),
                        c = color_plot[ik],
                        s=8, marker='D',
                        label='_nolegend_')
            
        if plot_best_model:
            plt.axvline(x = index_best, 
                        color = 'orchid', 
                        ls='--', 
                        label = 'best model')
        legend = ax.legend(loc='upper right')

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Losses')
        ax.set_xticks(np.arange(0, len(x)+1, 1))

        fig.savefig(figname)
        plt.close()
    
    def _export_loss_table(self, n_epochs):

        # x = np.linspace(0, n_epochs, len(self.losses['valid']))
        x_valid = [f'epoch_{round(float(x), 2)}' for x in np.arange(0, n_epochs+self.frac_measure, self.frac_measure)]
        len_array = len(self.losses['valid'])

        data = pd.DataFrame(index=x_valid)

        x_train = [f'epoch_{round(float(x), 2)}' for x in np.arange(1, n_epochs+1, 1)]
        x_test = [f'epoch_{round(float(x), 2)}' for x in np.arange(0, n_epochs+1, 1)]

        data.loc[x_train,'train'] = self.losses['train']
        data.loc[x_valid, 'valid'] = self.losses['valid']
        data.loc[x_test, 'test'] = self.losses['test']

        data.to_csv(os.path.join(self.outdir, 'loss_values' + '.csv'), sep='\t')
        
        
    def _plot_metric(self, metricname, n_epochs=0, plot_best_model=False):
        
        if plot_best_model:
            index_best = ((np.where(np.array(self.losses['valid']) == self.min_error['valid'])[0][-1]+1)\
            *self.frac_measure)-self.frac_measure

        logger.info(f'\n --> {metricname.upper()} Plot')

        color_plot = ['red', 'blue', 'green']
        labels = ['Train', 'Valid', 'Test']

        data = self.classmetrics[metricname]
        fig, ax = plt.subplots()
        for ik, name in enumerate(data):

            if n_epochs == -1 or n_epochs == 0:
                if name == 'train':
                    n_epochs = len(np.array(data[name]))
                else:
                    n_epochs = len(np.array(data[name]))+1

            if name == 'train':
                x = np.linspace(0, n_epochs, len(data[name]))
            else:
                x = np.linspace(-1, n_epochs, len(data[name]))
            plt.plot(x, np.array(data[name]), c=color_plot[ik], 
                    label=labels[ik])
            plt.scatter(x, np.array(data[name]), c=color_plot[ik], 
                        s=8, marker='D',
                        label='_nolegend_')
            
        if plot_best_model:
            plt.axvline(x = index_best, 
                        color = 'orchid', 
                        ls='--', 
                        label = 'best model')

        legend = ax.legend(loc='upper left')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metricname.upper())
        ax.set_xticks(np.arange(0, len(x)+1, 1))

        figname = os.path.join(self.outdir, metricname + '.png')
        fig.savefig(figname)
        plt.close()

    def _export_metrics(self, metricname:str, n_epochs:int, plot_best_model=False):
        """seperate plot for each metric (e.g accuracy/auc) over the epochs

        Args:
            metricname (str): name of the metric to obtain from classmetrics dict and to plot
            exec_epochs (int): number of executed epochs (may have been corrected for early stopping)
        """

        self._plot_metric(metricname, n_epochs=n_epochs, plot_best_model=plot_best_model)

        grp = self.f5.create_group(metricname)
        grp.attrs['type'] = metricname
            
        data = self.classmetrics[metricname]
        for k, v in data.items():
            grp.create_dataset(k, data=v)

    def _export_metric_table(self, metricname:str, epoch, exec_epochs:int):

        # check if dataframe has been initialized already
        if hasattr(self, 'metrics_table_valid'):
            data_valid = self.metrics_table_valid
            data_test = self.metrics_table_test
            data_train = self.metrics_table_train
        else:
            self.metrics_table_valid = data_valid = pd.DataFrame(index=[f'epoch_{i}' for i in list(range(0, exec_epochs+1))],
                                                     columns=self.metricnames)
            self.metrics_table_test = data_test = pd.DataFrame(index= [f'epoch_{i}' for i in list(range(0, exec_epochs+1))], 
                                                     columns=self.metricnames)
            self.metrics_table_train = data_train = pd.DataFrame(index = [f'epoch_{i}' for i in list(range(1, exec_epochs+1))], 
                                                     columns=self.metricnames)
        
        # add the metric to each of the valid, test and train dataframes
        data_valid[metricname] = self.classmetrics[metricname]['valid']
        data_test[metricname] = self.classmetrics[metricname]['test']
        data_train[metricname] = self.classmetrics[metricname]['train']

        if metricname == self.metricnames[-1]:
        # if metricname == self.classmetrics['valid'][-1]:
            data_valid.to_csv(os.path.join(self.outdir, 'valid_metrics' + '.csv'), sep='\t')
            data_test.to_csv(os.path.join(self.outdir, 'test_metrics' + '.csv'), sep='\t')
            data_train.to_csv(os.path.join(self.outdir, 'train_metrics' + '.csv'), sep='\t')
        
        self.metrics_table_test = data_test
        self.metrics_table_valid = data_valid
        self.metrics_table_train = data_train

    def _plot_scatter_reg(self, figname):
        """Plot a scatter plots of predictions VS targets.

        Useful to visualize the performance of the training algorithm

        Args:
            figname (str): filename
        """

        # abort if we don't want to plot
        if self.plot is False:
            return

        logger.info(f'\n  --> Scatter Plot: {figname}')

        color_plot = {'train': '#bc5090', 'valid': 'blue', 'test': 'green'}
        labels = ['train', 'valid', 'test']

        fig, ax = plt.subplots()

        xvalues = np.array([])
        yvalues = np.array([])

        for l in labels:

            if l in self.data:
                try:
                    targ = self.data[l]['targets'].flatten()
                except Exception:
                    logger.exception(
                        f'No target values are provided for the {l} set, ',
                        f'skip {l} in the scatter plot')
                    continue

                out = self.data[l]['outputs'].flatten()

                xvalues = np.append(xvalues, targ)
                yvalues = np.append(yvalues, out)

                ax.scatter(targ, out, c=color_plot[l], label=l)

        legend = ax.legend(loc='upper left')
        ax.set_xlabel('Targets')
        ax.set_ylabel('Predictions')

        # plot bisector
        #values = np.append(xvalues, yvalues)
        #border = 0.1 * (values.max() - values.min())
        ax.plot([0, 1],
                [0, 1])

        fig.savefig(figname)
        plt.close()

    def _plot_scatter_hist_reg(self, figname):
        
        labels = ['train', 'valid', 'test']
        
        if self.plot is False:
            return
        #def scatter_hist(x, y, hist_step, ticks_step, max_lim, title, xlabel, ylabel, bins=50, savefile=False):
        hist_step = 1000
        xticks_step = 5
        yticks_step = 5
        max_xlim = 1.0
        max_ylim = 1.0
        title = 'Target vs Prediction'
        xlabel = 'Target'
        ylabel = 'Prediction'
        
        logger.info(f'\n  --> Scatter Plot: {figname}')

        color_plot = {'train': '#bc5090', 'valid': 'blue', 'test': 'green'}
        
        
        # definitions for the axes
        left, width = 0.1, 0.65
        bottom, height = 0.1, 0.65
        spacing = 0.02
        
        rect_scatter = [left, bottom, width, height]
        rect_histx = [left, bottom + height + spacing, width, 0.2]
        rect_histy = [left + width + spacing, bottom, 0.2, height]
        
        for l in labels:
            if l in self.data:
                fn = figname.split('/')[:-1]
                fn.append(f'{l}_{figname.split("/")[-1]}')
                figname_l = ('/').join(fn)
                # start with a square Figure
                fig = plt.figure(figsize=(10, 10))
                
                fig.suptitle(title, fontsize=20, fontweight='bold', y=1.01)
                # Add labels    
                #fig.xlabel=xlabel
                #fig.ylabel=ylabel
                
                ax = fig.add_axes(rect_scatter)
                ax_histx = fig.add_axes(rect_histx, sharex=ax)
                ax_histy = fig.add_axes(rect_histy, sharey=ax)
                # no labels
                ax_histx.tick_params(axis="x", labelbottom=False)
                ax_histy.tick_params(axis="y", labelleft=False)

                # The data:
                xvalues = np.array([])
                yvalues = np.array([])
                try:
                    targ = self.data[l]['targets'].flatten()
                except Exception:
                    logger.exception(
                        f'No target values are provided for the {l} set, ',
                        f'skip {l} scatter/hist plot {figname}')
                    continue

                out = self.data[l]['outputs'].flatten()

                # X mean line
                ax.axvline(mean(targ), color='k', ls=':', lw=2.5)
                
                # X mean line
                ax.axhline(mean(out), color='k', ls=':', lw=2.5)

                #xvalues = np.append(xvalues, targ)
                #yvalues = np.append(yvalues, out)

                # the scatter plot:
                ax.scatter(targ, out, c=color_plot[l], label=l, alpha=0.5)

                ax.grid(which='major', axis='both')
                ax.set_xlim(0, max_xlim)
                ax.set_ylim(0, max_ylim)
                
                ax.set_xticks(list(np.linspace(0., max_xlim, xticks_step)))
                ax.set_xticklabels([f'{x:.1f}' for x in list(np.linspace(0., max_xlim, xticks_step))], fontsize=16, rotation=45)#, weight='bold')
                ax.set_yticks(list(np.linspace(0., max_ylim, yticks_step)))
                ax.set_yticklabels([f'{x:.1f}' for x in list(np.linspace(0., max_ylim, yticks_step))], fontsize=16, rotation=45)#, weight='bold')
                
                
                ax.set_xlabel(xlabel, fontweight='bold', size=18)
                ax.set_ylabel(ylabel, fontweight='bold', size=18)
                
                #ax.title.set_text(title)
                
                # Side histograms
                bins = max([int(len(targ)/100), 20])
                histx = ax_histx.hist(targ, bins=bins, color='#7BA4B8', rwidth=0.8)
                histy = ax_histy.hist(out, bins=bins, color='#7BA4B8', rwidth=0.8,  orientation='horizontal')
                
                #return histx, histy
                hist_max = max((max(histx[0]), max(histy[0])))
                hist_max += hist_max/10
                
                ax_histx.set_ylim(0, hist_max)
                ax_histy.set_xlim(0, hist_max)
                
                ax_histx.set_yticks(list(np.arange(0., hist_max, hist_step)))
                #ax_histx.set_yticklabels(list(range(0, int(hist_max), hist_step)), fontname='Serif')
                ax_histy.set_xticks(list(np.arange(0., hist_max, hist_step)))
                #ax_histy.set_xticklabels(list(range(0, int(hist_max), hist_step)), fontname='Serif')
                
                top_right_side = ax_histx.spines["right"]
                top_right_side.set_visible(False)
                top_top_side = ax_histx.spines["top"]
                top_top_side.set_visible(False)
                
                right_right_side = ax_histy.spines["right"]
                right_right_side.set_visible(False)
                right_top_side = ax_histy.spines["top"]
                right_top_side.set_visible(False)

                # Mean dotted lines
                ax_histx.axvline(mean(targ), color='k', ls=':', lw=2.5)
                ax_histy.axhline(mean(out), color='k', ls=':', lw=2.5)
                
                # Bisector
                line = mlines.Line2D([0, 1], [0, 1], color='k', ls='--')
                transform = ax.transAxes
                line.set_transform(transform)
                ax.add_line(line)

                fig.savefig(figname_l)
                plt.close()

    def _get_best_epoch(self):

        best_loss = self.min_error['valid']
        index_best = np.where(np.array(self.losses['valid']) == best_loss)[0][-1]
     
        epoch_best_flt = index_best*self.frac_measure
        if epoch_best_flt.is_integer():
            epoch_best = int(epoch_best_flt)
        else:
            epoch_best = np.rint(epoch_best_flt)

        return epoch_best
    
    def _plot_roc(self, figname, data):

        # if self.task == 'class' or self.contrastive:
        #     pred = self._get_binclass_prediction(data)
        # elif self.task == 'reg':
        pred = data['outputs']
        
        if self.task == 'class':
            pred = pred[:, 1]
            
        targets = data['targets']

        auc_score = classMetrics.roc_auc(pred, targets)
        tprs, fprs = classMetrics.tpr_fpr_thresholds(pred, targets)
            
        fig, ax = plt.subplots()
        fprs = [round(score, 3) for score in fprs]
        tprs = [round(score, 3) for score in tprs]
        ax.plot(fprs, tprs, linestyle='--', color='darkorange', lw = 2, label='ROC curve', clip_on=False)
        ax.plot([0, 1], [0, 1], color='navy', linestyle='--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.0])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC curve, AUC = %.2f'%auc_score)
        ax.legend(loc="lower right")

        fig.savefig(figname, bbox_inches='tight')
        plt.close()
            
    def _plot_boxplot_class(self, figname):
        """Plot a boxplot of predictions VS targets.

        It is only usefull in classification tasks.

        Args:
            figname (str): filename
        """

        # abort if we don't want to plot
        if not self.plot:
            return

        logger.info(f'\n  --> Box Plot: {figname}')

        color_plot = {'train': 'red', 'valid': 'blue', 'test': 'green'}
        labels = ['train', 'valid', 'test']

        nwin = len(self.data)

        fig, ax = plt.subplots(1, nwin, sharey=True, squeeze=False)

        iwin = 0
        for l in labels:

            if l in self.data:
                try:
                    tar = self.data[l]['targets']
                except Exception:
                    logger.exception(
                        f'No target values are provided for the {l} set, ',
                        f'skip {l} in the boxplot')
                    continue

                out = self.data[l]['outputs']
                data = [[], []]
                confusion = [[0, 0], [0, 0]]

                for pts, t in zip(out, tar):
                    r = F.softmax(torch.FloatTensor(pts), dim=0).data.numpy()
                    data[t].append(r[1])
                    confusion[t][bool(r[1] > 0.5)] += 1

                #print("  {:5s}: {:s}".format(l,str(confusion)))

                ax[0, iwin].boxplot(data)
                ax[0, iwin].set_xlabel(l)
                ax[0, iwin].set_xticklabels(['0', '1'])
                iwin += 1

        fig.savefig(figname, bbox_inches='tight')
        plt.close()

    def plot_hit_rate(self, figname):
        """Plot the hit rate of the different training/valid/test sets.

        The hit rate is defined as:
            The percentage of positive(near-native) decoys that are
            included among the top m decoys.

        Args:
            figname (str): filename for the plot
        """
        if self.plot is False:
            return

        logger.info(f'\n  --> Hitrate plot: {figname}\n')

        color_plot = {'train': 'red', 'valid': 'blue', 'test': 'green'}
        labels = ['train', 'valid', 'test']

        fig, ax = plt.subplots()
        for l in labels:
            if l in self.data:
                try:
                    hits = self.data[l]['hit']
                except Exception:
                    logger.exception(f'No hitrate computed for the {l} set.')
                    continue

                if 'hit' in self.data[l]:
                    hitrate = rankingMetrics.hitrate(hits)
                    m = len(hitrate)
                    x = np.linspace(0, 100, m)
                    plt.plot(x, hitrate, c=color_plot[l], label=f"{l} M={m}")
        legend = ax.legend(loc='upper left')
        ax.set_xlabel('Top M (%)')
        ax.set_ylabel('Hit Rate')

        fmt = '%.0f%%'
        xticks = mtick.FormatStrFormatter(fmt)
        ax.xaxis.set_major_formatter(xticks)

        fig.savefig(figname)
        plt.close()

    def _compute_hitrate(self, hit_cutoff=None):

        # define the target value threshold to compute the hits if save_hitrate is True
        if hit_cutoff is None:
            hit_cutoff = self.hit_cutoff
            logger.info(f'Use hit cutoff {self.hit_cutoff}')

        labels = ['train', 'valid', 'test']
        self.hitrate = {}

        # get the target ordering
        inverse = self.data_set.target_ordering == 'lower'
        if self.task == 'class':
            inverse = False

        for l in labels:

            if l in self.data:

                # get the target values
                out = self.data[l]['outputs']

                # get the target vaues
                targets = []
                try:
                    for fname, mol in self.data[l]['mol']:
                        f5 = h5py.File(fname, 'r')
                        targets.append(f5[mol + f'/targets/{self.data_set.select_target}'][()])
                        f5.close()
                except Exception:
                    logger.exception(
                        f'No target value {self.data_set.select_target} ',
                        f'provided for the {l} set. Skip Hitrate computation.')
                    continue

                # sort the data
                if self.task == 'class':
                    out = F.softmax(torch.FloatTensor(out),
                                    dim=1).data.numpy()[:, 1]
                ind_sort = np.argsort(out)

                if not inverse:
                    ind_sort = ind_sort[::-1]

                # get the targets of the recommendation
                targets = np.array(targets)[ind_sort]

                # make a binary list out of that
                binary_recomendation = (targets <= hit_cutoff).astype('int')

                # number of recommended hit
                npos = np.sum(binary_recomendation)
                if npos == 0:
                    npos = len(targets)
                    warnings.warn(
                        f'Non positive decoys found in {l} for hitrate plot')

    def _get_relevance(self, data, hit_cutoff=None):

        # define the target value threshold to compute the hits if save_hitrate is True
        if hit_cutoff is None:
            hit_cutoff = self.hit_cutoff
            logger.info(f'Use hit cutoff {self.hit_cutoff}')

        if hit_cutoff is not None:
            # get the target ordering
            inverse = self.data_set.target_ordering == 'lower'
            if self.task == 'class':
                inverse = False

            # get the target values
            out = data['outputs']

            # get the targets
            targets = []
            for fname, mol in data['mol']:

                f5 = h5py.File(fname, 'r')
                targets.append(f5[mol + f'/targets/{self.data_set.select_target}'][()])
                f5.close()

            # sort the data
            if self.task == 'class':
                out = F.softmax(torch.FloatTensor(out), dim=1).data.numpy()[:, 1]
            ind_sort = np.argsort(out)

            if not inverse:
                ind_sort = ind_sort[::-1]

            # get the targets of the recommendation
            targets = np.array(targets)[ind_sort]

            # make a binary list out of that
            return (targets <= hit_cutoff).astype('int')
        else:
            return (targets == None).astype('int')
    
    def _get_classmetrics(self, data, metricname):

        # get predctions
        if self.task == 'class' or self.contrastive:
            pred = self._get_binclass_prediction(data)
        elif self.task == 'reg':
            pred = data['outputs']
        # get real targets
        targets = data['targets']

        # get metric values
        if metricname == 'acc':
            return classMetrics.accuracy(pred, targets)
        elif metricname == 'tpr':
            return classMetrics.sensitivity(pred, targets)
        elif metricname == 'tnr':
            return classMetrics.specificity(pred, targets)
        elif metricname == 'ppv':
            return classMetrics.precision(pred, targets)
        elif metricname == 'f1':
            return classMetrics.F1(pred, targets)
        elif metricname == 'mcc':
            return classMetrics.mcc(pred, targets)
        elif metricname == 'auc':
            pred = data['outputs'][:,1] if self.task == 'class' else data['outputs']
            return classMetrics.roc_auc(pred, targets)
        elif metricname == 'rmse':
            return classMetrics.rmse(pred, targets)
        else:
            return None

    @staticmethod
    def _get_binclass_prediction(data):

        out = data['outputs']
        probility = F.softmax(torch.FloatTensor(out), dim=1).data.numpy()
        pred = probility[:, 0] <= probility[:, 1]
        return pred.astype(int)

    def _export_epoch_hdf5(self, epoch, data, include=['train', 'test', 'valid']):
        """Export the epoch data to the hdf5 file.

        Export the data of a given epoch in train/valid/test group.
        In each group are stored the predcited values (outputs),
        ground truth (targets) and molecule name (mol).

        Args:
            epoch (int): index of the epoch
            data (dict): data of the epoch
        """

        # create a group
        grp_name = 'epoch_%04d' % epoch
        grp = self.f5.create_group(grp_name)

        # create attribute for DeepXplroer
        grp.attrs['type'] = 'epoch'
        grp.attrs['task'] = self.task

        # loop over the pass_type: train/valid/test
        for pass_type, pass_data in data.items():

            # we don't want to breack the process in case of issue
            try:

                # create subgroup for the pass
                sg = grp.create_group(pass_type)

                # loop over the data: target/output/molname
                try:
                    for data_name, data_value in pass_data.items():

                        # mol name is a bit different
                        # since there are strings
                        if data_name == 'mol':
                            string_dt = h5py.special_dtype(vlen=str)
                            sg.create_dataset(
                                data_name, data=data_value, dtype=string_dt)

                        # output/target values
                        else:
                            sg.create_dataset(data_name, data=data_value)
                except AttributeError as ae:
                    print(f'skip {pass_type}, no iterable items')

            except TypeError:
                logger.exception("Error in export epoch to hdf5")

        self.f5.close()
        self.f5 = h5py.File(self.fname, 'a')
