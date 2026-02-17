# -*- coding: utf-8 -*-
"""
Created on Tue Feb 17 15:24:52 2026

@author: Verkholomov
"""

from models import Generator, Discriminator
import utils

import numpy as np
import scipy.sparse as sp
from sklearn.metrics import roc_auc_score, average_precision_score
import math

import torch
import torch.optim as optim
from torch.nn.functional import one_hot
from torch.autograd import grad
import time
from joblib import Parallel, delayed
import pbd
from matplotib import pyplot as plt
from typing import Tuple

class Trainer():
    def __init__(self, graph: sp, N: int, max_iterations: int = 20_000, rw_len: int = 16, batch_size: int = 128, H_gen: int = 40,
                 H_disc: int = 20, H_inp: int = 128, z_dim: int = 16, lr: float = 0.0003, n_critic: int = 3,
                 gp_weight: float = 10.0, betas: Tuple = (0.5, 0.9), l2_penalty_disc: float = 5e-5, l2_penalty_gen: float = 1e-7,
                 temp_start: float = 5.0, temp_decay: float = 1-5e-5, min_temp: float = 0.5, val_share: float = 0.1,
                 test_share: float = 0.05, seed: int = 498567, set_ops: bool = False):
        
        """
        graph: scypy_sparse_matrix
                Graph
                
        N: int
            number of nodes in graph to generate
        max_iterations: int
            Maximal iterations if the stopping criteria is not fulfilled
        rw_len: int
            Length of random walks to generate
        batch_size: int
            the batch size
        H_gen: int
            The hidden size of the generator
        H_disc: int
            The hidden size of the discriminator
        H_inp: int
            Inputsize of the LSTM-cells
        z_dim: int
            The dimension of the random noise that is used as input to the generator
        lr: float
            Learning rate that will be used both for generator and discriminator
        n_critics: int
            The number of discriminator iterations per generator training iteration
        gp_weight: float
            Gradient penalty weight for the Wasserstein GAN
        betas: Tuple
            Decay rates for Adam optimizers
        l2_penalty_gen: float
            L2 penalty on the genrator weight
        l2_penalty_disc : float
            L2 penalty on the discriminator weight
        temp_start: float
            The initial temperature for the Gumbel softmax
        temp_decay: float
            After each evaluation, the current temperature is updated as 
            current_temp: = max(temperature_decay*current_temp, min_temperature)
        min_temp: float
            The minimal temperature for the Gumbel softmax
        val_share: float
            Percentage of validation edges
        test_share: float
            Percentage of test edges
        seed: int
            Seed for numpy.random. It is used for splitting the graph in train, validation and test set
        """
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        