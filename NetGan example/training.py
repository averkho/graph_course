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
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_iterations = max_iterations
        self.rw_len = rw_len
        self.batch_size = batch_size
        self.N = N
        self.generator = Generator(H_inputs=H_inp, H=H_gen, z_dim=z_dim, N=N, rw_len=rw_len, temp=temp_start).to(self.device)
        self.discriminator = Discriminator(H_inputs=H_inp, H=H_gen, N=N, rw_len=rw_len).to(self.device)
        self.G_optimizer = optim.Adam(self.generator.parameters(), lr=lr, betas=betas)
        self.D_optimizer = optim.Adam(self.discriminator.parameters(), lr=lr, betas=betas)
        self.n_critic = n_critic
        self.gp_weight = gp_weight
        self.l2_penalty_disc = l2_penalty_disc
        self.l2_penalty_gen = l2_penalty_gen
        self.temp_start = temp_start
        self.temp_decay = temp_decay
        self.min_temp = min_temp
        
        self.graph = graph
        self.train_ones, self.val_onces, self.val_zeros, self.test_ones, self.test_zeros = utils.train_val_test_split_adjacency(graph, val_share, test_share, seed, undirected=True, connected=True, asserts=True, set_ops=set_ops)
        self.train_graph = sp.coo_matrix((np.ones(len(self.train_ones)), 
                                          (self.train_ones[:, 0], self.train_ones[:, 1]))).tocsr()
        assert(self.train_graph.toarray() == self.train_graph.toarray().T).all()
        self.walker = utils.RandomWalker(self.train_graph, rw_len, p=1, q=1, batch_size=batch_size)
        self.eo = []
        self.critic_loss = []
        self.generator_loss = []
        self.avp = []
        self.roc_auc = []
        self.best_performance = 0.0
        self.running = True
        
    def l2_regularization_G(self, G):
        l2_1 = torch.sum(torch.cat([x.view(-1) for x in G.W_down.weight]) **2 / 2)
        l2_2 = torch.sum(torch.cat([x.view(-1) for x in G.W_up.weight]) **2 / 2)
        l2_3 = torch.sum(torch.cat([x.view(-1) for x in G.W_up.bias]) **2 / 2)
        l2_4 = torch.sum(torch.cat([x.view(-1) for x in G.intermediate.weight]) **2 / 2)
        l2_5 = torch.sum(torch.cat([x.view(-1) for x in G.intermediate.bias]) **2 / 2)
        l2_6 = torch.sum(torch.cat([x.view(-1) for x in G.h_up.weight]) **2 / 2)
        l2_7 = torch.sum(torch.cat([x.view(-1) for x in G.h_up.bias]) **2 / 2)
        l2_8 = torch.sum(torch.cat([x.view(-1) for x in G.c_up.weight]) **2 / 2)
        l2_9 = torch.sum(torch.cat([x.view(-1) for x in G.c_up.bias]) **2 / 2)
        l2_10 = torch.sum(torch.cat([x.view(-1) for x in G.lstmcell.cell.weight]) **2 / 2)
        l2_11 = torch.sum(torch.cat([x.view(-1) for x in G.lstmcell.cell.bias]) **2 / 2)
        
        l2 = self.l2_penalty_gen * (l2_1 + l2_2 + l2_3 + l2_4 + l2_5 + l2_6 + l2_7 + l2_8 + l2_9 + l2_10 
                                    + l2_11)
        
        return l2
        
    def l2_regularization_D(self, D):
        
        l2_1 = torch.sum(torch.cat([x.view(-1) for x in D.W_down.weight]) **2 / 2)
        l2_2 = torch.sum(torch.cat([x.view(-1) for x in D.lstmcell.cell.weight]) **2 / 2)
        l2_3 = torch.sum(torch.cat([x.view(-1) for x in D.lstmcell.cell.bias]) **2 / 2)
        l2_4 = torch.sum(torch.cat([x.view(-1) for x in D.lin_out.weight]) **2 / 2)
        l2_5 = torch.sum(torch.cat([x.view(-1) for x in D.lstmcell.cell.bias]) **2 / 2)
        l2 = self.l2_penalty_disc * (l2_1 + l2_2 + l2_3 + l2_4 + l2_5)
        
        return l2
    
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        