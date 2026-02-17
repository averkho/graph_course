# -*- coding: utf-8 -*-
"""
Created on Mon Feb 16 13:39:12 2026

@author: Verkholomov
"""

import torch 
import torch.nn as nn
import numpy as np

class Generator(nn.Module):
    
    def __init__(self, H_inputs, H, z_dim, N, rw_len, temp):
        '''
        

        Parameters
        ----------
        H_inputs : int
            input dimension.
        H : int
            hidden dimension.
        z_dim : int
            latent dimension.
        N : int
            number of nodes (needed for up adn down projection).
        rw_len : int
            number of LSTM cells.
        temp : float
            temperature for gumbel softmax.

        Returns
        -------
        None.

        '''
        
        super(Generator, self).__init__()
        self.intermediate = nn.Linear(z_dim, H).type(torch.float64)
        torch.nn.init.xavier_uniform_(self.intermediate.weight)
        torch.nn.init.zeros_(self.intermediate.bias)
        self.c_up = nn.Linear(H,H).type(torch.float64)
        torch.nn.init.xavier_uniform_(self.c_up.weight)
        torch.nn.init.zeros_(self.c_up.bias)
        self.h_up = nn.Linear(H,H).type(torch.float64)
        torch.nn.init.xavier_uniform_(self.h_up.weight)
        torch.nn.init.zeros_(self.h_up.bias)
        self.lstmcell = LSTMCell(H_inputs, H).type(torch.float64)
        
        self.W_up = nn.Linear(H,N).type(torch.flot64)
        self.W_down = nn.Linear(N, H_inputs).type(torch.float64)
        self.rw_len = rw_len
        self.temp = temp
        self.H = H
        self.latent_dim = z_dim
        self.N = N
        self.H_inputs = H_inputs
        
    def forward(self, latent, inputs, device: str = 'cuda') -> torch.Tensor:
        intermediate = torch.tanh(self.intermediate(latent))
        hc = (torch.tanh(self.h_up(intermediate)), torch.tanh(self.c_up(intermediate)))
        out = []
        for i in range(self.rw_len):
            hh, cc = self.lstmcell(inputs, hc)
            hc = (hh, cc)
            h_up = self.W_up(hh)
            h_sample = self.gumbel_softmax_sample(h_up, self.temp, device)
            inputs = self.W_down(h_sample)
            out.append(h_sample)
        return torch.stack(out, dim=1)
    
    
    def sample_latent(self, num_samples, device) -> torch.Tensor:
        return torch.randn(num_samples, self.latent_dim).type(torch.float64).to(device)
    
    def sample(self, num_samples, device) -> torch.Tensor:
        noise = self.sample_latent(num_samples, device)
        input_zeros = self.init_hidden(num_samples).contiguous().type(torch.float64).to(device)
        generated_data = self(noise, input_zeros, device)
        return generated_data
    
    def sample_discrete(self, num_samples, device) -> np.array:
        with torch.no_grad():
            proba = self.sample(num_samples, device)
        return np.argmax(proba.cpu().numpy(), axis=2) 
    
    def sample_gumbel(self, logits, eps=1e-20) -> torch.Tensor:
        U = torch.rand(logits.shape, dtype=torch.float64)
        return -torch.log(-torch.log(U + eps) + eps)
    
    def gumbel_softmax_sample(self, logits, temperature, device, hard=True) -> torch.Tensor:
        gumbel = self.sample_gumbel(logits).type(torch.float64).to(device)
        y = logits + gumbel
        y = torch.nn.functional.softmax(y / temperature, dim=1)
        if hard:
            y_hard = torch.max(y, 1, keepdim=True)[0].eq(y).type(torch.float64).to(device)
            y = (y_hard - y).detach() + y
        
        return y
    
    def init_hidden(self, batch_size) -> torch.Tensor:
        weight = next(self.parameters()).data
        return weight.new(batch_size, self.H_inputs).zero_().type(torch.float64)
            
            
            
            
            
            
            
            
            
            
            
            
            
            
        
        
        
        
        
        
        
        
        
        