from training import Trainer
import matplotlib.pyplot as plt
import networkx as nx
import torch
import pandas as pd
import numpy as np
import scipy
from utils import graph_from_scores

graph = scipy.io.mmread('inf-USAir97.mtx')
graph = nx.from_scipy_sparse_array(graph)
graph = nx.to_numpy_array(graph)
graph[graph!=0] = 1.0
graph_nx = nx.from_numpy_array(graph)
graph_sparse = scipy.sparse.csr_matrix(graph)
n_edges = graph.sum()

nx.draw(graph_nx, node_size=25, alpha=0.5)

trainer = Trainer(graph_sparse, len(graph), max_iterations=20000, rw_len=12, batch_size=128, H_gen=40, H_disc=30, H_inp=128, z_dim=16, lr=0.0003,
                  n_critic=3, gp_weight=10.0, betas=(.5, .9), l2_penalty_disc=5e-5, l2_penalty_gen=1e-7, temp_start=5.0,  
                  val_share=0.2, test_share=0.1, seed=20, set_ops=False)

trainer.train(create_graph_every=100, plot_graph_every=200, num_samples_graph=50000, stopping_criterion='val')