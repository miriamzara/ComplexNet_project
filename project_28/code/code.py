import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import random
import matplotlib.animation as animation
import imageio_ffmpeg
import statsmodels.api as sm # OLS module for linear regression

spins = [-1, + 1]


def different_states_indicator(a, b):
    value = (1 - a * b)/2
    return value # 1 = different states, 0 = same state
    
def rho(G, state):
    count = 0
    for edge in list(G.edges()):
        start_node = edge[0]
        end_node = edge[1]
        count += different_states_indicator(state[start_node], state[end_node])
    return count/G.size()

def initial_state(num_nodes):
    states = []
    for node in range(num_nodes):
        states.append(random.choice(spins))
    return states

# ASYNCRONOUS UPDATE: one time step corresponds to 
# updating a number of nodes equal to the network size,
#  N, so that on average (!) every node is updated once
def evolution_step(G, state):
    N = G.number_of_nodes()
    for n in range(N):
        selected_node = random.choice(range(N))
        if G.degree(selected_node) > 0:
            selected_neighbour = random.choice(list(G.neighbors(selected_node)))
            state[selected_node] = state[selected_neighbour]
    return state

