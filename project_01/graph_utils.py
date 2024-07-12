import networkx as nx
import numpy as np

my_red = "#f53b3b" # red
my_blue = "#4287f5" # blue


def create_square_lattice(N):
    G = nx.grid_2d_graph(N, N)
    return G

def get_dense_adj_matrix(G):
    adj_matrix = nx.adjacency_matrix(G).todense()
    return np.array(adj_matrix)

def state_colors(state):
    colors = np.where(state == 1, my_red, my_blue)
    return colors.tolist()