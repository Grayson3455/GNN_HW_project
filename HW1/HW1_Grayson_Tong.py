"""
ACMS 80770-03: Deep Learning with Graphs
Instructor: Navid Shervani-Tabar
Fall 2022
University of Notre Dame
Homework 1: Programming assignment
"""

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from networkx.algorithms import bipartite
from networkx.generators.random_graphs import erdos_renyi_graph
import copy


# -- Initialize graphs
seed = 30
G = nx.florentine_families_graph()
nodes = G.nodes()

layout = nx.spring_layout(G, seed=seed)

# -- compute jaccard's similarity
"""
    This example is using NetwrokX's native implementation to compute similarities.
    Write a code to compute Jaccard's similarity and replace with this function.
"""

#--------------------------------------------------------------------------------------------
# input: 
#       G: graph
# Out put:
#       S: jaccard local overlap similarity measure
def S_jaccard(G):

    # sub-f for jaccard score
    # input:
    #       i,j: node labels
    #       G:   the graph object
    #       N_list: listed node attributes
    # output:
    #       Jaccard similarity score
    def Score(i,j,G,N_list):
        # --common neighbors(CN)--#
        # method 1: by default function
        #CN = sorted(nx.common_neighbors(G, N_list[i] , N_list[j]))
        #return len(CN)/(G.degree(N_list[i]) + G.degree(N_list[j]) - len(CN))
        # method 2: by adjacency matrix
        A = (nx.adjacency_matrix(G)).todense()
        CN = (A@A)[i,j]
        return CN/(G.degree(N_list[i]) + G.degree(N_list[j]) - CN)

    # find the node array
    nodes = G.nodes()
    # node length
    N     = len(nodes)
    nodes_list = list(nodes)
    return ((nodes_list[i], nodes_list[j],Score(i,j,G,nodes_list)) for i in range(N) for j in range(i+1,N))

pred = S_jaccard(G)
#pred = nx.jaccard_coefficient(G)
#--------------------------------------------------------------------------------------------



# -- keep a copy of edges in the graph
old_edges = copy.deepcopy(G.edges())

# -- add new edges representing similarities.
new_edges, metric = [], []
for u, v, p in pred:
    G.add_edge(u, v)
    print(f"({u}, {v}) -> {p:.8f}")
    new_edges.append((u, v))
    metric.append(p)

# -- plot Florentine Families graph
nx.draw_networkx_nodes(G, nodelist=nodes, label=nodes, pos=layout, node_size=600)
nx.draw_networkx_edges(G, edgelist=old_edges, pos=layout, edge_color='gray', width=4)

# -- plot edges representing similarity
"""
    This example is randomly plotting similarities between 8 pairs of nodes in the graph. 
    Identify the ”Ginori”
"""
ne = nx.draw_networkx_edges(G, edgelist=new_edges[:8], pos=layout, edge_color=np.asarray(metric[:8]), width=4, alpha=0.7)
plt.colorbar(ne)
plt.axis('off')
plt.show()