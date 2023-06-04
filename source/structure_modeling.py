from sklearn.neighbors import kneighbors_graph
import networkx as nx
import numpy as np
import torch
from source.utils import labels_to_numbers
from torch_geometric.utils.convert import to_networkx, from_networkx



def adjacencyMatrixUsingMnearestNeighbors(X, M = 5):
    
#     knn = NearestNeighbors(n_neighbors=M)
#     knn.fit(X)
    #dist_indx_arr = knn.kneighbors(X, return_distance=True, n_neighbors = M)
    
    A = kneighbors_graph(X, n_neighbors=(M-1), p=2, mode='connectivity', include_self=False) 
    A = A.toarray() # for some reason, is not symmetric
    A = np.maximum( A, A.T )
    
    return A#, dist_indx_arr # dist_indx_arr will be useful while connecting the graph


# Can be used if needed - maybe the graph does not need to be connected?
#
# Function implements trivial method of connecting the graph
#
# A - adjacency matrix
# returns: adjacency matrix with additional connections

def graphUsingMnearestNeighbors(X, M=5):
    return nx.from_numpy_array(adjacencyMatrixUsingMnearestNeighbors(X, M))


def connectTheGraph(A):
    # trivial case of adding an edge:
    W = A
    G = nx.from_numpy_array(W)
    graphs = list(nx.connected_components(G))
    for i in range(len(graphs)-1):

        for g in graphs[i]:
            for f in graphs[i+1]:
                W[f,g] = 1
                W[g,f] = 1
                break
            break
    return W

def torch_geometric_data_from_graph(G, df, labels_nr, train_len, val_len, test_len):
    data=from_networkx(G)
    data.x=torch.from_numpy(np.asmatrix(df)).float()

    data.y=torch.from_numpy(labels_nr)
    data.num_classes = len(np.unique(data.y))

    idx = [i for i in range(df.shape[0])]
    
    train_mask = torch.full_like(data.y, False, dtype=bool)
    train_mask[idx[:train_len]] = True
    data.train_mask=train_mask

    valid_mask = torch.full_like(data.y, False, dtype=bool)
    valid_mask[idx[train_len:train_len+val_len]] = True
    data.valid_mask=valid_mask

    test_mask = torch.full_like(data.y, False, dtype=bool)
    test_mask[idx[train_len+val_len:train_len+val_len+test_len]] = True
    data.test_mask=test_mask
    return data



# Function computes the Laplacian matrix based on the given graph representation
#
# A - connected graph represented by adjacency matrix
# returns: Laplacian matrix
def graphLaplacian(A):
    D = np.eye(A.shape[0]) * A.sum(axis=0)  # diagonal matrix of degrees
    return D  - A                           # L = D - A
    
    
# Function calculates eigenvalues and eigenvectors of Laplacian matrix
#
# L - Laplacian matrix representation
# returns: eigenvalues and eigenvectors
def calculateEigenVectorsOfGraphLaplacian(L):
    eigenValues, eigenVectors = np.linalg.eig(L)
    eigenValues, eigenVectors = eigenValues.real, eigenVectors.real
    
    idx = eigenValues.argsort()[::1]   
    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:, idx]
    
    return eigenValues, eigenVectors
    
    
# Function prepares node representation
#
# eigenVectors - eigenvectors of Laplacian matrix
# nodeRepresentationDim - dimension of representation
# n_of_components=1 - number of connected components in the graph
# returns: node representation
def nodeRepresentation(eigenVectors, nodeRepresentationDim, n_of_components=1):
    m = n_of_components                                         # we assume n of connected components = 1
    Z = eigenVectors[:,m:(nodeRepresentationDim + m)]           # we omit m first eigenvectors, where m is the number of components of graph from A
    return Z


# Function prepares node representation used in trivial spectral clustering
#
# X - data set
# M - number of neighbours used in KNN (initially 3)
# nodeRepresentationDim - dimension of node representation (initially 3)
# return: labels assigned to data
def spectralNR(X, M, nodeRepresentationDim):
    A = adjacencyMatrixUsingMnearestNeighbors(X, M=M)
    A = connectTheGraph(A) # ensure the connectivity
    L = graphLaplacian(A)
    w,v = calculateEigenVectorsOfGraphLaplacian(L)
    Z = nodeRepresentation(v,nodeRepresentationDim,n_of_components=1)
    return Z


def spectralRevisedAdjacencyMatrix(X, M1 = 5, Z_dim = 5, M2 = 5, force_connectivity = False):
    
    Z = spectralNR(X, M1, Z_dim) # calculates A, then uses laplacian egien vactors to create a new embedding
    A = adjacencyMatrixUsingMnearestNeighbors(Z, M2) # a new adjacency matrix calculated using kNN
    if force_connectivity:
        A = connectTheGraph(A)
    return A
    
