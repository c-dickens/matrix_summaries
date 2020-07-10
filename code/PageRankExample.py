import sys
import numpy as np
from numpy.linalg import norm
from numpy import dot, zeros
from scipy.io import loadmat
import networkx as nx

from syntheticDataMaker import SyntheticDataMaker
from frequentDirections import FrequentDirections

def pagerank(M, num_iterations: int = 100, d: float = 0.85):
    """PageRank: The trillion dollar algorithm.

    Parameters
    ----------
    M : numpy array
        adjacency matrix where M_i,j represents the link from 'j' to 'i', such that for all 'j'
        sum(i, M_i,j) = 1
        This just means that M is an adjacency matrix with outgoing links in the columns 
        and incoming links in the rows.
    num_iterations : int, optional
        number of iterations, by default 100
    d : float, optional
        damping factor, by default 0.85

    Returns
    -------
    numpy array
        a vector of ranks such that v_i is the i-th rank from [0, 1],
        v sums to 1
        
    Taken from https://en.wikipedia.org/wiki/PageRank#Simplified_algorithm

    ***********          ***********          ***********
    NB. IF USING NETWORKX TO DEFINE THE GRAPH IT ***MUST*** BE TRANSPOSED PRIOR TO CALLING 
    THIS FUNCTION!!!!!
    ***********          ***********          ***********

    """
    N = M.shape[1]
    v = np.random.rand(N, 1)
    v = v / np.linalg.norm(v, 1)
    M_hat = (d * M + (1 - d) / N)
    
    evals,evec = np.linalg.eig(M_hat)
    v1 = np.real(evec[:,0])
    v1 /= np.sum(v1)
    v1 = v1[:,None]
    for i in range(num_iterations):
        v = M_hat @ v
    error = np.linalg.norm(v - v1)    
    print('Error:{:.3f}'.format(error))
    return np.squeeze(v)

def main():

    # Build the Directed Graph object in networkx
    #n = 1000
    # n_clusters = 20
    # cluster_size = n // n_clusters
    # p_in_cluster = 0.25
    # p_out_cluster = 0.025
    #G = nx.planted_partition_graph(n_clusters, cluster_size, p_in_cluster, p_out_cluster,seed=100,directed=True)
    #G = nx.windmill_graph(n_clusters,cluster_size)
    # G = nx.davis_southern_women_graph()


    # # Get the adjaceny matrix in numpy format
    A = np.loadtxt('gre_1107.txt', delimiter=',')
    G = nx.from_numpy_array(A)
    print(A.shape, type(A), type(A[0,0]))
    print(type(G))
    A = nx.to_numpy_array(G)
    n = A.shape[0]
    print(f'{n} nodes in G')

    # Do page rank algorithm
    damping_factor = 0.85
    nx_pr = nx.pagerank(G, alpha=damping_factor)
    nx_pr_np = np.array(list(nx_pr.values())) # convert to list for comparison with other methods.
    
    # Numpy implementation of pagerank
    passes = 100
    A_T  = A.T # transposing so we are in the 'standard' pagerank setup
    A_T /= np.sum(A_T,axis=0) # normalise every column by number of out-neighbours
    v_pr = pagerank(A_T,num_iterations=passes)
    v_single = pagerank(A_T,num_iterations=1) # single pass power iteration

    # Sketch the input with Frequent Directions - WE USE THE NETWORKX --> NUMPY ADJACENCY MATRIX
    n = A.shape[0]
    ell = 50 # sketch size
    sketcher = FrequentDirections(n,ell)
    for i in range(n):
        row = A[i,:]
        sketcher.append(row)
    sketch = sketcher.get()

    # Now get the eigenvector
    _,_,Vt = np.linalg.svd(sketch,full_matrices=False)
    sv1 = Vt[0,:]
    sv1 /= np.sum(sv1)
    fd_pr =  damping_factor * sv1 + (1 - damping_factor/n)*1
    fd_pr /= np.sum(fd_pr)


    # Errors:
    print('Error (numpy - networkx): {0:.3E}'.format(np.linalg.norm(nx_pr_np-v_pr)**2))
    print('Error (numpy - networkx 1 pass): {0:.3E}'.format(np.linalg.norm(nx_pr_np-v_single)**2))
    print('Error (fd - networkx - no correction): {0:.3E}'.format(np.linalg.norm(nx_pr_np-sv1)**2))
    print('Error (fd - networkx - correction): {0:.3E}'.format(np.linalg.norm(nx_pr_np-fd_pr)**2))



    



if __name__ == '__main__':
    main()