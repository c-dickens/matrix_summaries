import numpy as np
from math import log, log2, floor,ceil, isclose, sqrt
from pprint import PrettyPrinter

class SparseCoreset:
    """
    An implementation of algorithm 2 from https://arxiv.org/pdf/2002.06296.pdf

    """
    def __init__(self,mat,acc=0.1,prob_fail=0.1,sample_size=None):
        """
        Inputs:
            mat - numpy ndarray of type float
            acc - float epsilon error
            prob_fail - float probability of failure
        """
        self.A = mat
        if self.A.ndim == 1:
            # tldr; take a (n,) array and reshape to (n,1) array
            # In python arrays with one dimension need to be converted
            # to a n x d array with d != None which can be done via the 
            # shorthand below.
            self.A = self.A[:,None]
        self.n,self.d = self.A.shape
        self.eps = acc
        self.delta = prob_fail
        
        self.theoretical_sample_size = ceil(3*self.d*self.eps**(-2)*(log2(self.d)**2 + log(1./self.delta)))
        if sample_size is not None:
            self.sample_size = sample_size
        else:
            # use theoretical bound
            self.sample_size = int(self.theoretical_sample_size)
        self.num_samplers = 8*self.sample_size
        
        # Experimental quantities
        self.theoretical_eps = self.theoretical_sample_size/self.sample_size
        self.theoretical_eps = sqrt(self.theoretical_eps)
        
        # Initialise the variables that evolve
        self.samplers = Samplers(self.num_samplers,self.d)

        
        print('**********    TESTING     **********')
        print('A.shape: ', self.A.shape)
        print('Accuracy: ', self.eps)
        print('Failure prob: ', self.delta)
        print('Sample size (lower bound-theory): ', self.theoretical_sample_size)
        print('Sample size (lower bound-stored): ', self.sample_size)
        print('Num Samplers (theory): ', 8*self.theoretical_sample_size)
        print('Num Samplers (stored): ', self.num_samplers)

        print('Theoretical eps: ', self.theoretical_eps)
        print('***********************************')
        
        
    def stream(self,block_size,seed=100):
        """
        Performs the streaming operation over input self.A
        """
        np.random.seed(seed)
        self.block_size = block_size
        self.num_blocks = int(ceil(self.n/self.block_size))
        self.covariance = np.zeros((self.d,self.d))
        
        for i in range(self.num_blocks):
            start = i*self.block_size
            end = np.min([(i+1)*self.block_size,self.n])
            block = self.A[start:end,:]
            #print(start,end,block.shape)
            #self.check[start:end,:] = block
            self.covariance += block.T@block
            
            # 2. iterate over the samplers to get uniform random numbers
            self.samplers.insert_point(block,list(range(start,end+1)))

            # 3. Obtain the thin SVD of current self.covariance matrix
            Z = self._thin_svd(self.covariance)
            current_rank = Z.shape[1] # rank (or total sensitivity) is num. columns
            
            # 4. Evaluate leverage scores in the samplers
            self.samplers.set_leverages(Z.T)
            
            # 5. Reject some of the stored indices.
            self.samplers.prune_samplers(current_rank)
            
            # 6. Get the singleton samplers
            self.samplers.get_singletons()
            
            # 7. Get the coreset
            weights,Q, scores, sampled_row_ids = self.samplers.get_coreset()
            #sampled_row_ids = list(self.samplers.get_coreset())
            # print('{} items sampled'.format(len(sampled_row_ids)))
            # print('{} distinct items sampled'.format(len(set(sampled_row_ids))))
            self.coreset = weights[:,None]*Q

        # This checks that the stored covariance == A^T A 
        # at the end of the stream.
        assert np.allclose(self.covariance, self.A.T@self.A)
        #print(self.samplers.samplers)
        # self.coreset = weights[:,None]*Q
    
    def _thin_svd(self, cov):
        """
        Obtains the Thin SVD of the matrix cov.
        Input:
            cov - d x d numpy ndarray type float
        Output:
            Z = RightInverse(D V^T) an r x d matrix with r = rank(cov) 
        First compute Z and then get a right inverse.
        A right inverse of X is:
        X* (X (X X^T)^{-1}) = I
        """
        # nb. This SVD implementation already returns V^T, *not* V
        U,S,Vt = np.linalg.svd(cov,full_matrices=False) 
        
        # Some auxiliary variables for error checking
        rank = np.linalg.matrix_rank(cov)
        
        # Remove the numerical unstable parts
        s = np.where(S > 1E-14)[0]
        S = S[s]
        Vt = Vt[s,:] # Remove the rows with numerically zero magnitude 
        D = np.sqrt(S) # Think we can broadcast here due to D being diagonal.
        Mz = np.diag(D)@Vt 
        
        
        Z = Mz.T@np.linalg.pinv(Mz@Mz.T)
        # print('The rank is: ', rank)
        # print('Shape if Mz@Z: ', (Mz@Z).shape)
        # assert np.allclose(Mz@Z, np.eye(rank))
        # assert Z.shape[1] == rank
        return Z
        
       

class Samplers:
    """
    Class to oeprate over all of the individual Sampler objects
    """
    def __init__(self, num_samplers,dimensionality):
        """
        Instantiates `num_samplers' independent SingleSamplers
        
        """
        self.samplers = {i : SingleSampler() for i in range(num_samplers)}
        self.dimensionality = dimensionality
        
    def insert_point(self, matrix, point_ids):
        '''
        Inserts `matrix' into each of the SingleSamplers in self.samplers
        point_ids is a list containing each of the row indices.
        ''' 
        for sampler in self.samplers.values():
            sampler.insert_point(matrix, point_ids)
            
    def set_leverages(self, mat):
        """
        Evaluates leverage scores wrt to `mat' for each of the 
        SingleSamplers
        """
        for sampler in self.samplers.values():
            sampler.set_leverage(mat)
    
    def prune_samplers(self,leverage_sum):
        """
        Removes all rows from the samplers which have a uniform 
        hash which is too large.
        """
        self.total_leverage = leverage_sum
        for s_id, sampler in self.samplers.items():
            sampler.prune_samplers(leverage_sum)
    
    def get_singletons(self):
        """
        Gets all of the singleton samplers
        """
        singletons = list(self.samplers.keys())
        for s_id,sampler in self.samplers.items():
            #sampler.dict_print()
            if len(sampler.get_keys()) != 1:
                singletons.remove(s_id)
        self.singleton_samplers = {s:self.samplers[s] for s in singletons}
        #print(f'There are {len(self.singleton_samplers)} singleton samplers.')
        # for i, _ in enumerate(self.samplers):
        #     s_i = self.samplers[i]
        #     r_id = s_i.get_keys()
        #     print('Sampled ids: ',r_id)
        #     sampled_ids[i] = r_id # Will throw an errow if r_id not a single item

        #     if  len(sampler_i) > 1:
        #         singletons.remove(i)
        # self.singleton_samplers = {self.samplers[i] for i in singletons}
        
    def get_coreset(self):
        """
        Obtains the coreset from the singleton samplers.
        Converts the dict into ndarray format.
        Returns:
            - sampled_ids: list of the original sampling indices
            - levs: list of leverage scores
            - samples: ndarray The rows sampled which correspond to self.A[sampled_ids,:]
            - weights: list containing the weight for each of the samples to make the coreset.
        
        nb. next(iter(...)) is just a more efficient way of doing list(dict.keys())[0]
        when we know that we only need the first key.
        See https://stackoverflow.com/questions/46042430/best-way-to-get-a-single-key-from-a-dictionary
        """
        sampled_ids = np.zeros((len(self.singleton_samplers),))
        levs = np.zeros((len(self.singleton_samplers),))
        subsample = np.zeros((len(self.singleton_samplers),self.dimensionality))
        
        for s_id, sampler_index in enumerate(self.singleton_samplers):
            # s_id enumerates the singleton_samplers map
            # sampler_index is the index of the singleton sampler in self.samplers
            # sampler is the singleton sampler we want to access.
            # i.e. sampler = Gamma_n^{i} from line 19 of Alg 2 in the paper.
            # and array `subsample' corresponds to their Q_n.
            sampler = self.samplers[sampler_index]
            data_index = next(iter(sampler.get_keys()))
            sampled_ids[s_id] = data_index
            item = sampler.get_item(data_index)
            levs[s_id] = item['leverage']
            subsample[s_id,:] = item['row']
            
        u_sampled_ids, u_local_index = np.unique(sampled_ids, return_index=True)
        B = subsample[u_local_index]
        leverage_scores = levs[u_local_index]
        unique_ids = u_sampled_ids
        
        weights = (1./np.sqrt(leverage_scores))
        if len(u_sampled_ids) > 0:
            weights *= np.sqrt(self.total_leverage / len(u_sampled_ids))
            #weights /= len(u_sampled_ids)
        else:
            weights *= 0.
        #print(weights)
        return weights, B, leverage_scores, unique_ids
        #return np.unique(sampled_ids).astype(int)
            
            
    
class SingleSampler:
    """
    A single sampler, independent of all of the others.
    """
    def __init__(self,):
        """
        Instantiates an empty dict for the sampler
        """
        self.sampler = {}
        
    def insert_point(self,points,point_ids):
        """
        Adds `points' into the sampler by looping over the rows of the array
        point_ids is the index for the dict
        """
        # 1. Get the uniform hash value
        u = np.random.uniform(size=len(points))
        
        # 2. Add to the dictionary
        for i,ai in enumerate(points):
            point_id = point_ids[i]
            self.sampler[point_id] = {
                'uniform_key' : u[i],
                'row'         : ai
            }
        
    def set_leverage(self, mat):
        """
        Evaluates the leverage score wrt matrix `mat'
        \| mat ai^T \|_2^2
        """
        for r_id in self.sampler.keys():
            ai = self.sampler[r_id]['row']
            self.sampler[r_id]['leverage'] = np.linalg.norm(mat@ai)**2
            #print(self.sampler[r_id]['leverage'])
            cond = (self.sampler[r_id]['leverage'] < 1) or isclose(self.sampler[r_id]['leverage'],1)
            assert(cond)
    
    def prune_samplers(self,leverage_sum):
        """
        Removes any point with a leverage ratio too large from 
        the samplers.
        """
        keys_to_del = []
        for r_id in self.sampler.keys():
            u = self.sampler[r_id]['uniform_key']
            lev_score = self.sampler[r_id]['leverage']
            lev_ratio = lev_score / (lev_score + leverage_sum)
            
            # If the uniform weight is too large then remove r_id from the sampler.
            # There is probably a better way to do this to avoid the auxiliary list
            if u > lev_ratio:
                keys_to_del.append(r_id)
        for k in keys_to_del:
            del self.sampler[k]
            
    def get_keys(self):
        """
        Returns the stored keys in dictionary
        """
        return list(self.sampler.keys())
    
    def get_item(self, key):
        """
        Returns the values of the dictionary key
        """
        return self.sampler[key]
    
    def dict_print(self):
        pp = PrettyPrinter(indent=4)
        pp.pprint(self.sampler)

