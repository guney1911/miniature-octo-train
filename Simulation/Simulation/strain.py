import time
import numpy as np

def generate_reg_matrix(size, targets, inputs,probabilities=None):
    """
    Generate the w matrix
    :param size: number of genes. The Matrix is always square for simplicity
    :param probabilities: [p_a,p_i] the ratios of activatory and inhibitory pathways
    :return: Regulation Matrix
    """
    if probabilities is not None:
        probabilities.append(1 - sum(probabilities))
        if sum(probabilities) < 0.99:
            raise ValueError('Probabilities must sum to 1')
    #print(size,environments,inputs)
    mat = np.random.choice([1, -1, 0], size=(size, size - targets), p=probabilities)

    input_mat = np.random.choice([1,-1,0],size=(size-targets,inputs),p = probabilities)

    for i in range(inputs): # ensure that input is actually used in the initial matrix
        if np.count_nonzero(input_mat[:,i]) == 0:
            ind = np.random.choice(range(size-targets),1)
            input_mat[ind,i] = 1
    input_mat = np.concatenate((np.zeros((targets,inputs)),input_mat),axis = 0)
    mat = np.concatenate((np.zeros((size, targets)), mat,input_mat), axis=1)
    np.fill_diagonal(mat, 1)

    return mat


class Strain:
    def __init__(self, copies:int ,N_t=8 , N_i=6,N=40, p1=0.04, p2=0.04, name="Default Strain", reg_mat = None):
        if reg_mat is None:
            self.reg_mat = generate_reg_matrix(N, N_t, N_i,[p1, p2])
        else:
            self.reg_mat = reg_mat

        self.name = name
        self.state = np.random.choice([-1., 1.], (copies,N))
        self.timestamp = time.time()
        self.v_history = np.empty((0,copies))
        self.alive = True
        self.avg_v = 0
        self.cells = copies

    def update_v_history(self,v):
        self.v_history = np.append(self.v_history, v,axis=0)

    def reset(self):
        #if not self.alive:
        self.state = np.random.choice([-1., 1.], self.state.shape)
        self.alive = True
        self.avg_v = 0
        self.v_history = np.empty((0,self.cells))
    def __str__(self):
        return f'{self.name}: {self.cells} cells'