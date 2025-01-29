import random
import string
import numpy as np
from itertools import product
"""
File contains helper functions that are used in various ways in the simulation.
"""

def generate_random_string(exclude_list=[], length=5):
    """
    Generates a random string of given length. Excludes the strings in the exclude list
    :param exclude_list: List of strings to exclude
    :param length: string length
    :return: string
    """
    if exclude_list is None:
        exclude_list = []
    while True:
        random_string = ''.join(random.choices(string.ascii_letters + string.digits, k=length))
        if random_string not in exclude_list:
            return random_string

def hamming_distance(v1, v2):
    """
    Distance between two states. only maxes sense if values are discrete.
    :param v1:
    :param v2:
    :return:
    """
    return np.sum(v1 != v2)

def get_vectors(num_vectors, len=8, min_distance=0.1, max_distance=float('inf')):
    """
    Finds vectors of given len that are within min_distance and max_distance. If no bounds are given, returns random vectors.
    Useful for finding distinct environment states with bounded distances.
    :param num_vectors: Number of vectors to return
    :param len: Length of vectors to return
    :param min_distance: Lower bound for Hamming distance
    :param max_distance: Upper bound for Hamming Distance
    :return: An array of vectors
    """
    all_vectors = np.array(list(product([-1, 1], repeat=len)))

    # Shuffle the list of vectors to introduce randomness
    np.random.shuffle(all_vectors)

    selected_vectors = [all_vectors[0]]  # Start with the first (now random) vector
    for _ in range(1, num_vectors):
        max_dist_seen = -1
        best_vector = None
        for v in all_vectors:
            # Compute the minimum Hamming distance to the selected set
            distances = [hamming_distance(v, s) for s in selected_vectors]
            min_dist = min(distances)
            max_dist = max(distances)
            if min_dist >= min_distance and max_dist <= max_distance:
                best_vector = v
                break
        if best_vector is None:
            raise ValueError("environment distance not possible")
        selected_vectors.append(best_vector)
    return np.array(selected_vectors)

def get_vectors_not_in_array(num_vectors, original_vectors,l=8, min_distance=0.1, max_distance=float('inf')):
    """
    Finds vectors of given len that are within min_distance and max_distance that are not in the original vectors.
    If no bounds are given, returns random vectors. Useful for finding additional environments.
    :param num_vectors: Number of vectors to return
    :param len: Length of vectors to return
    :param min_distance: Lower bound for Hamming distance
    :param max_distance: Upper bound for Hamming Distance
    :return: An array of vectors
    """
    all_vectors = np.array(list(product([-1, 1], repeat=l)))

    # Shuffle the list of vectors to introduce randomness
    np.random.shuffle(all_vectors)
    selected_vectors = [all_vectors[0]]  # Start with the first (now random) vector
    for _ in range(1, num_vectors):
        max_dist_seen = -1
        best_vector = None
        for v in all_vectors:
            # Compute the minimum Hamming distance to the selected set
            distances = [hamming_distance(v, s) for s in selected_vectors]
            distances_to_original = [hamming_distance(v, s) for s in original_vectors]
            min_dist = min(distances)
            max_dist = max(distances)
            if min_dist >= min_distance and max_dist <= max_distance and min(distances_to_original) >0:
                best_vector = v
                break
        if best_vector is None:
            raise ValueError("environment distance not possible")
        selected_vectors.append(best_vector)
    return np.array(selected_vectors)

def print_hamming_distances(array):
    """
    Prints the Hamming distances of vectors within an array
    :param array:
    :return:
    """
    for target in array:
        string = ""
        for target2 in array:
            d = hamming_distance(target, target2)
            string += str(d) + "  "
        print(string)

def generate_prob_mat(p_destruction, p_creation):
    """
    Generates the mutation probability matrix. Assumptions are:
    [p_-1,-1  p_-1,0   p_-1,1]
    [p_0,-1   p_0,0    p_0,1 ]
    [p_1,-1   p_1,0    p_1,1 ]

    Assumptions:
    - p_-1,-1 = p_1,1
    - p_-1,0 =  p_1,0
    - p_0,-1 =  p_0,1
    - p_-1,1 =  p_1,-1 = p_-1,0 * p_0,1

    - Sum over rows = 1
    :param p_destruction: p_-1,0 probability of single step mutation of an already existing pathway
    :param p_creation:  p_0,1 probability of pathway creation from 0
    :return: prob matrix
    """

    p_2 = p_destruction * p_creation
    p_id = 1 - p_2 - p_destruction
    p_00 = 1 - p_creation * 2
    return np.array([[p_id, p_destruction, p_2],
                     [p_creation, p_00, p_creation],
                     [p_2, p_destruction, p_id]])

def mutation_rate_function(v, v_max, epsilon, rate_max, rate_min):
    """
    This function is used when implementing the simulation with changing mutation rates.
    This will linearize the growth rate such that its output is linear in the distance between current state and environment state.
    Linearization and interpolation based on the growth rate curve.
    :param v: Current growth rate
    :param v_max: maximal growth rate
    :param epsilon: epsilon parameter of the model. Must be >0 for this function
    :param rate_max: Desired growth rate at v=1.
    :param rate_min: Desired growth rate at v=v_max
    :return: mutation rate for the current growth rate
    """
    # np.log(v_max/v) / np.log(v_max/epsilon)print linearizes the growth rate to [0,1]
    # (rate_max - rate_min) + rate_min rescales to [rate_min,rate_max]
    m = (rate_max - rate_min) / (np.log(v_max/1) / np.log(v_max/epsilon)) # ensure that mutation_function(1) = rate_max
    return np.log(v_max/v) / np.log(v_max/epsilon) * m + rate_min


def to_key(array):
    """
    Converts a state (i.e [-1,1,-1,-1]) to a human-readable string and hashable string (i.e "-+--" ).

    :param array: numpy array representing the state
    :return:
    """
    if array.ndim > 1:
        raise ValueError
    s = ""
    for i in array:
        if i == 0:
            s += "0"
        elif i > 0.9:
            s += "+"
        elif i < -0.9:
            s += "-"
        else:
            raise ValueError

    return s


def key_toarray(key):
    """
    inverse of to_key
    :param key:
    :return:
    """
    array = []
    for s in key:
        i = 0
        if s == "+":
            i = 1.
        elif s == "-":
            i = -1.
        elif s == "0":
            i = 0
        else:

            raise ValueError(f"{s} not expected!")
        array.append(i)
    return np.array(array)


def mutation(old_mat,probs):
    """
    Mutates the reg matrix, while keeping the model constraints
    :param old_mat:
    :param probs:
    :return:
    """
    new_mat = np.copy(old_mat)
    states = np.array([-1, 0, 1])
    state_to_index = {-1: 0, 0: 1, 1: 2}

    # Constraints on Reg Mat: Diagonal =1, No regulation effects caused by environment genes.

    # if true that part of the matrix is left unchanged

    mask = np.zeros_like(old_mat, dtype=bool)
    mask[:, :8] = True #Targets have no effect on the rest of the genes
    mask[:8, old_mat.shape[0]:] = True #Inputs have no direct effect on environments
    np.fill_diagonal(mask, True)
    non_masked_states = old_mat[~mask]

    # Map the current states to transition matrix row indices
    state_indices = np.vectorize(state_to_index.get)(non_masked_states)

    # Use NumPy's advanced indexing to gather the corresponding transition probabilities
    transition_probs = probs[state_indices]

    # Generate random values for each non-masked element
    random_values = np.random.rand(len(non_masked_states))

    # Get the cumulative probabilities for each row
    cumulative_probs = np.cumsum(transition_probs, axis=1)

    # Determine the new state indices based on random values
    new_state_indices = (random_values[:, None] > cumulative_probs).sum(axis=1)

    # Map the new state indices back to original state values
    new_states = states[new_state_indices]

    # Update the new_data_matrix with the new states for non-masked elements
    new_mat[~mask] = new_states

    return new_mat