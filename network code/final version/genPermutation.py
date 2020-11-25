'''
    This python script can be called to generate a permutation to use with an INN.
    The generated permutation is formatted in a way, that it can be copy and pasted into the actual code to create an INN.
'''
import numpy as np

def random_perm(numb,seed = 174242069):
    np.random.seed(seed)
    perm = np.random.permutation(numb)
    return perm
    
print("Please enter an integer determining how long the randomly permuted list should be!")

a = int(input())
perm = random_perm(a)
print(perm.tolist(),sep=', ')
print(np.argsort(perm).tolist(),sep=', ')