'''
K-th order polynomial feature transformation
'''
import numpy as np

def poly_trans(raw_x, K):
    poly_coef = np.arange(K+1) #polynomial transformation coefficients
    trans_x = raw_x.repeat(K+1)
    trans_x = trans_x.reshape(-1, K+1)
    trans_x = np.power(trans_x, poly_coef)
    return trans_x.T # transpose, so each column is one data