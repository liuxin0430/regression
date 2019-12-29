'''
implement 4 regression algorithms:
1. least squares(LS)
2. regularized LS(RLS)
3. L1-regularized LS(LASSO)
4. robust regression(RR)
'''
'''
In this project, I will use CVXOPT to solve quadratic programming and linear programming
You can install cvxopt by: pip install cvxopt
'''

import numpy as np
import cvxopt

'''
least squares(LS)
phi - h
theta - w
'''
def LS_regression(h, y):
    h_hT = np.dot(h, h.T)
    #print(h_hT.shape)
    if np.linalg.det(h_hT) == 0:
        print("This matrix is singular, cannot be inversed!")
        return None
    else:
        w = np.linalg.inv(h_hT).dot(h).dot(y)
        return w


'''
regularized LS(RLS)
lambda - r
'''
def RLS_regression(h, y, r=0.01):
    h_hT = np.dot(h, h.T)

    #I = np.identity(h_hT.shape[0])
    I = np.identity(len(h_hT))
    I = r*I
    h_hT = h_hT + I
    #print(h_hT.shape)
    if np.linalg.det(h_hT) == 0:
        print("This matrix is singular, cannot be inversed!")
        return None
    else:
        w = np.linalg.inv(h_hT).dot(h).dot(y)
        return w


'''
L1-regularized LS(LASSO)
least absolute shrinkage and selection operator
'''

def cvxopt_solve_qp(H, f, A=None, b=None, Aeq=None, beq=None):
    H = .5 * (H + H.T)  # make sure P is symmetric
    args = [cvxopt.matrix(H), cvxopt.matrix(f)]
    if A is not None:
        args.extend([cvxopt.matrix(A), cvxopt.matrix(b)])
        if Aeq is not None:
            args.extend([cvxopt.matrix(Aeq), cvxopt.matrix(beq)])
    sol = cvxopt.solvers.qp(*args)
    if 'optimal' not in sol['status']:
        return None
    return np.array(sol['x']).reshape((H.shape[1],))


def LASSO_regression(h, y, r=0.01):
    h_hT = np.dot(h, h.T)
    H1 = np.concatenate((h_hT,-h_hT), axis=1)
    H2 = np.concatenate((-h_hT,h_hT), axis=1)
    H = np.concatenate((H1,H2), axis=0)

    h_y = np.dot(h, y)
    f = -np.concatenate((h_y, -h_y), axis=0)
    f = f + r*np.ones(f.shape)

    A = -np.identity(len(f))
    b = np.zeros(f.shape)

    sol = cvxopt_solve_qp(H, f, A, b)
    w1 = sol[:int(len(sol)/2)] # w+
    w2 = sol[int(len(sol)/2):] # w-

    w = w1 - w2
    return w


'''
robust regression(RR)
'''
def cvxopt_solve_lp(f, A, b):
    sol = cvxopt.solvers.lp(cvxopt.matrix(f), cvxopt.matrix(A), cvxopt.matrix(b))
    return np.array(sol['x']).reshape((f.shape[0],))

def RR_regression(h,y):
    n = len(y)
    D = len(h)

    f = np.concatenate((np.zeros(D), np.ones(n)),axis=0)

    I_n = np.identity(n)
    A1 = np.concatenate((-h.T, -I_n),axis=1)
    A2 = np.concatenate((h.T, -I_n),axis=1)
    A = np.concatenate((A1,A2),axis=0)

    b = np.concatenate((-y,y),axis=0)
    sol = cvxopt_solve_lp(f, A, b)
    w = sol[:D]
    return w

