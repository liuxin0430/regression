'''
test regression methods on a simple dataset
'''
import numpy as np
import matplotlib.pyplot as plt
from poly_trans import poly_trans
from regression import LS_regression, RLS_regression, LASSO_regression, RR_regression

#plot the estimated function using polyx as inputs, along with the sample data
def visualize(pred_y, sampx, sampy, name):
    plt.figure()
    #plot the estimated function using polyx as inputs, along with the sample data
    plt.plot(polyx, pred_y, color="blue", linewidth=1.0, linestyle="-",label="estimated function")
    
    plt.scatter(sampx,sampy,s=1, c='black',label="samples")
    plt.legend()
    plt.grid(True, linestyle=":")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(name)
    plt.savefig(name+".jpg")

#get sample data
sampx_file = "data\\polydata_data_sampx.txt"
sampy_file = "data\\polydata_data_sampy.txt"
sampx = np.loadtxt(sampx_file)
sampy = np.loadtxt(sampy_file)
#get poly data
polyx_file = "data\\polydata_data_polyx.txt"
polyy_file = "data\\polydata_data_polyy.txt"
polyx = np.loadtxt(polyx_file)
polyy = np.loadtxt(polyy_file)


K = 5
names = ["LS","RLS","LASSO","RR"]
#polynomial features transformation
h = poly_trans(sampx, K)
print(h.shape)
print(sampy.shape)


#get parameters w
w_ls = LS_regression(h, sampy)
w_rls = RLS_regression(h, sampy)
w_lasso = LASSO_regression(h, sampy)
w_rr = RR_regression(h, sampy)


for alg_name in names:
    
    if alg_name == "LS":
        w = w_ls
    elif alg_name == "RLS":
        w = w_rls
    elif alg_name == "LASSO":
        w = w_lasso
    elif alg_name == "RR":
        w = w_rr

    #prediction f* for input x*
    trans_x = poly_trans(polyx, K)
    pred_y = np.dot(w.T, trans_x)
    #plot
    visualize(pred_y, sampx, sampy, alg_name)

    # mean-squared error between the learned function outputs and the true function outputs(polyy)
    mse = mse = (np.square(pred_y - polyy)).mean()
    print("mean-squared error of {} : {}".format(alg_name, mse))
