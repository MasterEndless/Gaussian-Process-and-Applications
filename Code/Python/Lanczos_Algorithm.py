import numpy as np
import math
import numpy.linalg as npl
import scipy.linalg as scil
from scipy.linalg import norm, solve
import xlwt
def RBF_ker(x1,x2,gamma):
    cov=np.exp(-(np.subtract.outer(x1, x2))**2/(2*gamma**2))
    return cov


def save(data, path):
    f = xlwt.Workbook()
    sheet1 = f.add_sheet(u'sheet1', cell_overwrite_ok=True)
    [h, l] = data.shape
    for i in range(h):
        for j in range(l):
            sheet1.write(i, j, data[i, j])
    f.save(path)
def LanczosTri(A,x):
    '''Tridiagonalize Matrix A via Lanczos Iterations'''

    # Check if A is symmetric
    if ((A.transpose() != A).all()):
        print("WARNING: Input matrix is not symmetric")
    n = A.shape[0]
    V = np.zeros(n * n).reshape(n, n)  # Tridiagonalizing Matrix
    # Begin Lanczos Iteration
    q = x / np.linalg.norm(x)
    V[:, 0] = np.ravel(q)
    r = A @ q
    a1 = q.T @ r

    r = r - np.multiply(a1 , q)
    b1 = norm(r)
    s_min = 0  # Initialize minimum eigenvalue
    # print("a1 = %.12f, b1 = %.12f"%(a1,b1))
    for j in range(2, n+1):
        v = q
        q = r / b1
        V[:, j - 1] = np.ravel(q)
        r = A @ q - b1 * v
        a1 = (q.T @ A) @ q
        r = r - np.multiply(a1 , q)
        b1 = norm(r)
        if b1 == 0: break  # Need to reorthonormalize

    # Tridiagonal matrix similar to A
    T = V.T @ A @ V
    # Normalize via Frobenius Norm
    alpha = norm(T) / norm(A)
    T = T / alpha
    a=np.dot(V.T,V)
    c='C:\\Users\\15871\\Desktop\\test.xls'
    return T

x=[9., 70., 2., 15., 8., 5., 5., 60.]
gamma=1
sigma=[0.1]*8
I=np.diag(sigma)
mmm_A=np.matrix(RBF_ker(x,x,gamma)+I)
b=np.matrix([ 1.,  1., -1., -1.,  1.,  1., -1., -1.]).T
T=LanczosTri(mmm_A,b)

a,b=npl.eig(T)
c,d=npl.eig(mmm_A)
print(a)
print(c)



