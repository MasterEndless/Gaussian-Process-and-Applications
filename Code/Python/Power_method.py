import time
import numpy as np
import numpy.linalg as npl
from scipy.linalg import cholesky, cho_solve
from matplotlib import pyplot as plt

def Power_method(A,q,t):
    #make sure A is np.array
    #q is number of computation, use this to get mean
    #t is the iteration of algorithm
    n=A.shape[0]
    eig_val=np.zeros(q)
    for j in range(q):
        ran=np.random.uniform(-1, 1, n)
        for k in range(n):
            if ran[k]>0:
                ran[k]=1
            else:
                ran[k]=-1
        x=ran[:,np.newaxis]
        for i in range(t):
            x=np.dot(A,x)
        eig_val[j]=np.dot(x.T,np.dot(A,x))/np.dot(x.T,x)
    eig_max=np.max(eig_val)
    return eig_max

def Random_trace_est(A,e,sigma):
    n=A.shape[0]
    p=np.ceil(20*np.log(2./sigma)/(e**2)).astype(int)
    mean=np.zeros(p)
    cov=np.diag(np.ones(p))
    G=np.random.multivariate_normal(mean,cov,n)
    gamma=0
    for i in range(p):
        gamma=gamma+np.dot(G[:,i].T,np.dot(A,G[:,i]))
    gamma=gamma/p
    return gamma

def Parallel_compu(C,G,p):
    for i in range(p):
        v=np.dot(C,G[:,i])
        gamma=np.dot(G[:,i].T,v)
        Gamma[i,0]=gamma
        for k in range(m-1):
            v=np.dot(C,v)
            gamma=np.dot(G[:,i].T,v)
            Gamma[i,k+1]=gamma
    return Gamma

def Random_log_deter_est(A,e,sigma,m):
    #A is np.array, e is accuracy, sigma is failure probability
    n=A.shape[0]
    t=np.log(4*n).astype(int)
    q=np.around(3*np.log(1/sigma)).astype(int)
    alpha=Power_method(A,q,t)*7
    I=np.eye(n)
    C=I-A/alpha
    #p=np.ceil(20*np.log(2./sigma)/(e**2)).astype(int)
    p=60
    mean=np.zeros(p)
    cov=np.diag(np.ones(p))
    G=np.random.multivariate_normal(mean,cov,n)
    Gamma=np.zeros((p,m))
    for i in range(p):
        v=np.dot(C,G[:,i])
        gamma=np.dot(G[:,i].T,v)
        Gamma[i,0]=gamma
        for k in range(m-1):
            v=np.dot(C,v)
            gamma=np.dot(G[:,i].T,v)
            Gamma[i,k+1]=gamma
    sum=0
    count=1
    col_sum = np.sum(Gamma, axis=0)
    for k in range(m):
        sum=sum+col_sum[k]/(p*count)
        count=count+1
    log_det_A=n*np.log(alpha)-sum

    return log_det_A








def RBF_ker(x1,x2,gamma):
    cov=np.exp(-(np.subtract.outer(x1, x2))**2/(2*gamma**2))
    return cov

x=np.random.rand(50)
sigma=[0.1]*50
gamma=1.0
I=np.diag(sigma)
mmm_A=RBF_ker(x,x,gamma)+I

det=Random_log_deter_est(mmm_A,0.99,0.01,2000)

lower_cholesky = cholesky(mmm_A, True)
log=2*np.log(npl.det(lower_cholesky))
list=[]
for i in np.linspace(100, 2000, num=20):
    det = Random_log_deter_est(mmm_A, 0.99, 0.01, int(i))
    list.append(abs(det - log) / abs(log))
a=np.linspace(100, 2000, num=20)
plt.figure(1)
plt.plot(a, list, 'r:', label=u'$Train_loss_TF$')
plt.xlabel('Order')
plt.ylabel('Relative Error')
plt.show()



