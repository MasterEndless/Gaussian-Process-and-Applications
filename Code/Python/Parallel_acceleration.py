# -*- coding:utf-8 -*-
#All the calculation must be calculated with np.array
import numpy as np
import scipy.linalg as scil
import numpy.linalg as npl
import time
from matplotlib import pyplot as plt
import csv
import math

def RBF_ker(X1,X2,gamma):
    #change this calculation to multi-variable matrix
    #X: n*p   p : dimension of features  n: number of samples
    n1=X1.shape[0]
    n2=X2.shape[0]
    cov=np.zeros((n1,n2))
    a=X1[1, :] - X2[2, :]
    for i in range(n1):
        for j in range(n2):
            cov[i,j]=np.exp((np.sum((X1[i,:]-X2[j,:])*(X1[i,:]-X2[j,:])))/(-2*gamma**2))
    return cov

def K_gamma_grad(X1,X2,gamma):
    n1 = X1.shape[0]
    n2 = X2.shape[0]
    grad = np.zeros((n1, n2))
    for i in range(n1):
        for j in range(n2):
            norm_part = np.sum((X1[i,:]-X2[j,:])*(X1[i,:]-X2[j,:]))
            grad[i,j]=norm_part*np.exp(-norm_part/(2*gamma**2))/(gamma **3)
    return grad

def K_theta_grad(X1,X2,gamma):
    n1=X1.shape[0]
    n2=X2.shape[0]
    cov=np.zeros((n1,n2))
    a=X1[1, :] - X2[2, :]
    for i in range(n1):
        for j in range(n2):
            cov[i,j]=np.exp((np.sum((X1[i,:]-X2[j,:])*(X1[i,:]-X2[j,:])))/(-2*gamma**2))
    return cov


def Cal_Grad(x_sam,y_sam,std_sam,gamma):
    K_pre=RBF_ker(x_sam,x_sam,gamma)
    d = np.diag(np.array(std_sam) ** 2)
    K=K_pre+d
    B=Matrix_generator(y_sam,100)
    I=np.eye(100)
    U=mBCG(K,B,I)
    K_grad_gamma=K_gamma_grad(x_sam,x_sam,gamma)
    K_grad_theta=K_theta_grad(x_sam,x_sam,gamma)
    alpha = U[:,0]
    trace=Trace_est(K_grad_gamma,U,B)

    #gradient of gamma
    #calculate left side
    left_ga=-1/2*np.dot(np.dot(alpha.T,K_grad_gamma),alpha)
    #right side
    right_ga=1/2*trace
    loss_grad_gamma=right_ga+left_ga


    return loss_grad_gamma

def Loss_func(x_sam,y_sam,std_sam,gamma):
    k=len(x_sam)
    K_pre=RBF_ker(x_sam,x_sam,gamma)
    d = np.diag(np.array(std_sam) ** 2)
    K=K_pre+d
    K_det=npl.det(K)
    left_side=-1/2*(np.log(K_det)+k*np.log(2*np.pi))
    lower_cholesky = cholesky(K, True)
    alpha = cho_solve((lower_cholesky, True), y_sam)
    y_sam_arr=np.array(y_sam)
    right_side = -1 / 2*np.dot(y_sam_arr.T,alpha)

    loss_func=right_side+left_side

    return loss_func


def Adam_Optimizer_train(x_sam,y_sam,std_sam,beta_1,beta_2,epsilon,alpha,N):
    m_t1 = 0
    v_t1 = 0
    m_t2 = 0
    v_t2 = 0
    t = 0

    gamma = -0.811024         #randomly initialize gamma
    for t in range(N):
        print(" The %d iteration" % (t+1))
        t=t+1
        g_t1= Cal_Grad(x_sam,y_sam,std_sam,gamma)  # computes the gradient of the stochastic function
        m_t1 = beta_1 * m_t1 + (1 - beta_1) * g_t1  # updates the moving averages of the gradient
        v_t1 = beta_2 * v_t1 + (1 - beta_2) * (g_t1 * g_t1)  # updates the moving averages of the squared gradient
        m_cap1 = m_t1 / (1 - (beta_1 ** t))  # calculates the bias-corrected estimates
        v_cap1 = v_t1 / (1 - (beta_2 ** t))  # calculates the bias-corrected estimates

        gamma_prev = gamma
        gamma = gamma - (alpha * m_cap1) / (math.sqrt(v_cap1) + epsilon)  # updates the parameters
        if (gamma == gamma_prev and theta == theta_prev):  # checks if it is converged or not
            break

    return gamma

def Matrix_generator(y,t):
    y_=np.array(y)
    row=y_.shape[0]
    y=y_[np.newaxis, :].T

    #generate random (-1,1) two point distribution
    Z=np.zeros((row,t))
    for i in range(t):
        ran=np.random.uniform(-1, 1, row)
        for j in range(row):
            if ran[j]>0:
                ran[j]=1
            else:
                ran[j]=-1
        Z[:,i]=ran
    B=np.hstack((y,Z))
    return B

#This algorithm has a flaw that it does not converge when t is small (t has be a large number)
def mBCG(mmm_A,B,P_1):
    #p is the number of iteration
    row=mmm_A.shape[0]
    t=B.shape[1]
    U=np.zeros((row,t))
    R=B-np.dot(mmm_A,U)
    Z=np.dot(P_1,R)
    D=Z
    T=np.zeros((t,t,t))
    alpha=np.zeros((t,1))
    beta=np.zeros((t,1))
    for j in range(t):
        V=np.dot(mmm_A,D)
        alpha = (np.sum(R*Z, axis=0)/(0.0001+np.sum(D*V,axis=0))).T
        U = (U.T+np.dot(np.diag(alpha),D.T)).T
        Z_old=np.sum(Z*R,axis=0)
        R=(R.T-np.dot(np.diag(alpha),V.T)).T
        Z=np.dot(P_1,R)
        beta=(np.sum(R*Z, axis=0)/(Z_old+0.0001)).T
        D=(Z.T+np.dot( np.diag(beta),D.T)).T
    return U

def Trace_est(K_grad_gamma,U,B):
    t=B.shape[1]-1
    sum=0
    for i in range(t):
        sum=sum+np.dot(U[:,i+1].T,np.dot(K_grad_gamma,B[:,i+1]))
    Tr=sum/t
    return Tr




def Data_extraction():
    csvFile = open("forestfires.csv", "r")
    reader = csv.reader(csvFile)
    month_dict = {'jan': '1', 'feb': '2', 'mar': '3', 'apr': '4', 'may': '5', 'jun': '6', 'jul': '7', 'aug': '8',
                  'sep': '9', 'oct': '10', 'nov': '11', 'dec': '12'}
    day_dict = {'mon': '1', 'tue': '2', 'wed': '3', 'thu': '4', 'fri': '5', 'sat': '6', 'sun': '7'}
    Data = np.zeros((517, 13))
    count = 0
    for item in reader:
        if reader.line_num == 1:
            continue
        rep_month = [month_dict[x] if x in month_dict else x for x in item]
        rep_day = [day_dict[x] if x in day_dict else x for x in rep_month]
        rep_number = [float(x) for x in rep_day]
        Data[count, :] = rep_number
        count = count + 1
    X = Data[0:517, 0:12]
    Y = Data[0:517, 12]
    return X,Y

sample_x,sample_y=Data_extraction()
sample_x=sample_x[0:100,:]
sample_y=sample_y[0:100]
N=sample_x.shape[0]
std_train = 0.1*np.ones(N)

#Setting of Adam Optimizer
alpha=0.001         #learning rate
beta_1=0.9
beta_2=0.999
epsilon=10**(-8)
time_list=[]
for i in np.linspace(100,2000,num=20):
    N=int(i)
    start=time.time()
    gamma=Adam_Optimizer_train(sample_x,sample_y,std_train,beta_1,beta_2,epsilon,alpha,N)
    end=time.time()
    time_list.append(end-start)

print(time_list)
iteration=np.linspace(100,2000,num=20)
plt.figure(1)
plt.plot(iteration, time_list, 'r:', label=u'iteration')
plt.xlabel('Iteration times')
plt.ylabel('Time')
plt.show()







