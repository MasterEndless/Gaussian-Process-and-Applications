import numpy as np
from scipy.linalg import cholesky, cho_solve
from sklearn.gaussian_process import GaussianProcessRegressor
import numpy.linalg as npl
from matplotlib import pyplot as plt
import csv
import math
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import time

def RBF_ker(X1,X2,gamma,theta):
    #change this calculation to multi-variable matrix
    #X: n*p   p : dimension of features  n: number of samples
    n1=X1.shape[0]
    n2=X2.shape[0]
    cov=np.zeros((n1,n2))
    a=X1[1, :] - X2[2, :]
    for i in range(n1):
        for j in range(n2):
            cov[i,j]=theta*np.exp((np.sum((X1[i,:]-X2[j,:])*(X1[i,:]-X2[j,:])))/(-2*gamma**2))
    return cov

def K_gamma_grad(X1,X2,gamma,theta):
    n1 = X1.shape[0]
    n2 = X2.shape[0]
    grad = np.zeros((n1, n2))
    for i in range(n1):
        for j in range(n2):
            norm_part = np.sum((X1[i,:]-X2[j,:])*(X1[i,:]-X2[j,:]))
            grad[i,j]=theta*norm_part*np.exp(-norm_part/(2*gamma**2))/(gamma **3)
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


def Cal_Grad(x_sam,y_sam,std_sam,gamma,theta):
    K_pre=RBF_ker(x_sam,x_sam,gamma,theta)
    d = np.diag(np.array(std_sam) ** 2)
    K=K_pre+d
    K_matrix=np.mat(K)
    K_inv=K_matrix.I.A
    K_grad_gamma=K_gamma_grad(x_sam,x_sam,gamma,theta)
    K_grad_theta=K_theta_grad(x_sam,x_sam,gamma)
    lower_cholesky = cholesky(K, True)
    alpha = cho_solve((lower_cholesky, True), y_sam)

    #gradient of gamma
    #calculate left side
    left_ga=-1/2*np.dot(np.dot(alpha.T,K_grad_gamma),alpha)
    #right side
    right_ga=1/2*np.trace(np.dot(K_inv,K_grad_gamma))
    loss_grad_gamma=right_ga+left_ga

    #gradient of theta
    #calculate left side
    left_th=-1/2*np.dot(np.dot(alpha.T,K_grad_theta),alpha)
    #right side
    right_th=1/2*np.trace(np.dot(K_inv,K_grad_theta))
    loss_grad_theta=right_th+left_th

    return loss_grad_gamma,loss_grad_theta

def Loss_func(x_sam,y_sam,std_sam,gamma,theta):
    k=len(x_sam)
    K_pre=RBF_ker(x_sam,x_sam,gamma,theta)
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
    theta = 5.046899               #initialize theta
    loss=np.zeros(N)
    for t in range(N):
        print(" The %d iteration" % (t+1))
        t=t+1
        g_t1,g_t2 = Cal_Grad(x_sam,y_sam,std_sam,gamma,theta)  # computes the gradient of the stochastic function
        m_t1 = beta_1 * m_t1 + (1 - beta_1) * g_t1  # updates the moving averages of the gradient
        v_t1 = beta_2 * v_t1 + (1 - beta_2) * (g_t1 * g_t1)  # updates the moving averages of the squared gradient
        m_cap1 = m_t1 / (1 - (beta_1 ** t))  # calculates the bias-corrected estimates
        v_cap1 = v_t1 / (1 - (beta_2 ** t))  # calculates the bias-corrected estimates

        m_t2 = beta_1 * m_t2 + (1 - beta_1) * g_t2  # updates the moving averages of the gradient
        v_t2 = beta_2 * v_t2 + (1 - beta_2) * (g_t2 * g_t2)  # updates the moving averages of the squared gradient
        m_cap2 = m_t2 / (1 - (beta_1 ** t))  # calculates the bias-corrected estimates
        v_cap2 = v_t2 / (1 - (beta_2 ** t))  # calculates the bias-corrected estimates

        gamma_prev = gamma
        theta_prev = theta
        gamma = gamma - (alpha * m_cap1) / (math.sqrt(v_cap1) + epsilon)  # updates the parameters
        theta = theta - (alpha * m_cap2) / (math.sqrt(v_cap2) + epsilon)  # updates the parameters
        #loss[t-1]=Loss_func(x_sam,y_sam,std_sam,gamma,theta)
        if (gamma == gamma_prev and theta == theta_prev):  # checks if it is converged or not
            #N=t
            #loss=loss[0:t]
            break
    #x_axis=np.array(range(N))
    #final_loss=Loss_func(x_sam,y_sam,std_sam,gamma)
    #real_loss=np.abs(loss-final_loss*np.ones(N))
    #if you want to plot the error vs time
    '''
    plt.figure(1)
    plt.plot(x_axis, real_loss, 'r:', label=u'$f(x) = x\,\sin(x)$')
    plt.xlabel('Iteration times')
    plt.ylabel('Loss')
    plt.title('The loss during each iteration time')
    plt.show()
    '''
    return gamma,theta

def GP_model_test(sample_x, sample_y, sample_std,test_x,gamma,samples,theta):
    cov_x=RBF_ker(sample_x,sample_x,gamma,theta)
    d = np.diag(np.array(sample_std) ** 2)
    lower_cholesky = cholesky(cov_x + d,True)
    weighted_sample_y = cho_solve((lower_cholesky, True), sample_y)
    cov_te_sam=RBF_ker(test_x,sample_x,gamma,theta)
    m=[]
    for i in cov_te_sam:
        m.append(cho_solve((lower_cholesky, True), i))
    cov = RBF_ker(test_x, test_x,gamma,theta) - np.dot(cov_te_sam, np.array(m).T)
    mean = np.dot(cov_te_sam, weighted_sample_y)
    N=np.random.multivariate_normal(mean, cov, samples)
    return N

def Get_Mean_Std(sample_x,sample_std,test_x,gamma,theta):
    cov_x=RBF_ker(sample_x,sample_x,gamma,theta)
    d = np.diag(np.array(sample_std) ** 2)
    lower_cholesky = cholesky(cov_x + d,True)
    weighted_sample_y = cho_solve((lower_cholesky, True), sample_y)
    test_x = np.array([test_x]).flatten()
    means, stds = [], []
    for row in test_x:
        S0 = RBF_ker(row, sample_x,gamma,theta)
        v = cho_solve((lower_cholesky, True), S0)
        means.append(np.dot(S0, weighted_sample_y))
        stds.append(np.sqrt(gamma ** 2 - np.dot(S0, v)))
    stds=np.array(stds)
    return means, stds

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

def computerCost(y_predict,y_real):
    m = y_real.shape[0]
    J = np.sum(np.square(y_predict-y_real))/(2000*m)
    return J

sample_x,sample_y=Data_extraction()
x_train1, x_test, y_train1, y_test = train_test_split(sample_x, sample_y, test_size = 0.3)


#Uniform sampling
N=sample_x.shape[0]
S=int(np.around(sample_x.shape[0]*0.7))      #number of samples
F=sample_x.shape[1]      #number of features
std_train = 0.1*np.ones((S))
y_index=np.around(np.linspace(0,N,num=S,endpoint=False))
y_index=y_index.astype(int)
x_train=np.zeros((S,F))
y_train=np.zeros(S)
count=0

for i in y_index:
    x_train[count,:]=sample_x[i,:]
    y_train[count]=sample_y[i]
    count=count+1



#Setting of Adam Optimizer
alpha=0.001         #learning rate
beta_1=0.9
beta_2=0.999
epsilon=10**(-8)
list=[]
for i in np.linspace(100,1100,num=11):
    i=int(i)
    start=time.time()
    gamma,theta=Adam_Optimizer_train(x_train,y_train,std_train,beta_1,beta_2,epsilon,alpha,i)
    end=time.time()
    print(end-start)
    list.append(end-start)


#interesting findings: gamma positive or negtive with the same numerical value (because the cost function is even
SAMPLES=20
predict_y_batch=GP_model_test(x_train,y_train,std_train,x_test,gamma,SAMPLES,theta)
predict_y=np.mean(predict_y_batch, axis=0)
gaussian_loss=computerCost(predict_y,y_test)

#Linear Regression
linear = linear_model.LinearRegression()
linear.fit(x_train,y_train)
y_predict=linear.predict(x_test)
linear_loss=computerCost(y_predict,y_test)

#Ridge Regression
ridge = linear_model.RidgeCV()
ridge.fit(x_train, y_train)
y_predict=ridge.predict(x_test)
ridge_loss= computerCost(y_predict,y_test)

#Lasso Regression
lasso = linear_model.Lasso(max_iter=10000, alpha=0.1)
lasso.fit(x_train, y_train)
y_predict=lasso.predict(x_test)
lasso_loss=computerCost(y_predict,y_test)

#Bayesian Ridge Regression
bayesian = linear_model.BayesianRidge(compute_score=True)
bayesian.fit(x_train, y_train)
y_predict=bayesian.predict(x_test)
bayesian_loss=computerCost(y_predict,y_test)

#Theil-Sen Regression
Theil_sen=linear_model.TheilSenRegressor()
Theil_sen.fit(x_train,y_train)
y_predict=Theil_sen.predict(x_test)
Theil_sen_loss=computerCost(y_predict,y_test)

#Plot the graph of all the loss function
name_list = ['Gaussian','Linear','Ridge','Lasso','Bayesian','Theil Sen']
num_list = [gaussian_loss,linear_loss,ridge_loss,lasso_loss,bayesian_loss,Theil_sen_loss]
plt.bar(range(len(num_list)), num_list,color='rgb',tick_label=name_list)
plt.title('Mean square error on each regression method')
plt.show()


