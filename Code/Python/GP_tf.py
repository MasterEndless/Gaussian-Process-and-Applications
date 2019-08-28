import tensorflow as tf
from numpy.random import RandomState
import numpy as np
from scipy.linalg import cholesky, cho_solve
from sklearn.gaussian_process import GaussianProcessRegressor
import numpy.linalg as npl
import math as m
from matplotlib import pyplot as plt
from itertools import cycle
import csv

def RBF_ker(X1,X2,gamma):
    #change this calculation to multi-variable matrix
    #X: n*p   p : dimension of features  n: number of samples
    n1=X1.shape[0]
    n2=X2.shape[1]
    cov=np.zeros((n1,n2))
    for i in range(n1):
        for j in range(n2):
            cov[n1,n2]=np.exp((-np.norm(X1[i,:]-X2[j,:]))/(2*gamma**2))
    return cov


def Loss_func(x_sam,y_sam,std_sam,gamma):
    k=len(x_sam)
    K_pre=RBF_ker(x_sam,x_sam,gamma)
    d = np.diag(np.array(std_sam) ** 2)
    K=K_pre+d
    K_det=npl.det(K)
    left_side=-1/2*np.log(K_det*((2*np.pi)**k))
    lower_cholesky = cholesky(K, True)
    alpha = cho_solve((lower_cholesky, True), y_sam)
    y_sam_arr=np.array(y_sam)
    right_side = -1 / 2*np.dot(y_sam_arr.T,alpha)

    loss_func=right_side+left_side

    return loss_func

def GP_model_train(x_sam,y_sam,Std):
    sess = tf.Session()
    gamma = tf.Variable(tf.random_normal([1, 1], stddev=1, seed=1))
    row=np.size(x_sam,0)
    col=np.size(x_sam,1)

    x = tf.placeholder(tf.float32, shape=(row,col), name='x-input')
    y_ = tf.placeholder(tf.float32, shape=(1, row), name='y-input')
    std=tf.placeholder(tf.float32, shape=(1,row), name='std-input')
    x1=tf.reshape(x, [row,1,col])
    x2=tf.reshape(x,[1,row,col])
    x3 = x1 - x2
    x4 = tf.square(x3)
    cov = tf.exp(tf.divide(tf.reduce_sum(x4, axis=-1),-2*tf.square(gamma)))

    #cov=tf.exp(tf.divide(tf.square(tf.transpose(x) - x),-2*tf.square(gamma)))           #change from here!!!!!!!!!
    alpha=tf.matrix_diag(tf.square(std))
    alpha_=tf.reshape(alpha,[row,row])
    K_ = tf.add(alpha_,cov)
    K=tf.reshape(K_,[row,row])

    K_inv = tf.placeholder(tf.float32, shape=(row,row), name='inv_matrix')
    K_det = tf.placeholder(tf.float32, shape=(1, 1), name='det_matrix')

    #Left Side
    cons=tf.pow(tf.multiply(2.0,tf.constant(m.pi)),row)
    cons_1=tf.multiply(cons,K_det)
    cons_2=tf.log(cons_1)
    left=tf.multiply(0.5,cons_2)
    #Right side
    y_=tf.reshape(y_,(1,row))
    rig_1=tf.matmul(y_,K_inv)
    rig_2=tf.matmul(rig_1,tf.transpose(y_))
    right=tf.multiply(0.5,rig_2)

    loss=tf.add(left,right)

    train_step = tf.train.AdamOptimizer(0.001).minimize(loss)
    #dataset
    X = np.array(x_sam).reshape(row,col)
    Y = np.array(y_sam).reshape(1,row)
    STD = np.array(Std).reshape(1,row)

    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        STEPS = 10000
        count=0
        loss=np.zeros(STEPS)
        for i in range(STEPS):
            npK = sess.run(K, feed_dict={x: X, y_: Y,std:STD})
            K_inv=npl.inv(npK)
            K_det=npl.det(npK)
            sess.run(train_step, feed_dict={K_inv: K_inv, y_: Y, K_det: K_det})

            #loss[i]=Loss_func(x_sam,y_sam,Std,gamma.eval()[0][0])
        #real_loss=Loss_func(x_sam,y_sam,Std,gamma.eval()[0][0])
        #loss=np.abs(loss-real_loss*np.ones(STEPS))
        #x_axis = np.array(range(STEPS))
        '''
        plt.figure(1)
        plt.plot(x_axis, loss, 'r:', label=u'$Train_loss_TF$')
        plt.xlabel('Iteration times')
        plt.ylabel('Loss')
        plt.title('The loss during each iteration time')
        plt.xlim(0, 600)
        plt.show()
        '''
        return gamma.eval()[0][0]

def GP_model_test(sample_x, sample_y, sample_std,test_x,gamma,samples):
    sample_x = np.array(sample_x)
    cov_x=RBF_ker(sample_x,sample_x,gamma)
    d = np.diag(np.array(sample_std) ** 2)
    lower_cholesky = cholesky(cov_x + d,True)
    weighted_sample_y = cho_solve((lower_cholesky, True), sample_y)
    cov_te_sam=RBF_ker(test_x,sample_x,gamma)
    m=[]
    for i in cov_te_sam:
        m.append(cho_solve((lower_cholesky, True), i))
    cov = RBF_ker(test_x, test_x,gamma) - np.dot(cov_te_sam, np.array(m).T)
    mean = np.dot(cov_te_sam, weighted_sample_y)
    N=np.random.multivariate_normal(mean, cov, samples)
    return N

def Get_Mean_Std(sample_x,sample_std,test_x,gamma):
    sample_x = np.array(sample_x)
    cov_x=RBF_ker(sample_x,sample_x,gamma)
    d = np.diag(np.array(sample_std) ** 2)
    lower_cholesky = cholesky(cov_x + d,True)
    weighted_sample_y = cho_solve((lower_cholesky, True), sample_y)
    test_x = np.array([test_x]).flatten()
    means, stds = [], []
    for row in test_x:
        S0 = RBF_ker(row, sample_x,gamma)
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
    X.astype(np.float32)
    Y.astype(np.float32)
    return X,Y

X,Y=Data_extraction()

N=np.size(Y)
STD = 0.1*np.ones((N))
SAMPLES = 10

gamma=GP_model_train(X,Y,STD)
print(gamma)

# The following code is for plotting the graph

'''
test_x = np.linspace(0, 10, 1000)
predict_y=GP_model_test(sample_x,sample_y,sample_s,test_x,gamma,SAMPLES).flatten()
mean,std=Get_Mean_Std(sample_x,sample_s,test_x,gamma)

plt.figure()
plt.plot(test_x, f(test_x), 'r:', label=u'$f(x) = x\,\sin(x)$')
plt.plot(sample_x, sample_y, 'r.', markersize=10, label=u'Observations')
plt.plot(test_x, predict_y, 'b-', label=u'Prediction')

plt.fill(np.concatenate([test_x, test_x[::-1]]),
         np.concatenate([predict_y - 1.9600 * std,
                        (predict_y + 1.9600 * std)[::-1]]),
         alpha=.5, fc='b', ec='None', label='95% confidence interval')

plt.xlabel('$x$')
plt.ylabel('$f(x)$')
plt.ylim(-10, 20)
plt.legend(loc='upper left')
plt.show()


#Test stability
predict_y_2=GP_model_test(sample_x,sample_y,sample_s,test_x,gamma,7)
test_x = np.linspace(0, 10, 1000).reshape(1,1000)
mean=np.array(mean).reshape(1,1000)
std=np.array(std).reshape(1,1000)
colors = cycle(['g', 'b', 'k', 'y', 'c', 'r', 'm'])

plt.errorbar(test_x, mean, yerr=std,
             ecolor='g', linewidth=1.5,
             elinewidth=0.5, alpha=0.75)
test_x = np.linspace(0, 10, 1000).reshape(1000,1)
count=0
for sample, c in zip(predict_y_2, colors):
    plt.plot(test_x, sample, c, linewidth=2.*np.random.rand(), alpha=0.5)
    count+=1
plt.show()
'''