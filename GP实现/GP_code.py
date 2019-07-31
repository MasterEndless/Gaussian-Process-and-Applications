import tensorflow as tf
from numpy.random import RandomState
import numpy as np
from scipy.linalg import cholesky, cho_solve
from sklearn.gaussian_process import GaussianProcessRegressor
import numpy.linalg as npl
import math as m
from matplotlib import pyplot as plt
from itertools import cycle

def RBF_ker(x1,x2,gamma):
    cov=np.exp(-(np.subtract.outer(x1, x2))**2/(2*gamma**2))
    return cov

def GP_model_train(x_sam,y_sam,Std):
    sess = tf.Session()
    gamma = tf.Variable(tf.random_normal([1, 1], stddev=1, seed=1))
    length=len(x_sam)

    x = tf.placeholder(tf.float32, shape=(1,length), name='x-input')
    y_ = tf.placeholder(tf.float32, shape=(1, length), name='y-input')
    std=tf.placeholder(tf.float32, shape=(1, length), name='std-input')

    cov=tf.exp(tf.divide(tf.square(tf.transpose(x) - x),-2*tf.square(gamma)))

    K = tf.add(tf.matrix_diag(tf.square(std)),cov)
    K=tf.reshape(K,(length,length))
    K_deter=tf.matrix_determinant(K)
    K_inv=tf.matrix_inverse(K)

    #Left Side
    cons=tf.pow(tf.multiply(2.0,tf.constant(m.pi)),length)             #k=3 here
    cons_1=tf.multiply(cons,K_deter)
    cons_2=tf.log(cons_1)
    left=tf.multiply(0.5,cons_2)
    #Right side
    y_=tf.reshape(y_,(1,length))
    rig_1=tf.matmul(y_,K_inv)
    rig_2=tf.matmul(rig_1,tf.transpose(y_))
    right=tf.multiply(0.5,rig_2)

    loss=tf.add(left,right)

    train_step = tf.train.AdamOptimizer(0.001).minimize(loss)
    #dataset
    X = np.array(x_sam).reshape(1,length)
    Y = np.array(y_sam).reshape(1,length)
    STD = np.array(Std).reshape(1,length)

    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        STEPS = 5000
        count=0
        for i in range(STEPS):
            sess.run(train_step, feed_dict={x: X, y_: Y,std:STD})
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



def f(x):               #simple test function
    return x * np.sin(x)


sample_x = [1., 3., 5., 6., 7., 8.]
sample_y = f(sample_x)
sample_s = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
SAMPLES = 1

gamma=GP_model_train(sample_x,sample_y,sample_s)

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
