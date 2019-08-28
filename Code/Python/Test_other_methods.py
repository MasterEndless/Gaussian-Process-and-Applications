import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
import csv
from sklearn.cross_validation import train_test_split
from sklearn.gaussian_process import GaussianProcessRegressor

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

def computerCost(y_predict,y_real):
    m = y_real.shape[0]
    J = np.sum(np.square(y_predict-y_real))/(2000*m)
    return J

#First dataset forestfire
# build train and test dataset with ratio 7:3
X,y=Data_extraction()
x_train1, x_test, y_train1, y_test = train_test_split(X, y, test_size = 0.3)

#Uniform sampling
N=X.shape[0]
S=int(np.around(X.shape[0]*0.7))      #number of samples
F=X.shape[1]      #number of features

y_index=np.around(np.linspace(0,N,num=S,endpoint=False))
y_index=y_index.astype(int)
x_train=np.zeros((S,F))
y_train=np.zeros(S)
count=0

for i in y_index:
    x_train[count,:]=X[i,:]
    y_train[count]=y[i]
    count=count+1

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

gaussian=GaussianProcessRegressor()
gaussian.fit(x_train,y_train)
y_predict=gaussian.predict(x_test)
gaussian_loss=computerCost(y_predict,y_test)

#Plot the graph of all the loss function
name_list = ['Linear','Ridge','Lasso','Bayesian','Theil Sen','Gaussian']
num_list = [linear_loss,ridge_loss,lasso_loss,bayesian_loss,Theil_sen_loss,gaussian_loss]
plt.bar(range(len(num_list)), num_list,color='rgb',tick_label=name_list)
plt.title('Mean square error on each regression method')
plt.show()


