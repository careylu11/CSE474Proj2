import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt
import scipy.io
import matplotlib.pyplot as plt
import pickle

def ldaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmat - A single d x d learnt covariance matrix 
    # Combine and sort based on class
    covmat = np.cov(np.transpose(X))
    d = X.shape[1]
    X = np.hstack((X,y))
    X = X[X[:,2].argsort()]
    indicies = []
    # Find where to split
    for i in range(1,len(y)):
     if (X[i][2] != X[i-1][2]):
      indicies.append(i)
    means = np.arange(10,dtype=int)
    means = means.reshape(d,len(indicies)+1)
    split = np.vsplit(X,indicies)
    # Take mean and put into means matrix
    for i in range(0,len(split)):
     x = split[i][:,0].mean()
     y = split[i][:,1].mean()
     means[0][i] = x
     means[1][i] = y


    return means,covmat

def qdaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmats - A list of k d x d learnt covariance matrices for each of the k classes
    covmats = np.array([])
    # IMPLEMENT THIS METHOD
    d = X.shape[1]
    X = np.hstack((X,y))
    X = X[X[:,2].argsort()]
    indicies = []
    covmats = np.array([[[0,0],[0,0]]])
    # Find where to split
    for i in range(1,len(y)):
     if (X[i][2] != X[i-1][2]):
      indicies.append(i)
    means = np.arange(10,dtype=float)
    means = means.reshape(d,len(indicies)+1)
    split = np.vsplit(X,indicies)
    # Take mean and put into means matrix
    # Also get covariance for each k class and add it to covmats
    for i in range(0,len(split)):
     split[i] = np.delete(split[i], 2, 1)
     cov = np.cov(np.transpose(split[i]))
     adder = np.array([cov])
     covmats = np.vstack((covmats,adder))
     x = split[i][:,0].mean()
     y = split[i][:,1].mean()
     means[0][i] = x
     means[1][i] = y
    covmats = np.delete(covmats, 0, 0)
    return means,covmats

def ldaTest(means,covmat,Xtest,ytest):
    # Inputs
    # means, covmat - parameters of the LDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value

    theta = np.linalg.det(covmat)
    means = np.transpose(means)
    X = Xtest.shape[0]
    bigmatrix = np.array([0,0,0,0,0])
    first = 1/(theta*np.sqrt(2*np.pi))
    #iterate through Xtest and our means matrix calculating pdf
    for x in Xtest:
     rowofbig = np.array([])
     for m in means:
      numerator = x - m
      numerator = np.dot(numerator, np.transpose(numerator))
      denom = theta*theta
      power = -.5*(numerator/denom)
      second = np.exp(power)
      prob = first*second
      rowofbig = np.hstack((rowofbig,prob))
     bigmatrix = np.vstack((bigmatrix, rowofbig))

    #Get the highest probabilities from our new matrix
    bigmatrix = np.delete(bigmatrix, 0, 0)
    pdfs = np.argmax(bigmatrix, axis=1)
    pdfs = np.add(pdfs,1)
    #pdfs = pdfs.astype(float)
    #Compares our probability of k class to ytest and return a true/false matrix
    acc = 0.0
    #If our value is true then our predicition matches the ytest
    for i in range(ytest.shape[0]):
     arr = np.array(ytest[i])
     if (arr[0] == pdfs[i]):
      acc = acc + 1
    acc = acc/100
    # IMPLEMENT THIS METHOD
    return acc

def qdaTest(means,covmats,Xtest,ytest):
    # Inputs
    # means, covmats - parameters of the QDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    #theta = np.linalg.det(covmat)
    covs = [0.0,0.0,0.0,0.0,0.0]
    means = np.transpose(means)
    for i in range(covmats.shape[0]):
     det = np.linalg.det(covmats[i])
     covs[i] = det
    X = Xtest.shape[0]
    bigmatrix = np.array([0,0,0,0,0])
    #iterate through Xtest and our means matrix calculating pdf
    for x in Xtest:
     rowofbig = np.array([])
     for m in range(means.shape[0]):
      theta = covs[m]
      numerator = (x - means[m])/theta
      first = 1/(theta*np.sqrt(2*np.pi))
      numerator = np.dot(numerator, np.transpose(numerator))
      #denom = theta*theta
      power = -.5*(numerator)
      second = np.exp(power)
      prob = first*second
      rowofbig = np.hstack((rowofbig,prob))
     bigmatrix = np.vstack((bigmatrix, rowofbig))

    #Get the highest probabilities from our new matrix
    bigmatrix = np.delete(bigmatrix, 0, 0)
    pdfs = np.argmax(bigmatrix, axis=1)
    pdfs = np.add(pdfs,1)
    #Compares our probability of k class to ytest and return a true/false matrix
    acc = 0.0
    #If our value is true then our predicition matches the ytest
    for i in range(ytest.shape[0]):
     arr = np.array(ytest[i])
     #print("ytest[0],pdfs[i]",arr[0],pdfs[i])
     if (arr[0] == pdfs[i]):
      acc = acc + 1
    acc = acc/100
    # IMPLEMENT THIS METHOD
    return acc


def learnOLERegression(X,y):
    # Inputs:                                                         
    # X = N x d 
    # y = N x 1                                                               
    # Output: 
    # w = d x 1
    print("learning" , X.shape)
    print("learning" , y.shape)                                                                
    w = np.dot(np.linalg.inv(np.dot(X.transpose(),X)),np.dot(X.transpose(),y))
   # print w.shape
    # IMPLEMENT THIS METHOD                                                   
    return w

def learnRidgeERegression(X,y,lambd):
    # Inputs:
    # X = N x d                                                               
    # y = N x 1 
    # lambd = ridge parameter (scalar)
    # Output:                                                                  
    # w = d x 1
    n = X.shape[0]
    w = np.dot(np.linalg.inv(((n*lambd*np.identity(X.shape[1])) + np.dot(np.transpose(X),X))),np.dot(np.transpose(X),y))
  
    return w

def testOLERegression(w,Xtest,ytest):
    # Inputs:
    # w = d x 1
    # Xtest = N x d
    # ytest = X x 1
    # Output:
    
    N = Xtest.shape[0]
    
    m2 = np.transpose(ytest - np.dot(Xtest,w))
    m3 = np.square(m2)
    m4 = np.sum(m3)
   
    M = np.sqrt(m4)
    rmse = M/N
    print("rmse value", rmse)
    return rmse

def regressionObjVal(w, X, y, lambd):

    # compute squared error (scalar) and gradient of squared error with respect
    # to w (vector) for the given data X and y and the regularization parameter
    # lambda                                                                  
    # IMPLEMENT THIS METHOD

    y1 = np.zeros((242,))
    
    for i in range (242):
        y1[i] = y[i]
    
    n = X.shape[0]
    x1 = np.dot(X, w)
   
    sumProduct = y1 - x1
    bracketValue = np.dot(sumProduct.transpose(), sumProduct)
    error = np.sum(bracketValue)/(2*n)
    reg = (lambd/2)*(np.dot(np.transpose(w),w))
    error = (error + reg)
    
    x2 = np.dot(X.transpose(), X)
    x4 = np.dot(w.transpose(), x2)
    x3 = np.dot(X.transpose(), y1)
  
    error_grad = ((x4-x3)/n)+(lambd*w)
    error_grad = error_grad

    print("error is: ", error)
    return error, error_grad


def mapNonLinear(x,p):
    # Inputs:                                                                  
    # x - a single column vector (N x 1)                                       
    # p - integer (>= 0)                                                       
    # Outputs:                                                                 
    # Xd - (N x (d+1))                                                         
    # IMPLEMENT THIS METHOD
    Xd = np.ones((x.shape[0], p+1))

    for i in range (0, p+1):
        Xd[:, i] = pow(x, i)
    return Xd

# Main script


# Problem 1
# load the sample data
print("STARTING PROBLEM 1---------------------------------------")
X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'))            
# LDA
means,covmat = ldaLearn(X,y)
ldaacc = ldaTest(means,covmat,Xtest,ytest)
print('LDA Accuracy = '+str(ldaacc))
# QDA
means,covmats = ldaLearn(X,y)
qdaacc = ldaTest(means,covmats,Xtest,ytest)
print('QDA Accuracy = '+str(qdaacc))

# Problem 2
print("STARTING PROBLEM 2---------------------------------------")
X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'))   
# add intercept
X_i = np.concatenate((np.ones((X.shape[0],1)), X), axis=1)
Xtest_i = np.concatenate((np.ones((Xtest.shape[0],1)), Xtest), axis=1)
X_train_i = np.concatenate((np.ones((X.shape[0],1)), X), axis=1)

w = learnOLERegression(X,y)
mle = testOLERegression(w,Xtest,ytest)

w_i = learnOLERegression(X_i,y)
mle_i = testOLERegression(w_i,Xtest_i,ytest)

#Error training
mle_training = testOLERegression(w,X,y)

w_i_training = learnOLERegression(X_i,y)
mle_i_training = testOLERegression(w_i_training, X_train_i, y)


# print('RMSE without intercept (test) '+str(mle))
# print('RMSE with intercept (test) '+str(mle_i))
#
# print('RMSE without intercept (training) '+str(mle_training))
# print('RMSE with intercept (training) '+str(mle_i_training))

'''
# Problem 3
print("STARTING PROBLEM 3---------------------------------------")

k = 80
lambdas = np.linspace(0, 0.004, num=k)
i = 0
rmses3 = np.zeros((k,1))
rmses3_training = np.zeros((k,1))
for lambd in lambdas:
    w_l = learnRidgeERegression(X_i,y,lambd)
    #rmses3[i] = testOLERegression(w_l,Xtest_i,ytest)
    rmses3_training[i] = testOLERegression(w_l,X_train_i,y)
    i = i + 1
plt.plot(lambdas,rmses3)
plt.plot(lambdas,rmses3_training)
plt.show()

#Problem 4
print("STARTING PROBLEM 4---------------------------------------")
k = 101
i = 0
lambdas = np.linspace(0, 0.004, num=k)

rmses4 = np.zeros((k,1))
rmses4_training = np.zeros((k,1))
opts = {'maxiter' : 200}    # Preferred value.
w_init = np.zeros((X_i.shape[1],1))
for lambd in lambdas:
    args = (X_i, y, lambd)
    w_l = minimize(regressionObjVal, w_init, jac=True, args=args,method='CG', options=opts)
    w_l_1 = np.zeros((X_i.shape[1],1))
    
    for j in range(len(w_l.x)):
        w_l_1[j] = w_l.x[j]
    rmses4[i] = testOLERegression(w_l_1,Xtest_i,ytest)
    rmses4_training[i] = testOLERegression(w_l_1,X_train_i,y)
    i = i + 1

plt.plot(lambdas,rmses4)
plt.plot(lambdas,rmses4_training)
plt.show()


# Problem 5
print("STARTING PROBLEM 5---------------------------------------")

pmax = 7
lambda_opt = lambdas[np.argmin(rmses4)]
rmses5 = np.zeros((pmax,2))
for p in range(pmax):
    Xd = mapNonLinear(X[:,2],p)
    Xdtest = mapNonLinear(Xtest[:,2],p)
    w_d1 = learnRidgeERegression(Xd,y,0)
    rmses5[p,0] = testOLERegression(w_d1,Xdtest,ytest)
    w_d2 = learnRidgeERegression(Xd,y,lambda_opt)
    rmses5[p,1] = testOLERegression(w_d2,Xdtest,ytest)
    
plt.plot(range(pmax),rmses5)
plt.show()
plt.legend(('No Regularization','Regularization'))
'''