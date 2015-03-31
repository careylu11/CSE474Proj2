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
    # initalize k matrices for k classes

    '''
    ones = np.array([0,0])
    twos = np.array([0,0])
    threes = np.array([0,0])
    fours = np.array([0,0])
    fives = np.array([0,0])

    # create k matrices
    for i in range(0,y.shape[0]):
     if(y[i] == 1):
      ones = np.vstack((ones,X[i]))
     elif(y[i] == 2):
      twos = np.vstack((twos,X[i]))
     elif(y[i] == 3):
      threes = np.vstack((threes,X[i]))
     elif(y[i] == 4):
      fours = np.vstack((fours,X[i]))
     else:
      fives = np.vstack((fives,X[i]))

    # hacky delete initialization values
    ones = np.delete(ones,0,0)
    twos = np.delete(twos,0,0)
    threes = np.delete(threes,0,0)
    fours = np.delete(fours,0,0)
    fives = np.delete(fives,0,0)

    onesrow = [ones[:,0].mean(),ones[:,1].mean()]
    twosrow = [twos[:,0].mean(),twos[:,1].mean()]
    threesrow = [threes[:,0].mean(),threes[:,1].mean()]
    foursrow = [fours[:,0].mean(),fours[:,1].mean()]
    fivesrow = [fives[:,0].mean(),fives[:,1].mean()]

    print(onesrow)
    print(twosrow)
    print(threesrow)
    print(foursrow)
    print(fivesrow)

    means = np.array(onesrow)
    #np.transpose(means)
    print("means", means)'''

    X = np.hstack((X,y))
    X.sort()
    covmat = []
    means = []
    print(X)
    # IMPLEMENT THIS METHOD

    return means,covmat

def qdaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmats - A list of k d x d learnt covariance matrices for each of the k classes

    # IMPLEMENT THIS METHOD
    means = []
    covmats = []
    return means,covmats

def ldaTest(means,covmat,Xtest,ytest):
    # Inputs
    # means, covmat - parameters of the LDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    acc = 0
    # IMPLEMENT THIS METHOD
    return acc

def qdaTest(means,covmats,Xtest,ytest):
    # Inputs
    # means, covmats - parameters of the QDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    means = []
    covmats = []
    acc = 0
    # IMPLEMENT THIS METHOD
    return acc

def learnOLERegression(X,y):
    # Inputs:
    # X = N x d
    # y = N x 1
    # Output:
    # w = d x 1
    w = np.dot(np.linalg.inv(np.dot(X.T,X)),np.dot(X.T,y))
    # IMPLEMENT THIS METHOD
    return w

def learnRidgeRegression(X,y,lambd):
    # Inputs:
    # X = N x d
    # y = N x 1
    # lambd = ridge parameter (scalar)
    # Output:
    # w = d x 1
    w = np.dot(np.linalg.inv(((lambd*np.identity(X.shape[1])) + np.dot(np.transpose(X),X))),np.dot(np.transpose(X),y))
    return w

def testOLERegression(w,Xtest,ytest):
    # Inputs:
    # w = d x 1
    # Xtest = N x d
    # ytest = X x 1
    # Output:
    N = Xtest.shape[0]
    m1 = np.transpose(ytest - np.dot(Xtest,w))
    m2 = ytest - np.dot(Xtest,w)
    M = np.sqrt(np.dot(m1,m2))
    rmse = M/N
    print(rmse)
    return rmse

def regressionObjVal(w, X, y, lambd):

    # compute squared error (scalar) and gradient of squared error with respect
    # to w (vector) for the given data X and y and the regularization parameter
    # lambda
    # IMPLEMENT THIS METHOD

    N = X.shape[0]
    A = np.dot(X, w)
    B = pow((y - A), 2)
    C = np.dot(lambd, (np.dot(w.transpose(), w)))
    error = (np.sum(B)+C)/N

    deviation = np.dot(X, w) - y
    A = (2/N) * np.dot(deviation.transpose(), X)
    B = np.dot(lambd, w.transpose())
    error_grad = A + B
    error_grad = error_grad.transpose()

    print("error is: ", error)
    print("error_grad is: ", error_grad)
    return error_grad, error

def mapNonLinear(x,p):
    # Inputs:
    # x - a single column vector (N x 1)
    # p - integer (>= 0)
    # Outputs:
    # Xd - (N x (d+1))
    # IMPLEMENT THIS METHOD
    Xd = np.ones(x.shape[0], p + 1)
    for i in range(p):
        Xd[i, ] = pow(x, i)

    return Xd.conj().transpose()

# # Main script
# # Problem 1
# # load the sample data
# X,y,Xtest,ytest = pickle.load(open('sample.pickle', 'rb'), encoding = 'latin1')
# # LDA
# means,covmat = ldaLearn(X,y)
# ldaacc = ldaTest(means,covmat,Xtest,ytest)
# print('LDA Accuracy = '+str(ldaacc))
# # QDA
# means,covmats = ldaLearn(X,y)
# qdaacc = ldaTest(means,covmats,Xtest,ytest)
# print('QDA Accuracy = '+str(qdaacc))


# Problem 2

X,y,Xtest,ytest = pickle.load(open('diabetes.pickle', 'rb'), encoding = 'latin1')# add intercept
X_i = np.concatenate((np.ones((X.shape[0],1)), X), axis=1)
Xtest_i = np.concatenate((np.ones((Xtest.shape[0],1)), Xtest), axis=1)

w = learnOLERegression(X,y)
mle = testOLERegression(w,Xtest,ytest)

w_i = learnOLERegression(X_i,y)
mle_i = testOLERegression(w_i,Xtest_i,ytest)

print('RMSE without intercept '+str(mle))
print('RMSE with intercept '+str(mle_i))

# Problem 3
k = 21
lambdas = np.linspace(0, 1.0, num=k)
i = 0
rmses3 = np.zeros((k,1))
for lambd in lambdas:
    w_l = learnRidgeRegression(X_i,y,lambd)
    rmses3[i] = testOLERegression(w_l,Xtest_i,ytest)
    i = i + 1
plt.plot(lambdas,rmses3)


# Problem 4
lambdas = np.linspace(0, 1.0, num=k)
k = 21
i = 0
rmses4 = np.zeros((k,1))
opts = {'maxiter' : 50}    # Preferred value.
w_init = np.zeros((X_i.shape[1],1))
for lambd in lambdas:
    args = (X_i, y, lambd)
    w_l = minimize(regressionObjVal, w_init, jac=True, args=args,method='CG', options=opts)
    rmses4[i] = testOLERegression(w_l,Xtest_i,ytest)
    i = i + 1
plt.plot(lambdas,rmses4)


# Problem 5
pmax = 7
lambda_opt = lambdas[np.argmax(rmses4)]
rmses5 = np.zeros((pmax,2))
for p in range(pmax):
    Xd = mapNonLinear(X[:,2],p)
    Xdtest = mapNonLinear(Xtest[:,2],p)
    w_d1 = learnRidgeRegression(Xd,y,0)
    rmses5[p,0] = testOLERegression(w_d1,Xdtest,ytest)
    w_d2 = learnRidgeRegression(Xd,y,lambda_opt)
    rmses5[p,1] = testOLERegression(w_d2,Xdtest,ytest)
plt.plot(range(pmax),rmses5)
plt.legend('No Regularization','Regularization')
