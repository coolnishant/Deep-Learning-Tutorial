# coding: utf-8
import numpy as np

#Function for initializing random weights
def init_weights():
    W = np.random.rand(784)
    b = 0
    return W, b

def step(z):
    return 1 * (z > 0)

def propagate(W,b,X,Y):
    A = step(np.dot(X,W.T) + b)
    Error = Y - A
    return Error

def optimize(W,b,X,Y,num_iterations, learning_rate):
    for i in range(num_iterations):
        Error = propagate(W,b,X,Y)
#        print ("Error is {} for iteration {}".format(np.mean(Error), i+1))
        W = W + np.multiply(learning_rate , (np.dot(Error.T,X))) 
        b = b + np.multiply(learning_rate ,np.mean( Error))
    params = {"W" : W, "b" : b}
    return params

def model(X,Y,num_iterations,learning_rate):
    W, b = init_weights()
    #print ("Initial W is {} and b is {}".format(W,b))
    params = optimize(W,b,X,Y,num_iterations, learning_rate)
    #print ("Final updated W and b are")
    #print ("W is" ,params["W"])
    #print ("b is ",params["b"])
    
    ni = params["W"]
    ni = ni.reshape((28,28))
    
    import matplotlib.pyplot as plt
    
    plt.imshow(ni, cmap='gray')
    plt.show()
    
    #return params['W'],params['b']    
    return params

def validation(X,params):
    W = params["W"]
    b = params["b"]
    A = step(np.dot(X,W.T) + b)
    return  A


from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print('Training Shape: ',x_train.shape)
print('Test Shape: ',x_test.shape)
#print(y_train)
#print(y_test)

#import matplotlib.pyplot as plt
#matplotlib_inline
#plt.imshow(x_train[100],cmap='gray')
for i in range(0,10):
    y_train_i= (y_train==i)
    y_test_i = (y_test==i) 
    
#    print(y_test_i.shape)
    x_test = x_test.reshape(10000,784)
#    print(x_test.shape)
    y_train_i.shape
#    print(y_train_i.shape)
    x_train=x_train.reshape(60000,784)
    x_train.shape
#    print(x_train.shape)
    
    def mymain():
        learning_rate = 0.2
        num_of_interations = 10
        params = model(x_test ,y_test_i ,num_of_interations,learning_rate)
        return params
    
    params = mymain()
    
    yhat = validation(x_test,params)
    acc = np.mean(y_test_i==yhat)
    print ("Single perceptron accuracy is {}%".format(acc*100))
    
    for i in range(10):
        
        yhat = validation(x_test[i],params)
        acc = np.mean(y_test_i[i]==yhat)
#        print (str(i+1)+". Original Data "+str(y_test_i[i])+" Model Output "+str(yhat))
    
    yhat = validation(x_test,params)
    acc = np.mean(y_test_i==yhat)
#    print ("Single perceptron accuracy is {}%".format(acc*100))

#2 Miss Classification