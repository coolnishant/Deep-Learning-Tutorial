#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 11:14:35 2019

@author: nish
"""

import numpy as np
from sklearn import datasets

import math

def sigmoid(x):
  return 1 / (1 + math.exp(-x))
sigmoid_v = np.vectorize(sigmoid)

def feed_forward(model, x):
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    # Forward propagation
#    print(x.dot(W1))
    a1 = x.dot(W1) + b1
#    print(z1)
    y1 = sigmoid_v(a1)
    a2 = a1.dot(W2) + b2
    out = sigmoid_v(a2)
    return a1, y1, a2, out

def tanh_deriv(x):
    return 1.0 - np.tanh(x)**2

def backprop(X,y,model,a1,y1,a2,output):
    delta3 = (y-output).dot((output.T).dot(1-output))
#    print(dz2.shape)
    dW2 = (y1.T).dot(delta3)
    
    db2 = np.sum(delta3, axis=0)
    
    delta2 = delta3.dot((model['W2'].T).dot((y1.T).dot(1-y1)))
    
    dW1 = np.dot(X.T, delta2)
    db1 = np.sum(delta2, axis=0)
#    print(db1.shape)
    return dW1, dW2, db1, db2

def calculate_loss(model,X,Y):
    num_examples = X.shape[0]

    a1, y1, a2, out = feed_forward(model, X)
#    for i in range(0,num_examples):
#        print(Y[i],out[i])
#    print('shape of Y is ',Y.shape,'\nshape of out is ',out.shape)
    mse = np.square(Y - out)
#    .mean(axis=1) #axis 1 is row wise mean.
    #axis 0 is coloumn wise mean.

    loss = np.sum(mse)
#    print('loss is ',loss)
    return loss/num_examples


def train(model, X, Y,learning_rate):
    previous_loss = float('inf')
    i = 0
    losses = []
    #maximum 10 epoch
    while i<10:
        a1,y1,a2,output = feed_forward(model, X)
        #backpropagation
        dW1, dW2, db1, db2 = backprop(X,Y,model,a1,y1,a2,output)
#        print('Shape of W2: {} dW2: {}'.format(model['W2'].shape,dW2.shape))
#        print('Shape of W1: {} dW1: {}'.format(model['W1'].shape,dW1.shape))
        #update weights and biases
        model['W1'] -= learning_rate * dW1
        model['b1'] -= learning_rate * db1
        model['W2'] -= learning_rate * dW2
        model['b2'] -= learning_rate * db2
        loss = calculate_loss(model, X, Y)
        losses.append(loss)
        i+=1;
        print ("Loss after iteration {}: {}".format(i, loss))  
      
        #if new loss is less than 1% of previous loss then break;
        if(previous_loss-loss) < 0.01*previous_loss:
            break;
            
        previous_loss=loss
        
    return model, losses


def main():
    iris = datasets.load_iris()
    Xo = iris.data
    Yo = iris.target
    
    X = Xo[:40]
    Xtest = Xo[40:50]
    X = X + Xo[50:90]
    Xtest = Xtest + Xo[90:100]
#
#    
    Y = Yo[:40]
    Ytest = Yo[40:50]
    Y = Y + Yo[50:90]
    Ytest = Ytest + Yo[90:100]
    
#    X = Xo[:100]
#    Y = Yo[:100]
    
    Y = np.reshape(Y, (Y.shape[0], 1))
    
    print("---------------")

    W1 = np.random.randn(4,2)
    b1 = np.zeros((1,2))
    W2 = np.random.randn(2,1)
    b2 = np.zeros((1,1))

    model = {}
    model['W1'] = W1;
    model['b1'] = b1;
    model['W2'] = W2;
    model['b2'] = b2;
    learning_rate=0.1;

    model, losses = train(model,X, Y, learning_rate);

#    print('The output Losses are',losses);
    
    loss = calculate_loss(model, Xtest, Ytest)
    print('Accuracy % on training set is: ',100-loss)
    print("---------------")


if __name__ == '__main__':
    main()
