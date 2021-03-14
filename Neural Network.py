# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 15:08:03 2021

@author: d-tje
"""
import random
import numpy as np
import matplotlib.pyplot as plt



class Network():
    
    def __init__(self,sizes,seed=None):
        if seed:
            np.random.seed(102)
        self.layers = len(sizes)
        self.Nodes = sizes
        self.biases = [np.random.randn(y) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
        
    def Forward(self,a):
        for b,w in zip(self.biases,self.weights):
            if len(b)>1:
                a = sigmoidFunction(np.dot(w,a)+b)
            else:
                a = (np.dot(w,a)+b)
        return a
        
    def Evaluate(self,data,accuracy1=None):
        result = [(self.Forward(x),y) for x,y in data]
        sums = sum((x-y)**2 for (x,y) in result)
        sum1 = sum(1-abs(x-y)/y for (x,y) in result)
        accuracy = (1/len(result)*sum1)*100
        averagesums = (1/len(result)*sums)
        if accuracy1:
            return averagesums, result, accuracy
        else:
            return averagesums, result
    
    
    
    def Update_mini_batch(self,batch,eta):
        delta_b = [np.zeros(b.shape) for b in self.biases]
        delta_w = [np.zeros(w.shape) for w in self.weights]
        for x,y in batch:
            change_delta_b, change_delta_w = self.BackProp(x, y)
            delta_b = [b+db for b,db in zip(delta_b,change_delta_b)]
            delta_w = [w+dw for w,dw in zip(delta_w,change_delta_w)] 
        self.weights = [weight - (eta/len(batch))*nw for weight,nw in zip(self.weights,delta_w)]
        self.biases = [bias - (eta/len(batch))*nb for bias,nb in zip(self.biases,delta_b)]

        
    def BackProp(self,x,y):
        delta_b = [np.zeros(b.shape) for b in self.biases]
        delta_w = [np.zeros(w.shape) for w in self.weights]
        a = x
        activations = [x]
        zs = []
        for b,w in zip(self.biases,self.weights):
            z = np.dot(w,a) + b
            zs.append(z)
            if len(b) > 1:
                a = sigmoidFunction(z)
                activations.append(a)
            else:
                a = z
                activations.append(a)
        delta = cost_derivative(activations[-1], y)
        delta_b[-1] = delta
        delta_w[-1] = np.outer(delta,activations[-2])
        for l in range(2,self.layers):
             z = zs[-l]
             sp = sigmoid_derivative(z)
             delta = np.dot(self.weights[-l+1].transpose(),delta) *sp
             delta_b[-l] = delta
             delta_w[-l] = np.outer(delta,activations[-l-1])
        return delta_b,delta_w
    
    def StochasticGD(self,trainingData,epochs,batchsize,eta,testdata=None):
        n = len(trainingData)
        self.trainingprogresssion = []
        self.testprogression = []
        for i in range(1,epochs):
            random.shuffle(trainingData)
            minibatches = [trainingData[k:k+batchsize] for k in range(0,n,batchsize)]
            for minibatch in minibatches:
                self.Update_mini_batch(minibatch, eta)
            if testdata:
                mse,result = self.Evaluate(testdata)
                self.testprogression.append(mse)
                mse1,result1 = self.Evaluate(trainingData)
                self.trainingprogresssion.append(mse1)
                print("Epoch {0} MseTraining : {1} MseTest: {2}".format(i,mse1,mse))
            else:
                mse,result = self.Evaluate(trainingData)
                self.trainingprogresssion.append(mse)
                print("Epoch: {0} Mse: {1}".format(i,mse))
        

def sigmoid_derivative(x):
    return sigmoidFunction(x)*(1-sigmoidFunction(x))

def cost_derivative(activation,y):
    return 2*(activation - y)
        
def sigmoidFunction(x):
    return 1/(1 + np.exp(-x))

def DataExtraction(filename,delete=True):
    data = np.loadtxt(open(filename, "rb"), delimiter=",", skiprows=1)
    if delete:
        data = np.delete(data,0,1)
    realData = []
    for i in range(len(data)):
        realData.append((data[i,:-1],data[i,-1]))
    return realData

def PlotNetworkResults(Results):
    list1 = []
    list2 = []
    for (x,y) in Results:
        list1.append(float(x))
        list2.append([y])
    plt.plot(list2,label='actual value')
    plt.plot(list1,label='prediction')
    plt.legend()
    plt.show()

def PlotMSEProgression(Network):
    x = Network.testprogression
    y = Network.trainingprogresssion
    plt.title('Test MSE')
    plt.plot(x)
    plt.show()
    plt.title("Training MSE")
    plt.plot(y)
    plt.show()

def FindLearningRate(Data):
    scale = np.ones(6)
    result = []
    for i in range(len(scale)):
        scale[i] = scale[i]*(10**(-i-1))
    for j in scale:
        net = Network([5,5,1],True)
        net.StochasticGD(Data,100,20,j)
        mse,predictions = net.Evaluate(Data)
        result.append([int(mse)])
    return scale[np.argmin(result)]
    
net = Network([5,20,1],True)
TrainingData = DataExtraction("trainingdata2.csv")
TestData = DataExtraction("testdata2.csv")

# =============================================================================
# d = FindLearningRate(TrainingData)
# =============================================================================


net.StochasticGD(TrainingData, 2, 10, 0.00001,TestData)
mse,predictions, accuracy = net.Evaluate(TestData,True)

c = net.weights

for i in c:
    for j in i:
        for d in j:
            print(d)


PlotNetworkResults(predictions)
PlotMSEProgression(net)



