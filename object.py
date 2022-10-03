import numpy as np
np.random.seed(0)

X = [[1.0 ,2.0 ,3.0 ,2.5],
            [2.0,5.0,-1.0,2.0],
            [-1.5,2.7,3.3,-0.8]] 

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights=0.1*np.random.randn(n_inputs,n_neurons)
        self.baises=np.zeros((1,n_neurons))
    #keep weights low we want weights to stay between -1 and 1
    def forward(self,inputs):
        self.output=np.dot(inputs,self.weights)+self.baises
       
class Activation_ReLU:
    def forward(self, inputs)
        self.output=np.maximum(inputs,0)
    

layer1=Layer_Dense(4,5)
layer2=Layer_Dense(5,2)
#print(X)
layer1.forward(X)
#print(layer1.output)
ayer2.forward(layer1.output)
print(layer2.output)

