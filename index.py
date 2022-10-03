import sys
import numpy as np
import matplotlib


inputs = [[1.0 ,2.0 ,3.0 ,2.5],
            [2.0,5.0,-1.0,2.0],
            [-1.5,2.7,3.3,-0.8]] 
            #batch inputs to allow for generalizations 
            #of error, but dont add the enitre sample set otherwise it will fit but 
            #not generalize
weights=[[0.2, 0.8, -0.5,1],[0.5, -.91, .26,-.5],[-.26, -0.27,.17,.87]]
biases=[2,3,0.5]
weights2=[
    [0.1, -0.14, 0.5],
    [-0.5, 0.12, -0.33],
    [-.44, 0.73, -0.13]
]
biases2=[-1,2,-0.5]


## transpose weights so matrix multiplication works each row of the result is
#the set of outputs from the corresponding input
layer1_outputs=np.dot(inputs,np.array(weights).T)+biases
layer2_outputs=np.dot(layer1_outputs,np.array(weights2).T)+biases2
print(layer2_outputs)

    

