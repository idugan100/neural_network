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
biases=[2,3,.5]


## transpose weights so matrix multiplication works each row of the result is
#the set of outputs from the corresponding input
output=np.dot(inputs,np.array(weights).T)+biases
print(output)

    

