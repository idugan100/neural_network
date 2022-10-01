import sys
import numpy as np
import matplotlib


inputs = [1 ,2 ,3,2.5]
weights1 = [0.2, 0.8, -0.5,1]
weights2 = [0.5, -.91, .26,-.5]
weights3 = [-.26, -0.27,.17,.87]
bias1=2
bias2=3
bias3=0.5

output1=np.dot(inputs,weights1)+bias1
output2=np.dot(inputs,weights2)+bias2
output3=np.dot(inputs,weights3)+bias3
outputs=[output1,output2,output3]
print(outputs)
