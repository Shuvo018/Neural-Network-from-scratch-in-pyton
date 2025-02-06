import numpy as np
inputs = [1, 2, 3, 2.5]
weights = [[0.2, 0.8, -0.5, 1.0],
            [0.5, -0.91, 0.26, -0.5],
            [-0.26, -0.27, 0.17, 0.87]]
biases = [2, 3, 0.5]

layer_outputs = np.dot(weights, inputs) + biases
print(layer_outputs)

# how dot product works
"""
For simple example of dot product,

inputs = [1, 2, 3, 2.5]
weights = [0.2, 0.8, -0.5, 1.0]
np.dot(weights, inputs) = 0.2*1 + 0.8*2 + -0.5*3 + 1.0*2.5 = 2.8

----------------------------------
----------------------------------

For 2D weights and inputs,

np.dot(weights, inputs) = 
[np.dot(weights[0], inputs),
 np.dot(weights[1], inputs),
 np.dot(weights[2], inputs)]

 = [(0.2*1 + 0.8*2 + -0.5*3 + 1.0*2.5),
    (0.5*1 + -0.91*2 + 0.26*3 + -0.5*2.5),
    (-0.26*1 + -0.27*2 + 0.17*3 + 0.87*2.5)]
 = [ 2.8   -1.79   1.885]
"""