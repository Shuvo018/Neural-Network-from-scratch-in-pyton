from nnfs.datasets import spiral_data
import matplotlib.pyplot as plt
import numpy as np
import nnfs

nnfs.init()

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases        
class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

X, y = spiral_data(samples=100, classes=3)
# print(X)
# plt.scatter(X[:,0], X[:,1], c=y, cmap='brg')
# plt.show()

dense1 = Layer_Dense(2, 3)
activation1 = Activation_ReLU()
dense1.forward(X)

print(dense1.output[:5])

activation1.forward(dense1.output)
print('actionvation')
print(activation1.output[:5])