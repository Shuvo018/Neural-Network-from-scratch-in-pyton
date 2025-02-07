inputs = [1, 2, 3, 2.5]
weights = [[0.2, 0.8, -0.5, 1.0],
            [0.5, -0.91, 0.26, -0.5],
            [-0.26, -0.27, 0.17, 0.87]]
biases = [2, 3, 0.5]

layer1_output = [
    # neuron 1
    inputs[0]*weights[0][0] +
    inputs[1]*weights[0][1] +
    inputs[2]*weights[0][2] +
    inputs[3]*weights[0][3] + biases[0],

    # neuron 2
    inputs[0]*weights[1][0] +
    inputs[1]*weights[1][1] +
    inputs[2]*weights[1][2] +
    inputs[3]*weights[1][3] + biases[1],

    # neuron 3
    inputs[0]*weights[2][0] +
    inputs[1]*weights[2][1] +
    inputs[2]*weights[2][2] +
    inputs[3]*weights[2][3] + biases[2]
]
print('layer1_output')
print(layer1_output)

weights2 = [[0.1, -0.14, 0.5],
            [-0.5, 0.12, -0.33]]
biases2 = [-1, -0.5]

layer2_output = [
    # neuron 4
    layer1_output[0]*weights2[0][0] +
    layer1_output[1]*weights2[0][1] +
    layer1_output[2]*weights2[0][2] + biases2[0],

    # neuron 5
    layer1_output[0]*weights2[1][0] +
    layer1_output[1]*weights2[1][1] +
    layer1_output[2]*weights2[1][2] + biases2[1]
]
print('layer2_output')
print(layer2_output)

weight3 = [0.1, -0.14]
bias3 = [0.5]

layer3_output = [
    # neuron 6
    layer2_output[0]*weight3[0] +
    layer2_output[1]*weight3[1] + bias3[0]
]

print('layer3_output')
print(layer3_output)