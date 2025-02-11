import numpy as np

soft_max = np.array([[0.7, 0.1, 0.2], 
                    [0.1, 0.5, 0.4], 
                    [0.02, 0.9, 0.08]])

class_target = np.array([0, 1, 1])

# for i in range(len(class_target)):
#     print(soft_max[i][class_target[i]])

# short way

for i, distribution in zip(class_target, soft_max):
    print(distribution[i])

# print(soft_max)

# print(soft_max[[0,1,2], [0,1,1]])
print(soft_max[range(len(soft_max)), class_target])

v = soft_max[range(len(soft_max)), class_target]

# batch loss
neg_log = -np.log(soft_max[range(len(soft_max)), class_target])

average_loss = np.mean(neg_log)
print(average_loss)

# Accuracy

predictions = np.argmax(soft_max, axis=1)
print("predictions : ", predictions)
if len(class_target.shape) == 2:
    class_target = np.argmax(class_target, axis=1)
accuracy = np.mean(predictions == class_target)
print('acc:', accuracy)
