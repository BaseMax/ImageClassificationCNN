import random
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten

X_train = np.loadtxt('input.csv', delimiter = ",")
Y_train = np.loadtxt('labels.csv', delimiter = ",")

X_test = np.loadtxt('input_test.csv', delimiter = ",")
Y_test = np.loadtxt('labels_test.csv', delimiter = ",")

X_train = X_train.reshape(len(X_train), 100, 100, 3)
Y_train = Y_train.reshape(len(Y_train), 1)
X_test = X_test.reshape(len(X_test), 100, 100, 3)
Y_test = Y_test.reshape(len(Y_test), 1)

X_train = X_train / 255.0
X_test = X_test / 255.0

print('Shape of X_train: ', X_train.shape)
print('Shape of Y_train: ', Y_train.shape)
print('Shape of X_test: ', X_test.shape)
print('Shape of Y_test: ', Y_test.shape)

print(X_train[1,:])

idx = random.randint(0, len(X_train))
plt.imshow(X_train[idx, :])
plt.show()

model = Sequential(

)