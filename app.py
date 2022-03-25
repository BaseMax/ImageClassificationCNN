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

# randomly show an image
idx = random.randint(0, len(X_train))
# plt.imshow(X_train[idx, :])
# plt.show()

# model = Sequential([
#     Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)),
#     MaxPooling2D(pool_size=(2, 2)),
    
#     Conv2D(64, (3, 3), activation='relu'),
#     MaxPooling2D(pool_size=(2, 2)),

#     Flatten(),
#     Dense(128, activation='relu'),
#     Dense(1, activation='sigmoid')
# ])
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))

# model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
# model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# opt = keras.optimizers.SGD(learning_rate=0.001)
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

# model.fit(X_train, Y_train, epochs = 10, batch_size = 32)
model.fit(X_train, Y_train, epochs = 5, batch_size = 64)

model.evaluate(X_test, Y_test)

# Making predictions
for i in range(0, 5):
    idx2 = random.randint(0, len(Y_test))
    plt.imshow(X_test[idx2, :])
    plt.show()

    y_pred = model.predict(X_test[idx2, :].reshape(1, 100, 100, 3))
    print(y_pred)
    y_pred = y_pred > 0.5
    print(y_pred)

    if y_pred == False:
        pred = 'dog'
    else:
        pred = 'cat'

    print("Our model predicts that the image is a " + pred)
    print("The actual label is " + str(Y_test[idx2]))
    print("\n")
