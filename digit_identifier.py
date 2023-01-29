import numpy as np
from tensorflow import keras
import scipy.io
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.utils import np_utils
import sklearn
from sklearn.model_selection import train_test_split

# Load the MNIST dataset
mat_file = scipy.io.loadmat('train_32x32.mat')
X = mat_file['X']
y = mat_file['y']

# re-shuffling the dimensions of the features
# changing from X(rows, cols, color channles, samples) to X(samples, rows, cols, color channels)
X = np.transpose(X, (3, 0, 1, 2))
m = X.shape[0]

# re-labeling 10's to 0's in the labels
y = list(map(lambda x: 0 if x==10 else x, y))
y = np.array(y).reshape(m, 1)

# print(X.shape) # check the shape of X
# print(y.shape) # check the shape of y

# Split the datasets
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Reshape the data
X_train = X_train.reshape(X_train.shape[0], 32, 32, 3)
X_test = X_test.reshape(X_test.shape[0], 32, 32, 3)

# Normalize the data
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

# One-hot encode the labels
y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)

# Create the model
model = Sequential()
model.add(Conv2D(70, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='SGD', metrics=['accuracy'])

# Fit the model
model.fit(X_train, y_train, batch_size=250, epochs=20, verbose=1, validation_data=(X_test, y_test))
# test accuracy = 0.7286

# Evaluate the model
score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])