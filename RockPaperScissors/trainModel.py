import os
from sklearn import model_selection
from tensorflow import get_logger
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D
import pickle
get_logger().setLevel('INFO')

path = os.path.join(os.path.abspath(os.path.join(os.getcwd(), os.pardir)), 'Data')

with open(os.path.join(path, 'X.pickle'), 'rb') as f1:
    X = pickle.load(f1)
with open(os.path.join(path, 'y.pickle'), 'rb') as f2:
    y = pickle.load(f2)

X = X.reshape((-1, 100, 100, 3))

x_train, x_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

convLayers = 3
convLayer_size = 64

model = Sequential()
model.add(Conv2D(64, (3, 3), input_shape=X.shape[1:]))
model.add(Activation('relu'))

for i in range(convLayers - 1):
    model.add(Conv2D(convLayer_size, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))

model.add(Flatten())
model.add(Dense(4))
model.add(Activation('softmax'))

model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=32, epochs=8, validation_split=0.1)
loss, acc = model.evaluate(x_test, y_test)

print(f'Loss={loss}, acc={acc}')

model.save(os.path.join(path, 'model'))
