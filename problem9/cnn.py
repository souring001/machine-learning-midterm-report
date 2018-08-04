import numpy
import keras

from keras.datasets import mnist
from keras import models
from keras import layers
from keras.utils import to_categorical

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

#reshape images to 1D
train_images = train_images.reshape((60000, 28, 28, 1))
train_images = train_images.astype('float32')/255
test_images = test_images.reshape((10000, 28, 28, 1))
test_images = test_images.astype('float32')/255

#translate 4 to [0001000000]
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

print(train_images.shape)
print(train_labels.shape)

#relu is a kind of function
network = models.Sequential()

network.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
network.add(layers.MaxPool2D((2, 2)))
network.add(layers.Conv2D(64, (3, 3), activation='relu'))
network.add(layers.MaxPool2D((2, 2)))
network.add(layers.Conv2D(64, (3, 3), activation='relu'))
network.add(layers.MaxPool2D((2, 2)))
network.add(layers.Flatten())

network.add(layers.Dense(28, activation='relu'))
network.add(layers.Dense(10, activation='softmax'))

#categorical_crossentropy as an error
network.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
network.summary()

#epochs = repeat, but too many, overfitting
network.fit(train_images, train_labels, epochs=5, batch_size=128)

#accuracy for test data
test_loss, test_acc = network.evaluate(test_images, test_labels)
print(test_acc)