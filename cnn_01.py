from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

# Memuat dataset MNIST
(x_train, y_train), (x_valid, y_valid) = mnist.load_data()

# print(x_train.shape)
# print(x_valid.shape)
# # Mengambil gambar pertama dari x_train
# image = x_train[0]

# # Menampilkan gambar menggunakan plt.imshow
# plt.imshow(image, cmap='gray')
# plt.show()

# creating the model
# install the model
from tensorflow.keras.models import Sequential

model = Sequential()

# input layer
from tensorflow.keras.layers import Dense
model.add(Dense(units=512, activation='relu', input_shape=(784,)))

# creating hidden layer
model.add(Dense(units = 512, activation='relu'))


# creating hidden layer
model.add(Dense(units = 10, activation='softmax'))

model.summary()

# compiling 
model.compile(loss='categorical_crossentropy', metrics=['accuracy'])

# training
history = model.fit(
    x_train, y_train, epochs=5, verbose=1, validation_data=(x_valid, y_valid)
)

# clear the memory
import IPython
app = IPython.Application.instance()
app.kernel.do_shutdown(True)
