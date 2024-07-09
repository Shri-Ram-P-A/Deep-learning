from tensorflow.keras.datasets import mnist
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

(x_train,y_train),(x_test,y_test) = mnist.load_data()

x_train = x_train/255
x_test = x_test/255

ANN_model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape = (28,28)),
    tf.keras.layers.Dense(128,activation='relu'),
    tf.keras.layers.Dense(64,activation = 'relu'),
    tf.keras.layers.Dense(10,activation = 'softmax')]
)

ANN_model.compile(loss="sparse_categorical_crossentropy",optimizer = 'adam',metrics = ['accuracy'])

ANN_model.fit(x_train,y_train,epochs = 100)
t = ANN_model.evaluate(x_test,y_test)
print("Accuracy :",t[1])
print(np.argmax(ANN_model.predict(x_test)[0]))
plt.imshow(x_test[0])
plt.show()
