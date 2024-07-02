import numpy as np
import tensorflow as tf

x = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([[0],[1],[1],[0]])

model  = tf.keras.Sequential([
    tf.keras.layers.Dense(2,input_dim = 2,activation='sigmoid'),
    tf.keras.layers.Dense(1,activation='sigmoid')
])

model.compile(loss="binary_crossentropy",optimizer = "adam", metrics = ['accuracy','recall','precision'])

model.fit(x,y,epochs = 1299)

print(model.evaluate(x,y))

