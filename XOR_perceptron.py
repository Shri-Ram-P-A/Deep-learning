import numpy as np
import tensorflow as tf

x = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([[0],[1],[1],[0]])

model  = tf.keras.Sequential([
    tf.keras.layers.Dense(2,input_dim = 2,activation='sigmoid'),
    tf.keras.layers.Dense(1,activation='sigmoid')
])

model.compile(loss="binary_crossentropy",optimizer = "adam", metrics = ['accuracy','recall'])

model.fit(x,y,epochs = 1299)

met = model.evaluate(x,y)
print("Loss :",met[0],"Accuracy :",met[1])
print("Prediction :",np.round(model.predict(x)))
