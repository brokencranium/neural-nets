import tensorflow as tf
import numpy as np
from tensorflow.keras import layers

if __name__ == '__main__':
    model = tf.keras.Sequential([layers.Dense(units=1, input_shape=[1])])
    model.compile(optimizer='sgd', loss='mse')
    xs = np.array([-1, 0, 1, 2, 3, 4], dtype=float)
    ys = np.array([-3, -1, 1, 3, 5, 7], dtype=float)
    model.fit(xs, ys, epochs=500)
    print(model.predict([10]))

# Bedroom model
# model = keras.Sequential([
#     keras.layers.Dense(units=1, input_shape=[1], activation='linear')
# ])
# model.compile(optimizer='sgd', loss='mean_squared_error', metrics=['accuracy'])
# xs = np.array([1,2,3,4,5,6,7,8,9,10],dtype=float)
# ys = np.array([.1,.15,.2,.25,.3,.35,.4,.45,.5,.55], dtype=float)
# model.fit(xs,ys,epochs=10)
# print(model.predict([15.0]))
