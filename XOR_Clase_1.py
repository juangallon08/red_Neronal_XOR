# -*- coding: utf-8 -*-
"""
control inteligente clase 1
solucion de la XOR 

@author: jgall
"""

import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense

# cargamos las 4 combinaciones de las compuertas XOR
training_data = np.array([[0,0],[0,1],[1,0],[1,1]], "float32")
# y estos son los resultados que se obtienen, en el mismo orden
target_data = np.array([[0],[1],[1],[0]], "float32")

model = Sequential()
model.add(Dense(16, input_dim=2, activation='relu'))
model.add(Dense(32, activation='tanh'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='mean_squared_error',
optimizer='adam',
metrics=['binary_accuracy'])

model.fit(training_data, target_data, epochs=1000)

# evaluamos el modelo
scores = model.evaluate(training_data, target_data)

print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
print (model.predict(training_data).round())

salida= model.predict(training_data)
error= np.sqrt(np.sum(np.square(target_data-salida)))/4
print("Error\n",error)


model.save("model41.h5")

