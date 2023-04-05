# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 08:48:13 2023

@author: jgall
"""

#Se importan dependencias
import numpy as np
from keras.models import model_from_json
import matplotlib.pyplot as plt
from matplotlib import cm

# cargar json y crear el modelo
json_file = open('model.h5', 'r')
loaded_model_json = json_file.read()
json_file.close()

loaded_model = model_from_json(loaded_model_json)
# cargar pesos al nuevo modelo
loaded_model.load_weights("model.h5")
print("Cargado modelo desde disco.")

# Compilar modelo cargado y listo para usar.
loaded_model.compile(loss='binary_crossentropy',
optimizer='adam',
metrics=['accuracy'])
xtotal=[]
ytotal=[]

#Pares de vectores para las combinaciones de entradas
vecx=np.arange(-1.5, 1.5, 0.1)
vecy=np.arange(-1.5, 1.5, 0.1)
for x2 in range (30):
    yt=[]
#Se crean las parejas en entradas y se evalúa el modelo
    for x1 in range(30):
        vec=vecx[x1],vecy[x2]
        vec=np.array(vec)
        vec= vec[np.newaxis]
        xtotal.append(vec)
        yf=loaded_model.predict(vec)
        yt.append(float(np.array(yf)))
        ytotal.append(np.array(yt))
        print(x2+1)
        
#Se genera grafica 3D
vecx,vecy= np.meshgrid(vecx,vecy)
ytotal=np.array(ytotal)
fig= plt.figure(1)
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(vecx, vecy, ytotal,cmap=cm.coolwarm,rstride=1, cstride=1)
ax.set_zlim(-1.01,1.01)
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.show()