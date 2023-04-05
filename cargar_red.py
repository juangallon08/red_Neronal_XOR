import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense
from keras.models import load_model
import matplotlib.pyplot as plt
from matplotlib import cm

array=np.load("matriz.npz")
matriz=array['arr_0']

entrada=(matriz[:,0:4])
salida=(matriz[:,4])

training_data = entrada
target_data = salida


# cargar json y crear el modelo
loaded_model = load_model('modelo1.h5')
loaded_model.predict(training_data)

xtotal=[]
ytotal=[]

#Pares de vectores para las combinaciones de entradas
vecx=np.arange(-2.0, 2.0, 0.1)
vecy=np.arange(-2.0, 2.0, 0.1)

for x2 in range (40):
    yt=[]
    for x1 in range(40):
        vec=vecx[x1],vecy[x2]
        vec=np.array(vec)
        vec= vec[np.newaxis]
        xtotal.append(vec)
        yf=loaded_model.predict(vec)
        yt.append(float(np.array(yf)))
    ytotal.append(np.array(yt))
        

#Se genera grafica 3D
vecx,vecy= np.meshgrid(vecx,vecy)
ytotal=np.array(ytotal)
fig= plt.figure(1)
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(vecx, vecy, ytotal,cmap=cm.coolwarm,rstride=1, cstride=1)
ax.set_zlim(-2.01,2.01)
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.show()