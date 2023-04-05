import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

u=0
k=1.6222
tao=0.3699
delta=0.1
t=0
z=[]
y=0
tt=[]
uu=[]


while t<=180:
      
      if t>=20:
         u=40
      if t>=40:
         u=80   
      if t>=60:
         u=100
      if t>=80:
         u=100
      if t>=100:
         u=75 
      if t>=120:
         u=25
      if t>=160:
         u=0
      y =((((u*k)-y)*delta)/tao)+y
      #y =((((u*delta*k)/tao)+y)*(1/(1+delta/tao)))
      #print(y)
      uu.append(u)
      z.append(y)
      tt.append(t)
      t=t+1
      print(y)

           
zmax=max(z)
zmin=min(z)
umax=max(uu)
umin=min(uu)

normu=[]
normz=[]
for i in range(len(z)):
    nz=(z[i]-zmin)/(zmax-zmin)
    normz.append(nz)
    nu=(uu[i]-umin)/(umax-umin)
    normu.append(nu)
    
plt.plot(tt,z, color="red")
plt.show()
planta=[uu,z]
planta=np.transpose(planta)

uk_1=normu[1:len(normu)-1]
uk_2=normu[0:len(normu)-2]
yk_1=normz[1:len(normz)-1]
yk_2=normz[0:len(normz)-2]
yk=normz[2:]

matriz=np.zeros((len(z)-2,5))
matriz[:, 0]=uk_1
matriz[:, 1]=uk_2
matriz[:, 2]=yk_1
matriz[:, 3]=yk_2
matriz[:, 4]=yk

np.savez("matriz", matriz)
