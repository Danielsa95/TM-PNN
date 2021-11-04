import NN_structure as nn
import numpy as np
import matplotlib.pyplot as plt
import helper_fun as tmp


### Coefficients and evaluating the matrices of weights
P0=np.array([[0],[0]])
P1=np.array([0,1,-2,0]).reshape(2,2)
P2=np.array([0,1,0,0,-1,0]).reshape(2,3)
P3=np.zeros(8).reshape(2,4)
coeff=(P0,P1,P2,P3)
Tf=5
dt=Tf/100
time=np.arange(0,Tf+dt,dt)
x0=np.array([[0.8,0.8]])
x1=np.array([[0.5,0.5]])   ### initial point

#### Creating the PNN
pnn=nn.PNN(coeff, dt, 2)  ### The weights are just inizialized

### evaluation of the solution by TM-PNN
x_0, y_0 = nn.pred(int(Tf/dt),pnn,x0)         #### sol wrt the initial condition x0
x_1, y_1 = nn.pred(int(Tf/dt),pnn,x1)         #### sol wrt the initial condition x1

#### Real solution --> RK45
data=np.loadtxt("LV_05.dat")             ### this is the solution related to x0
te_mpo_1=data[:,0]
rk_x1=data[:,1]
rk_y1=data[:,2]
data_2=np.loadtxt("LV_08.dat")             ### this is the sol related to x1
te_mpo_2=data_2[:,0]
rk_x2=data_2[:,1]
rk_y2=data_2[:,2]


##### plot

figure,axis=plt.subplots(2,2)
axis[0,0].plot(time,x_0,'r-')
axis[0,0].set_ylabel('x')
axis[0,0].plot(time,x_1,'b-')
axis[0,0].set_title('Plot of the PNN solutions')
axis[0,1].plot(te_mpo_1,rk_x1,'r--')
axis[0,1].plot(te_mpo_2,rk_x2,'b--')
axis[0,1].set_title('Plot of the RK solutions')
axis[1,0].plot(x_0,y_0,'r-')
axis[1,0].plot(x_1,y_1,'b-')
axis[1,0].set_ylabel('y')
axis[1,0].set_xlabel('x')
axis[1,1].plot(rk_x1,rk_y1,'r--')
axis[1,1].set_xlabel('x')
axis[1,1].plot(rk_x2,rk_y2,'b--')
figure.show()

#### evaluation of the norm
index=tmp.find_index(time,te_mpo_1)  ## i know how to find the values in data
index_2=tmp.find_index(time,te_mpo_2)


# conversion of x_m e y_m from list to array
tmp1 = np.zeros(len(x_0))
tmp2 = np.zeros(len(y_0))
for i in range(len(x_0)):
    tmp1[i]=x_0[i]
    tmp2[i]=y_0[i]
x_0=tmp1
y_0=tmp2

# conversion of x_p e y_p from list to array
tmp1 = np.zeros(len(x_1))
tmp2 = np.zeros(len(y_1))
for i in range(len(x_1)):
    tmp1[i] = x_1[i]
    tmp2[i] = y_1[i]
x_1 = tmp1
y_1 = tmp2


norma_x_0 = np.linalg.norm(x_0[index]-rk_x1) ### ||sol_1 -RK||_2
norma_y_0 = np.linalg.norm(y_0[index]-rk_y1) ###

norma_x_1 = np.linalg.norm(x_1[index_2]-rk_x2)  ### ||sol_2 -RK||_2
norma_y_1 = np.linalg.norm(y_1[index_2]-rk_y2)  ###


print("########### Norm in x and y with the real sol and x0 as initial point  ################")
print(norma_x_0 , norma_y_0)
print("########### Norm in x and y with the real sol and x1 as initial point  ################")
print(norma_x_1 , norma_y_1)



