import numpy as np
import helper_fun as tmp
import weights as mn
import matplotlib.pyplot as plt
"""
#  In this code there is a comparison between three solutions of the gravity fall problem :
#  1) the solution found using the TM method;
#  2) the solution found using Euler method;
#  3) the solution found using RK45 method (the one of the /pacs-examples/Examples/src/RKFSolver )
"""
m=100       # mass
g=9.81      # acceleration
k=0.392     # resistant coefficient
x0=0        # initial point

## Coefficients of the equation
p0=g
p1=0
p2=-k/m
p3=0

# Time
dt=1e-1
Tf=15
time=np.arange(0,Tf+dt,dt)

# Get the weights
lista=(p0,p1,p2,p3)
W0,W1,W2,W3 = mn.weight(lista,dt,1)

### solution 1)
solution_1=tmp.solve((W0,W1,W2,W3),x0,Tf/dt)

### solution 2)
solution_2=list()
x=x0
solution_2.append(x0)
for i in np.arange(dt,Tf+dt,dt):
    x += g*dt - (dt*k*x**2)/m
    solution_2.append(x)


### real sol
sol= lambda t : np.sqrt(m*g/k) * np.tanh(t * np.sqrt(k*g/m))


### evaluation of the cumulative square error
real_sol = sol(time)
cse_1 = np.zeros(len(time))
cse_2 = np.zeros(len(time))


cse_1[0] = np.linalg.norm(solution_1[0] - real_sol[0])
cse_2[0] = np.linalg.norm(solution_2[0] - real_sol[0])
for i in np.arange(1,len(time),1):
    cse_1[i] = cse_1[i - 1] + np.linalg.norm(solution_1[i] - real_sol[i])
    cse_2[i] = cse_2[i - 1] + np.linalg.norm(solution_2[i] - real_sol[i])


# Evaluating the MSE for different values of dt = [0.1, 0.2, 0.3, ..., 1.]
mse_1,mse_2=tmp.MSE(coef=lista,m=m,Tf=Tf,init=x0)

###############                 PLOT                   ###############
figure,axis=plt.subplots(2,2)

axis[0][0].plot(time,solution_1,'r--')
axis[0][0].set_title('TM-solution')
axis[0][0].set_ylabel('Velocity (m/s)')

axis[0][1].plot(time,solution_2,'b--')
axis[0][1].set_title('Euler solution')

axis[1][0].plot(time,cse_1,'r--')
axis[1][0].set_ylabel('Cumulative square error')
axis[1][0].set_xlabel('Time (s)')

axis[1][1].plot(time,cse_2,'b--')
axis[1][1].set_xlabel('Time (s)')
figure.show()

plt.plot(np.arange(0.1,1.1,0.1),mse_1,'g-',label='TM_sol')
plt.plot(np.arange(0.1,1.1,0.1),mse_2,'-',color='orange',label='Euler sol')
plt.xlabel('Time step')
plt.ylabel('MSE with analytical solution')
plt.title('Plot of the MSE')
plt.legend(loc='upper left')
plt.show()





