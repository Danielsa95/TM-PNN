import weights as mn
import numpy as np
import helper_fun as tmp
import matplotlib.pyplot as plt

"""           #######      CHARGED PARTICLE
#  In this file there is a comparison between two solutions :
#  1) the solution_1 build using the TM method and the weights found by the 'weights.py' code;
#  2) the solution_2 build using the TM method and the weights of the paper[2]. 

A measure of the error is found using RK45 method (the one of the /pacs-examples/Examples/src/RKFSolver ).
"""

#Initialization of the parameters and the coefficients of the problem
R=10
P0=np.array([[0],[0]])
P1=np.array([0,1,-2,0]).reshape(2,2)
P2=np.array([0,0,0,1/R,0,0]).reshape(2,3)
P3=np.array([0,0,0,0,0,0,0,0]).reshape(2,4)

Tf=10*np.pi/4
dt=Tf/10
tempo=np.arange(0,Tf+dt,dt)
x0=np.array([[0.2],[0.1]])

nstep=1000
coef=(P0,P1,P2,P3)

# Finding the weights matrices
W0,W1,W2,W3 = mn.weight(coef,dt,2,nstep)
tmp.show(W0,W1,W2,W3)

################             solution_1     ###############
solution_1=tmp.solve2((W0,W1,W2,W3),x0,Tf/dt)
x_m=[solution_1[i][0] for i in range(len(solution_1))]
y_m=[solution_1[i][1] for i in range(len(solution_1))]


################             solution_2    ###############
W0p=np.array([[0],[0]])
W1p=np.array([0.44,0.63,-1.3,0.44]).reshape(2,2)
W2p=np.array([0.23e-1,0.12e-1,0.26e-2,0.4e-1,0.35e-1,0.12e-1]).reshape(2,3)
W3p=np.array([0.21e-3,0.17e-3,0.47e-4,0.56e-5,0.83e-3,0.95e-3,0.32e-3,0.47e-4]).reshape(2,4)

solution_2=tmp.solve2((W0p,W1p,W2p,W3p),x0,Tf/dt)
xp=[solution_2[i][0] for i in range(len(solution_2))]
yp=[solution_2[i][1] for i in range(len(solution_2))]

################     RK_solution     ###############

data=np.loadtxt("datacharged.dat")
te_mpo=data[:,0]
xx=data[:,1]
yy=data[:,2]

############### plot #####################
figure,axis=plt.subplots(2,3)
axis[0,0].plot(te_mpo,xx,'r--')
axis[0,0].set_title('Real_sol')
axis[0,0].set_ylabel('x')
axis[0,1].plot(tempo,x_m,'r-')
axis[0,1].set_title('Sol_1')
axis[1,0].plot(xx,yy,'b--')
axis[1,0].set_ylabel('y')
axis[1,0].set_xlabel('x')
axis[1,1].plot(x_m,y_m,'b-')
axis[1,1].set_xlabel('x')
axis[0,2].plot(tempo,xp,'b-')
axis[0,2].set_title('Sol_2')
axis[1,2].plot(xp,yp)
axis[1,2].set_xlabel('x')

figure.show()

###############                  Evaluation of the norm between the matrices of weights ##################
list1=(W0,W1,W2,W3)
list2=(W0p,W1p,W2p,W3p)
diff=tmp.norma(list1,list2)
print('########### The norm between the matrices of weights : ')

print(diff)

################                 Evaluation of the norm                    ################
# Now i use the RK45 sol to evaluate the distance between the 'real' sol and solution_1 and solution_2 .

# Here i use the find_index auxiliar function to find where are the indexes of the solutions_1
# where there are the corresponding value of the time in RK45 solution.


index=tmp.find_index(tempo,te_mpo)  ## i know how to find the values in data

dati=data[index]

norma_x_m = np.linalg.norm(x_m-dati[:,1]) ### ||sol_1 -RK||_2
norma_y_m = np.linalg.norm(y_m-dati[:,2]) ###

norma_x_p = np.linalg.norm(xp-dati[:,1]) ### ||sol_2 -RK||_2
norma_y_p = np.linalg.norm(yp-dati[:,2]) ##


print("########### Norm in x of sol_1 and sol_2 wrt the real solution  ################")
print(norma_x_m , norma_x_p)
print("########### Norm in y of sol_1 and sol_2 wrt the real solution ################")
print(norma_y_m , norma_y_p)



