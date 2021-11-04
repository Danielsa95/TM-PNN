import weights as mn
import numpy as np
import Code.helper_fun as tmp
import matplotlib.pyplot as plt

"""           #######      PENDULUM 
#  In this file there is a comparison between two solutions :
#  1) the solution_1 build using the TM method and the weights found by the 'weights.py' code;
#  2) the solution_2 build using the TM method and the weights of the paper [1]. 

A measure of the error is found using RK45 method (the one of the /pacs-examples/Examples/src/RKFSolver ).
"""


#Initialization of the parameters and the coefficients of the problem
g=9.81
L=0.3
P0=np.array([[0],[0]])
P1=np.array([0,1,-g/L,0]).reshape(2,2)
P2=np.array([0,0,0,0,0,0]).reshape(2,3)
P3=np.array([0,0,0,0,(g/6*L),0,0,0]).reshape(2,4)


# Finding the weights
dt=0.1                      # time step for the solution and for the evaluation of the weights matrices ;
Tf=5.                       # final time ;
nstep=1000                  # step used in the first sub-interval [0,dt] .
tempo=np.arange(0,Tf+dt,dt)
x0=np.array([[0.3],[0]])    # initial solution
coef=(P0,P1,P2,P3)
W0,W1,W2,W3 = mn.weight(coef,dt,2,nstep)
tmp.show(W0,W1,W2,W3)


################             solution_1     ###############

solution_1=tmp.solve2((W0,W1,W2,W3),x0,Tf/dt)
x_m=[solution_1[i][0][0] for i in range(len(solution_1))]
y_m=[solution_1[i][1][0] for i in range(len(solution_1))]

plt.plot(tempo,x_m,'r-')
plt.title('Solution_1')
plt.xlabel('Time')
plt.ylabel('Angle')
plt.show()



#################            solution_2            ##############
W1p=np.array([[0.84,0.09],[-3.1,0.84]])
W3p=np.array([[0.02,0.0023,0.00012,2.3e-6],[0.43,0.064,0.0044,0.00012]])
solution_2=tmp.solve2((W0,W1p,W2,W3p),x0,Tf/dt)
x_p=[solution_2[i][0][0] for i in range(len(solution_2))]
y_p=[solution_2[i][1][0] for i in range(len(solution_2))]
plt.plot(tempo,x_p,'b-')
plt.title('Solution_2')
plt.xlabel('Time')
plt.ylabel('Angle')
plt.show()



################                  RK solution                  ##############
data=np.loadtxt("resultpen.dat")
te_mpo=data[:,0]
x_x=data[:,1]
y_y=data[:,2]

plt.plot(x_x,'m-')
plt.title('RK ')
plt.xlabel('Time')
plt.ylabel('Angle')
plt.show()


################                 Plot of three solutions                  ################

plt.plot(tempo,x_m,'b-',label="Sol_1")
plt.plot(tempo,x_p,'rx',label="Sol_2")
plt.plot(te_mpo,x_x,'m-',label="RK")
plt.title("Plot of the three solutions")
plt.xlabel('Time')
plt.ylabel('Angle')
plt.legend()
plt.xlim(0,2)
plt.grid()
plt.show()
plt.plot(tempo,y_m,'b-',label="Sol_1")
plt.plot(tempo,y_p,'rx',label="Sol_2")
plt.plot(te_mpo,y_y,'m-',label="RK")
plt.title("Plot of the three solutions")
plt.xlabel('Time')
plt.ylabel('Velocity')
plt.legend()
plt.xlim(0,2)
plt.grid()
plt.show()

###############                  Evaluation of the norm between the matrices of weights ##################
list1=(W0,W1,W2,W3)
list2=(W0,W1p,W2,W3p)
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

norma_x_p = np.linalg.norm(x_p-dati[:,1]) ### ||sol_2 -RK||_2
norma_y_p = np.linalg.norm(y_p-dati[:,2]) ##


print("########### Norm in x of sol_1 and sol_2 wrt the real solution  ################")
print(norma_x_m , norma_x_p)
print("########### Norm in y of sol_1 and sol_2 wrt the real solution ################")
print(norma_y_m , norma_y_p)


