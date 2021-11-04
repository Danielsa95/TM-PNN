import Code.weights as mn
import numpy as np
import Code.helper_fun as tmp
import matplotlib.pyplot as plt

"""           #######     Van Der Pol
#  In this file there is a comparison between two solutions :
#  1) the solution_1 build using the TM method and the weights found by the 'weights.py' code;
#  2) the solution_2 build using the TM method and the weights of the paper[2]. 

A measure of the error is found using RK45 method (the one of the /pacs-examples/Examples/src/RKFSolver ).
"""


#Initialization of the coefficients
P0=np.array([[0],[0]])
P1=np.array([0,1,-1,1]).reshape(2,2)
P2=np.array([0,0,0,0,0,0]).reshape(2,3)
P3=np.array([0,0,0,0,0,-1,0,0]).reshape(2,4)


# inizializzo i parametri del metodo TM-PNN
dt=0.01                      # time step for the solution
Tf=100.                      # final time
nstep=1000                  # n_step in the first sub-intervall
time=np.arange(0,Tf+dt,dt)
x0=np.array([[1],[2]])
lista=(P0,P1,P2,P3)
W0,W1,W2,W3 = mn.weight(lista,dt,2,nstep)
tmp.show(W0,W1,W2,W3)


################            solution_1      ###############

solution_1=tmp.solve2((W0,W1,W2,W3),x0,Tf/dt)
x_m=[solution_1[i][0][0] for i in range(len(solution_1))]
y_m=[solution_1[i][1][0] for i in range(len(solution_1))]

plt.plot(time,x_m,'r-')
plt.title('Solution_1')
plt.xlabel('Time')
plt.ylabel('x')
plt.show()
plt.plot(x_m,y_m,'r-')
plt.title('Phase_Solution_1')
plt.xlabel('x')
plt.ylabel('y')
plt.show()



#################             solution_2             ##############

W0_p=np.array([[0],[0]])
W1_p=np.array([[0.99995067,0.01004917],[-0.01004917,1.00999984]])
W2_p=np.array([0,0,0,0,0,0]).reshape(2,3)
W3_p=np.array([[1.59504733e-7,-4.94822066e-5,-3.0576750e-7,-7.90629025e-10],[4.94821629e-5,-1.00975145e-2,-9.96173322e-5,-3.30168067e-07]])
lista1=(W0_p,W1_p,W2_p,W3_p)

solution_2=tmp.solve2(lista1,x0,Tf/dt)
x_p=[solution_2[i][0][0] for i in range(len(solution_2))]
y_p=[solution_2[i][1][0] for i in range(len(solution_2))]

plt.plot(time,x_p,'b-')
plt.title('Solution_2')
plt.xlabel('Time')
plt.ylabel('x')
plt.show()

plt.plot(x_p,y_p,'b-')
plt.title('Phase_Solution_2')
plt.xlabel('x')
plt.ylabel('y')
plt.show()




################                  soluzione RK                  ##############
data=np.loadtxt("resultVDP.dat")
te_mpo=data[:,0]
x_x=data[:,1]
y_y=data[:,2]

plt.plot(te_mpo,x_x,'m-')
plt.title('RK ')
plt.xlabel('Time')
plt.ylabel('x')
plt.show()


################                 Plot delle 3 soluzioni                   ################

plt.plot(time,x_m,'b-',label="Sol_1")
plt.plot(time,x_p,'rx',label="Sol_2")
plt.plot(te_mpo,x_x,'m-',label="RK")
plt.title("Comparison between x variable of the three solutions")
plt.xlabel('Time')
plt.ylabel('x')
plt.legend()
plt.xlim(0,10)
plt.grid()
plt.show()
plt.plot(x_m,y_m,'b-',label="Sol_1")
plt.plot(x_p,y_p,'rx',label="Sol_2")
plt.plot(x_x,y_y,'m-',label="RK")
plt.title("Comparison between the three phase")
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid()
plt.show()

###############                  Evaluation of the norm between the matrices of weights ##################
list1=(W0,W1,W2,W3)
list2=(W0_p,W1_p,W2_p,W3_p)
diff=tmp.norma(list1,list2)
print('########### The norm between the matrices of weights : ')

print(diff)

################                 Evaluation of the norm                    ################
# Now i use the RK45 sol to evaluate the distance between the 'real' sol and solution_1 and solution_2 .

# Here i use the find_index auxiliar function to find where are the indexes of the solutions_1
# where there are the corresponding value of the time in RK45 solution.
# NB : the code is divided in an if-else statement for the reason that the auxiliar function find the indexes
# of the bigger vector in which there are the same time value that are in the shorter one. That means that
# if there is a change in dt (and so the len of the variable 'time' changes) the else will run.

if len(te_mpo)>len(time) :


    index=tmp.find_index(time,te_mpo)  ## i know how to find the values in data

    dati=data[index]

    norma_x_m = np.linalg.norm(x_m-dati[:,1]) ### ||sol_1 -RK||_2
    norma_y_m = np.linalg.norm(y_m-dati[:,2]) ###

    norma_x_p = np.linalg.norm(x_p-dati[:,1]) ### ||sol_2 -RK||_2
    norma_y_p = np.linalg.norm(y_p-dati[:,2]) ##


    print("########### Norme del mio metodo e del paper per le x ################" )
    print(norma_x_m , norma_x_p)
    print("########### Norme del mio metodo e del paper per le y ################" )
    print(norma_y_m , norma_y_p)


#
else :

   index=tmp.find_index(time,te_mpo)  ## i know how to find the values in data

   # conversion of x_m e y_m from list to array
   tmp1 = np.zeros(len(x_m))
   tmp2 = np.zeros(len(y_m))
   for i in range(len(x_m)):
        tmp1[i]=x_m[i]
        tmp2[i]=y_m[i]
   x_m=tmp1
   y_m=tmp2

   # conversion of x_p e y_p from list to array
   tmp1 = np.zeros(len(x_p))
   tmp2 = np.zeros(len(y_p))
   for i in range(len(x_p)):
       tmp1[i] = x_p[i]
       tmp2[i] = y_p[i]
   x_p = tmp1
   y_p = tmp2


   norma_x_m = np.linalg.norm(x_m[index]-x_x) ### ||sol_1 -RK||_2
   norma_y_m = np.linalg.norm(y_m[index]-y_y) ###

   norma_x_p = np.linalg.norm(x_p[index]-x_x)  ### ||sol_2 -RK||_2
   norma_y_p = np.linalg.norm(y_p[index]-y_y)  ###

   print("########### Norm in x of sol_1 and sol_2 wrt the real solution  ################")
   print(norma_x_m , norma_x_p)
   print("########### Norm in y of sol_1 and sol_2 wrt the real solution ################")
   print(norma_y_m, norma_y_p)

