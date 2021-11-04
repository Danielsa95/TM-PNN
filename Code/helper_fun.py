import numpy as np
import Code.weights as mn
from math import ceil

# Simple function for the printing of the weights values
def show(W0,W1,W2,W3):
    print('######    W0   ######')
    print(W0)
    print('######    W1   ######')
    print(W1)
    print('######    W2   ######')
    print(W2)
    print('######    W3   ######')
    print(W3)
    return 0



# Input  -> vet : a vector of the form [a,b] ;
#        ->   n : the desidered power ;
# Output -> the n_th power of Kronecker with same term reduction.
def kron(vet,n) :
    res=np.zeros(n+1)
    for i in range(n+1):
        res[i]=vet[0]**(n-i) * vet[1]**i
    return res



# Input  ->  w0 : the 1st vector of weights ;
#        ->   W : a matrix of weight like W1, W2 or W3 ;
# Output ->  Kronecker product  with same term reduction.
def kron_0(w0,W) :
    temp = np.array([[w0[0][0], 0], [w0[1][0] / 2, w0[0][0] / 2], [0, w0[1][0]]])
    return np.matmul(temp,W)



# Input  ->  W1 : the W1 matrix of weights ;
#        ->   W : a matrix of weight like W1, W2 or W3 ;
# Output ->  Kronecker product  with same term reduction.
def kron_1(W1,W) :
  M3=np.kron(W1,W)
  M3[1, :] = (M3[1, :] + M3[2, :]) / 2
  M3 = np.delete(M3, (2), axis=0)
  for i in range(W.shape[1]-1) :
   M3[:,1+i]=M3[:,1+i]+M3[:,i+W.shape[1]]
  for i in range(W.shape[1]-1):
   M3=np.delete(M3,(W.shape[1]),axis=1)
  return M3



# Input  ->  W0 : the W0 vector of weights ;
#        ->  W1 : the W1 matrix of weights;
#        ->  W2 : the W2 matrix of weights;
# Output ->  Kronecker product with same term reduction.
def kron_123(w0,w1,w2):
    if w1.shape[1]==2:
        M=np.kron(w0,kron_1(w1,w2))
    if w1.shape[1]==1:
        M=np.kron(w0,kron_0(w1,w2))
    M[1,:] = (2*M[1,:] + M[3,:])/3
    M[2,:] = (M[2,:] + 2 * M[4,:])/3
    M = np.delete(M, (3,4), axis=0)
    return M



# Input  ->  W1 : the W1 matrix of weights;
# Output -> the 3rd power of Kronecker with same term reduction of W1 .
def kron_1_3(W1):
    M=np.kron(W1,kron_1(W1,W1))
    M = np.delete(M, (3, 4), axis=0)
    M[:,1]=M[:,1] + M[:,3]
    M[:, 2] = M[:, 2] + M[:, 4]
    M = np.delete(M, (3, 4), axis=1)
    return M



#This function takes the list of coefficients' matrices and return them.
def coeff(lista):
    P0=lista[0]
    P1=lista[1]
    P2=lista[2]
    P3=lista[3]
    return P0,P1,P2,P3

#########################################################################################################
#########################################################################################################

##    FROM NOW THERE ARE ONLY SUPPORT FUNCTIONS USED IN FILE DIFFERENT FROM WEIGHTS  !!  ##

#########################################################################################################
#########################################################################################################

# Function to solve a 1D problem.
# Input  ->  pesi       : list of the weights;
#        ->  val_init   : the initial value x0 ;
#        ->  n_step     : # of step of the TM iterative method.
# Output ->  list of solutions, that are values of the function in every dt .
def solve(pesi,val_init,n_step):
 x=val_init
 sol=np.array([[]])
 sol=np.append(sol,x)
 for i in range(int(n_step)) :
    x=pesi[0] + pesi[1]*x + pesi[2]*x**2 + pesi[3]*x**3
    sol=np.append(sol,x)
 return sol



# Function to solve a 2D problem.
# Input  ->  pesi       : list of the weights;
#        ->  val_init   : the initial value X0=(x0,y0) ;
#        ->  n_step     : # of step of the TM iterative method.
# Output ->  list of solutions, that are values of the function in every dt .
def solve2(pesi,val_init,n_step):
    n_step=ceil(n_step)
    x = val_init
    sol = list()
    sol.append(x)
    for i in range(int(n_step)):
        x = pesi[0] + np.matmul(pesi[1],x) + np.matmul(pesi[2],kron(x,2).reshape(3,1)) +\
            np.matmul(pesi[3],kron(x,3).reshape(4,1))
        sol.append(x)
    return sol


# Input  -> lista1 : a list of the matrices evaluated with the TM method implemented.
#        -> lista2 : a list of the matrices given in the paper.
# Output -> vet_norme : a vector in which in every position there is an estimate (the norm-2) of the distance
#                       between the two type of weights : the ones found with TM method and those of paper.
def norma(lista1,lista2):
    vet_norme=list()
    for i in range(int(len(lista1))):
        tmp=np.linalg.norm(lista1[i]-lista2[i])
        vet_norme.append(tmp)
    return vet_norme



# A function that given a vector "vet" and a number "x" return the index of the vector in which there is the
# nearest value wrt x .
def find_near(vet,x):
    k=np.abs(vet[0]-x)
    idx=0
    for i in range(len(vet)) :
        diff=np.abs(vet[i]-x)
        if(k>diff):
             k=diff
             idx=i
    return idx



# To make a comparison between the solution that i obtain using the TM method and the rk45 method there
# is a preliminar thing to do : the first solution is discretize in a constant time step dt,differently
# the second is referred to some value of time.
# Using this function taking in inputs two vector (like the two vectors of the times) returns a vector
# that contain indices : every elements of the shorter vector is searched in the bigger one, if it is found
# then the indexes in where it is is signed, if not then the function find_near is called.
# Remember that both the two vectors contain value of time between 0 and Tf .
def find_index(x,y):
    if len(x) < len(y):
        tmp = x
        x = y
        y = tmp
    count = 0
    idx = list()
    find = False
    for i in range(len(y)):
        while count + i < len(x) - 1 and find == False:
            if (y[i] == x[i + count]):
                idx.append(i + count)
                find = True
            count += 1
        if count + i >= len(x) - 1:
            count = idx[-1]
            k = find_near(x, y[i])
            idx.append(k)
            count = k
        find = False

    return idx

## This function is used for Gravity_fall, it is used to evaluate MSE for dt = 0.1, 0.2 , ... , 1.0 .
# Input  -> coef : the list of the coefficients of the equation ;
#        -> m    : the mass ;
#        -> Tf   : the final time ;
#        -> init : the initial point.
# Output -> mse_1,mse_2  : they are two vectors that contains the evaluation of the MSE.

def MSE(coef,m,Tf,init):
    dt_init=0.1
    mse_1=np.zeros(10)
    mse_2=np.zeros(10)
    p0,p1,p2,p3=coef
    ### real sol
    sol = lambda t: np.sqrt(m * 9.81 / 0.392) * np.tanh(t * np.sqrt(0.392 * 9.81 / m))
    lis = (p0, p1, p2, p3)

    for i in np.arange(1, 11, 1):
        dt = dt_init * i
        W0_mse, W1_mse, W2_mse, W3_mse = mn.weight(lis, dt, 1)
        sol_real=sol(np.arange(dt,Tf+dt,dt))

        ### solution 1)
        sol_1 = solve((W0_mse, W1_mse, W2_mse, W3_mse), init, ceil(Tf / dt))
        sol_1 = sol_1[1:]
        mse_1[i-1]=np.mean((sol_1-sol_real)**2)


        ### solution 2)
        sol_2 = np.zeros(ceil(Tf / dt))
        x = init
        for j in np.arange(dt, Tf + dt, dt):
            x +=  9.81 * dt - (dt * 0.392 * x ** 2) / m
            sol_2[round(j/dt)-1]=x
        mse_2[i-1] = np.mean((sol_2 - sol_real) ** 2)

    return mse_1 , mse_2
