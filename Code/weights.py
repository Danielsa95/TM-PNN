"""
In this file there is the code that are used in all files to FIND THE WEIGHTS of the TaylorMap method.
In particular there are weights up to a third order ( W0,W1,W2,W3) that can be usefull to systems which have
a polynomial form up to a third order ( ax^3 + bx^2 + cx + d) or that can be approximated with a Taylor expansion.

Important: the system that are considered are 1D or 2D. (An higher order can be an interesting enlargement).
"""
import numpy as np
import helper_fun as tmp



# funzione che riceve la lista delle P (i coeff del sistema) , Tf lo step time del problema, che sarà l'estremo
# destro dell'intervallo in cui calcolo le W , dim che indica la dimensione del problema,
# n_step invece indica quanti step devo fare nel primo intervallo per ottenere le W.

# estremo dell'intrevallo ->> Tf. (suppongo che l'estremo sinistro sia 0).

# Input  ->   coef : list of P0,P1,P2,P3  which are the matrices of coefficients of the system ;
#        ->   Tf   : the right extrema of the interval, that is the point in which we want the value of the
#                   time-dependent functions of the differential system of the weights ;
#        ->   dim  : the dimension of the dynamical system ( 1 or 2 ) ;
#        -> n_step : the number of step of forward Euler to solve the system of the weights.

# Output -> list of weights' matrices.

def weight(coef,Tf,dim,n_step=1000):
    dt=Tf/n_step

    if dim==2 :
        P0,P1,P2,P3=tmp.coeff(coef)


                                 ###### WEIGHTS INITIALIZATION ######

######## W0
        W0_0 = np.array([[0],[0]])
        W0   = np.array([[0],[0]])

######### W1
        W1_0 = np.eye(2)
        W1   = np.eye(2)

######## W2
        W2_0 = np.zeros(6).reshape(2,3)
        W2   = np.zeros(6).reshape(2,3)

######## W3
        W3_0 = np.zeros(8).reshape(2,4)
        W3   = np.zeros(8).reshape(2,4)

                              ####### EVALUATION OF WEIGHTS ######

        for i in range(n_step) :
            # W0
            W0=W0_0 + dt * ( P0 + np.matmul(P1,W0_0) + np.matmul(P2,tmp.kron(W0_0,2)).reshape(2,1) +
                             np.matmul(P3,tmp.kron(W0_0,3).reshape(4,1)))
            # W1
            W1 = W1_0 + dt * (np.matmul(P1, W1_0) + np.matmul(P2, 2*tmp.kron_0(W0,W1) )  +
                              3*np.matmul(P3,tmp.kron_123(W0_0,W0_0,W1_0)))
            # W2
            W2 = W2_0 + dt * (np.matmul(P1, W2_0) + np.matmul(P2,  tmp.kron_1(W1_0, W1_0)) +
                              np.matmul(P2,tmp.kron_0(W0_0,W2_0)) + 3*np.matmul(P3,tmp.kron_123(W0_0,W0_0,W2_0)) +
                              3* np.matmul(P3, tmp.kron_123(W0_0,W1_0,W1_0) ))
            # W3
            W3 = W3_0 + dt * (np.matmul(P1,W3_0) + np.matmul(P2,2*tmp.kron_0(W0_0,W3_0)) +
                              np.matmul(P2, 2 * tmp.kron_1(W1, W2)) + np.matmul(P3,tmp.kron_1_3(W1_0)) +
                              np.matmul(P3, 3 * tmp.kron_123(W0_0, W0_0, W3_0)) + np.matmul(P3,6*tmp.kron_123(W0_0,W1_0,W2_0)))

            # Update
            W0_0=W0
            W1_0=W1
            W2_0=W2
            W3_0=W3


    if dim ==1 :

        p0,p1,p2,p3=tmp.coeff(coef)

    ######  WEIGHTS INITIALIZATION
        W0_0 = 0
        W1_0 = 1
        W2_0 = 0
        W3_0 = 0
        W0 = 0
        W1 = 0
        W2 = 0
        W3 = 0
        for i in range(n_step):
            W0=W0_0 + dt*(p0 + p1*W0_0 + p2*W0_0**2 + p3*W0_0**3)
            W1=W1_0 + dt*(p1*W1_0 + 2 * p2*W0_0*W1_0 + 3*p3*W1_0*W0_0**2 )
            W2=W2_0 + dt*(p1*W2_0 + p2*(W1_0**2 + 2*W0_0*W2_0) + 3*p3*(W2_0*W0_0**2 + W0_0 * W1_0**2) )
            W3=W3_0 + dt*(p1*W3_0 + 2*p2*(W0_0*W3_0+W1_0*W2_0) + p3*(W1_0**3 + 3*W3_0*W0_0**2  +
                                                                      6 * W0_0*W1_0*W2_0) )
        # Update :
            W0_0=W0
            W1_0=W1
            W2_0=W2
            W3_0=W3
    return (W0,W1,W2,W3)