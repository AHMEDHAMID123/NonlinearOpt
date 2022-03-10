 #!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on Fri May 17 00:40:05 2021

@author: aabdelhamid
"""
import numpy as np;

    
    
def newton(grad,hess,x_0):
    """
     find the minimiser of f(x) numerically
    using undamped newton's algorithm.
    the function assumes that the hessian is postive semidefinite.
    

    Parameters
    ----------
    grad : lambda expression
        the gradient of function f(x) 
    hess : lambda expression
        the hessioan matrix of function f(x)
    x_0 : np.array
        the intial value of x

    Returns
    -------
        lsit containing : x_minimiser,
        gradient norm at this point, 
        number of iterations to find the solution

    """
    
    tol = 0.001
    nummax = 1000
    num = 0
    #intialize x_n = x_0
    x_n = np.array(x_0)

    #det of the gradient at x_n
    normgrad = np.linalg.norm(grad(x_n))

    # check if the intial value is below the tolerance 
    if normgrad <= tol:
        return [x_n,normgrad,num]
    # in case of the intial value is not the accepted
    else:
        # start a loop to start iterating  
        for i in range(nummax):
            #checking the size of x_n if 1-D or N-D
            if x_n.size > 1:
                #caluate the deccent direction
                d =   np.linalg.solve(hess(x_n),-1*grad(x_n))
            else:
                d = -grad(x_n)/hess(x_n)
            # assign the value of the next x_n+1
            x_n = x_n + d
            num = i
            #calulate the det of the gradient
            normgrad = np.linalg.norm(grad(x_n))
            # checking if the det is less than the tolerance 
            if normgrad <= tol:
                return [x_n,normgrad,num]
            elif num == nummax and normgrad >tol:
                return print("no solution after 1000 iteration")
