# -*- coding: utf-8 -*-
"""
Created on Mon May 31 17:38:44 2021

@author: Ahmed Abdelhamid younes Zeid
"""

#!/usr/bin/python
import numpy as np;

def gradient_method(f,grad,x_0,gamma=1, delta = 0.1, tol = 0.001, nummax = 10000):
    """
    

    Parameters
    ----------
    f : Lambda expression returns numpy array
        opjective function.
    grad : lambda expression returns numpy array
        gradient of the function.
    x_0 : numpy array
        intial vector.
    gamma : scalar, optional
        . efficiency coeefficient, default value is 1.
    delta : scalar, optional
        value of the flatterning coefficient. The default is 0.1.
    tol : scalar, optional
        error term. The default is 0.001.
    nummax : scalar, optional
        maxx number of itterations before termination. The default is 10000.

    Returns
    -------
    list [x,gradient norm , number of iterations]
    where x vector that solution to objective function f,
    gradient norm is the norm of the gradient of f at x
    the number of iteration excuted to reach the solution x
    
    """
    
    def armijo(f,df,x_val,gamma,delta):
        '''
        

        Parameters
        ----------
    f : Lambda expression returns numpy array
        opjective function.
        df : lambda expression returns numpy array
        gradient of the function.
        x_val : numpy array
            vector x at which we are investigating the function during the current iterate.
        gamma : scalar

        delta : scalar
            flatterning coeff.

        Returns
        -------
        s_armijo : scalar
            efficient step size.

        '''
        # define variable s_armijo 
        s_armijo = 0
        # calculating s_0 intial set size efficient
        s_0 = - gamma* (df(x_val).transpose().dot(-df(x_val))) /(np.linalg.norm(df(x_val)))**2
        # checking if s_0 satisfy the Armijo condition
        if f(x_val+s_0*-df(x_val)) <= f(x_val)+delta*s_0*df(x_val).dot(-df(x_val)):
            # set s_armijo to s_0 if Armijo condtion is satisfied
            s_armijo = s_0
            
       # in case of Armijo get into a while loop to reduce s_0 till it statisfy Armijo
        while f(x_val+s_0*-df(x_val)) > f(x_val) +delta*s_0*df(x_val).dot(-df(x_val)):
            # reduce the step size each itteration 
                s_0 *=0.5
                s_armijo = s_0
        
        return s_armijo 
 
    # intialize a counter num = 0
    num = 0
    # set x vector to the intial value x_0
    x = x_0
    # calculate the norm of the gradient in x
    normgrad = np.linalg.norm(grad(x))
    # check if x satisfy the stopping condition
    if normgrad  <= tol:
        return [x,normgrad,num]
    # if not get into a while loop to apply the gradient algorithm  
    else:
        while(num < nummax):
            # d descent direction 
            d = -grad(x)
            # calculate s step size efficient 
            s = armijo(f,grad,x,gamma,delta)
            # calculate new x using the gradient algorithm
            x = x + s*d
            #calculate the gradient of x
            normgrad  = np.linalg.norm(grad(x))
            num +=1
            # check if x satisfy the stopping condition
            if normgrad <= tol:
                return [x,normgrad,num]

        
    

    return print("no solution after {} iteration".format(nummax))
