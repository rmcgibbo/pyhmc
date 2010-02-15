"""Energy functions for phase model
"""

import numpy as np

#-----------------------------------------------------------------------------
# Energy functions
#-----------------------------------------------------------------------------

def f_multivariate_normal(x,M):
    """Energy function for multivariate normal distribution
    """
    return .5*np.dot(np.dot(x,M),x)

def f_phasedist(theta,M):
    """Energy function for phase model
    """
    x = np.zeros(2*len(theta))
    x[::2] = np.cos(theta)
    x[1::2] = np.sin(theta)
    return -.5*np.dot(np.dot(x,M),x)

def f_phasedist_biased(theta,M):
    """Energy function for phase model with bias term
    """
    x = np.zeros(2*len(theta)+2)
    x[0] = 1
    x[2::2] = np.cos(theta)
    x[3::2] = np.sin(theta)
    return -.5*np.dot(np.dot(x,M),x)


#-----------------------------------------------------------------------------
# Energy gradients
#-----------------------------------------------------------------------------

def g_multivariate_normal(x,M):
    """Energy gradient for multivariate normal distribution
    """
    return .5*np.dot(x,M+M.T)

def g_phasedist(theta,M):
    """Energy gradient for phase model
    """
    x = np.zeros(2*len(theta))
    x[::2] = np.cos(theta)
    x[1::2] = np.sin(theta)
    xdot = np.zeros((2*len(theta),len(theta)))
    xdot[::2,:] = np.diag(-np.sin(theta))
    xdot[1::2,:] = np.diag(np.cos(theta)) 
    return -np.dot(np.dot(x,M),xdot)

def g_phasedist_biased(theta,M):
    """Energy gradient for phase model with bias term
    """
    x = np.zeros(2*len(theta)+2)
    x[0] = 1
    x[2::2] = np.cos(theta)
    x[3::2] = np.sin(theta)
    xdot = np.zeros((2*len(theta)+2,len(theta)))
    xdot[2::2,:] = np.diag(-np.sin(theta))
    xdot[3::2,:] = np.diag(np.cos(theta)) 
    return -np.dot(np.dot(x,M),xdot)

