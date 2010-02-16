"""Hybrid Monte Carlo Sampling

This package is a straight-forward port of the functions hmc2.m and
hmc2_opt.m from the MCMCstuff matlab toolbox written by Aki Vehtari
<http://www.lce.hut.fi/research/mm/mcmcstuff/>.
   
The code is originally based on the functions hmc.m from the netlab toolbox
written by Ian T Nabney <http://www.ncrg.aston.ac.uk/netlab/index.php>.

The portion of algorithm involving "windows" is derived from the C code for
this function included in the Software for Flexible Bayesian Modeling
written by Radford Neal <http://www.cs.toronto.edu/~radford/fbm.software.html>.
   
This software is distributed under the BSD License (see LICENSE file).

Authors
-------
- Kilian Koepsell <kilian@berkeley.edu>
"""

#-----------------------------------------------------------------------------
# Public interface
#-----------------------------------------------------------------------------
__all__ = ['hmc']

#-----------------------------------------------------------------------------
# Imports
#-----------------------------------------------------------------------------
import numpy as np
import os

#-----------------------------------------------------------------------------
# Module globals
#-----------------------------------------------------------------------------

# Global variable to store state of momentum variables: set by set_state
# Used to initialize variable if set
HMC_MOM = None

# List of default options
default_options = dict(display=False,
                       checkgrad=False,
                       steps=1,
                       nsamples=1,
                       nomit=0,
                       persistence=False,
                       decay=0.9,
                       stepadj=0.2,
                       window=1,
                       use_cython=True)

#-----------------------------------------------------------------------------
# Functions
#-----------------------------------------------------------------------------

def check_grad(func, grad, x0, *args):
    """from scipy.optimize
    """
    _epsilon = np.sqrt(np.finfo(float).eps)
    def approx_fprime(xk,f,epsilon,*args):
        f0 = f(*((xk,)+args))
        grad = np.zeros((len(xk),), float)
        ei = np.zeros((len(xk),), float)
        for k in range(len(xk)):
            ei[k] = epsilon
            grad[k] = (f(*((xk+ei,)+args)) - f0)/epsilon
            ei[k] = 0.0
        return grad
    
    return np.sqrt(np.sum((grad(x0,*args)-approx_fprime(x0,func,_epsilon,*args))**2))

def hmc(f, x, gradf,
        args=(),
        display=False,
        checkgrad=False,
        steps=1,
        nsamples=1,
        nomit=0,
        persistence=False,
        decay=0.9,
        stepadj=0.2,
        window=1,
        use_cython=True,
        return_energies=False,
        return_diagnostics=False,
        ):
    """Hybrid Monte Carlo sampling.

    Uses a hybrid Monte Carlo algorithm to sample from the distribution P ~
    exp(-f), where f is the first argument to hmc. The Markov chain starts
    at the point x, and the function gradf is the gradient of the `energy'
    function f.

    Parameters
    ----------

    f : function
      Energy function.
    x : 1-d array
      Starting point for the sampling Markov chain.
    gradf : function
      Gradient of the energy function f.

    Optional Parameters
    -------------------

    display : bool
      If True, enables verbose display output. Default: False
    steps : int
      Defines the trajectory length (i.e. the number of leapfrog
      steps at each iteration).
    nsamples : int
      The number of samples retained from the Markov chain.
    nomit : int
      The number of samples omitted from the start of the chain
    persistence : bool
      If True, momentum persistence is used (i.e. the momentum
      variables decay rather than being replaced). Default: False
    decay : float
      Defines the decay used when a persistent update of (leap-frog)
      momentum is used. Bounded to the interval [0, 1.). Default: 0.9
    stepadj : float
      The step adjustment used in leap-frogs. Default: 0.2
    window : int
      The size of the acceptance window. Default: 1
    checkgrad : bool
      If True, instead of sampling only numerically checks the
      function gradient. Default: False
    return_energies : bool
      If True, the energy values for all samples are returned. Default: False
    return_diagnostics : bool
      If True, diagnostic information is returned (see below). Defauls: False
    args : tuple
      additional arguments to be passed to f() and gradf().

    Returns
    -------

    samples : array
      Array with data samples in rows.

    Optional return values
    ----------------------
    
    energies : array
      If return_energies is True, also returns an array of the energy values
      (i.e. negative log probabilities) for all samples.

    diagn : dict
      If return_diagnostics is True, also returns a dictionary with diagnostic
      information (position, momentum and acceptance threshold) for each step
      of the chain in diagn.pos, diagn.mom and diagn.acc respectively.
      All candidate states (including rejected ones) are stored in diagn['pos'].
      The diagn dictionary contains the following items:

      pos : array
       the position vectors of the dynamic process
      mom : array
       the momentum vectors of the dynamic process
      acc : array
       the acceptance thresholds
      rej : int
       the number of rejections
      stp : float
       the step size vectors
    """
    global HMC_MOM

    # check some options
    assert steps >= 1, 'step size has to be >= 1'
    assert nsamples >= 1, 'nsamples has to be >= 1'
    assert nomit >= 0, 'nomit has to be >= 0'
    assert decay >= 0, 'decay has to be >= 0'
    assert decay <= 1, 'decay has to be <= 1'
    assert window >= 0, 'window has to be >= 0'
    if window > steps:
        window = steps
        print "setting window size to step size %d" % window

    if persistence:
        alpha = decay
        salpha = np.sqrt(1-alpha**2);
    else:
        alpha = salpha = 0.

    nparams = len(x)
    epsilon = stepadj

    # Check the gradient evaluation.
    if checkgrad:
        # Check gradients
        error = check_grad(f, gradf, x, *args)
        print "Energy gradient error: %f"%error
        return error

    # Initialize matrix of returned samples
    samples = np.zeros((nsamples, nparams))

    # Return energies?
    if return_energies:
        energies = np.zeros(nsamples)
    else:
        energies = np.zeros(0)

    # Return diagnostics?
    if return_diagnostics:
        diagn_pos = np.zeros(nsamples, nparams)
        diagn_mom = np.zeros(nsamples, nparams)
        diagn_acc = np.zeros(nsamples)
    else:
        diagn_pos = np.zeros((0,0))
        diagn_mom = np.zeros((0,0))
        diagn_acc = np.zeros(0)

    if not persistence or HMC_MOM is None or nparams != len(HMC_MOM):
        # Initialise momenta at random
        p = np.random.randn(nparams)
    else:
        # Initialise momenta from stored state
        p = HMC_MOM
        
    # Main loop.
    all_args = [f,
                x,
                gradf,
                args,
                p,
                samples,
                energies,
                diagn_pos,
                diagn_mom,
                diagn_acc,
                nsamples,
                nomit,
                window,
                steps,
                display,
                persistence,
                return_energies,
                return_diagnostics,
                alpha,
                salpha,
                epsilon]

    if use_cython:
        try:
            os.environ['C_INCLUDE_PATH']=np.get_include()
            import pyximport; pyximport.install()
            from hmc2x import hmc_main_loop as c_hmc_main_loop
            print "Using compiled code"
            nreject = c_hmc_main_loop(*all_args)
        except:
            print "Using pure python code"
            nreject = hmc_main_loop(*all_args)
    else:
        print "Using pure python code"
        nreject = hmc_main_loop(*all_args)

    if display:
        print '\nFraction of samples rejected:  %g\n'%(nreject/float(nsamples))

    # Store diagnostics
    if return_diagnostics:
        diagn = dict()
        diagn['pos'] = diagn_pos   # positions matrix
        diagn['mom'] = diagn_mom   # momentum matrix
        diagn['acc'] = diagn_acc   # acceptance treshold matrix
        diagn['rej'] = nreject/float(nsamples)   # rejection rate
        diagn['stps'] = epsilon    # stepsize vector

    # Store final momentum value in global so that it can be retrieved later
    if persistence:
        HMC_MOM = p
    else:
        HMC_MOM = None

    if return_energies or return_diagnostics:
        out = (samples,)
    else:
        return samples
    
    if return_energies: out += (energies,)
    if return_diagnostics: out += (diagn,)
    return out

def hmc_main_loop(f, x, gradf, args, p, samples,
                  energies, diagn_pos, diagn_mom, diagn_acc,
                  nsamples, nomit, window, steps, display,
                  persistence, return_energies, return_diagnostics,
                  alpha, salpha, epsilon):
    nparams = len(x)
    nreject = 0              # number of rejected samples
    window_offset = 0        # window offset initialised to zero
    k = -nomit       # nomit samples are omitted, so we store
    
    # Evaluate starting energy.
    E = f(x, *args)

    while k < nsamples:  # samples from k >= 0
        # Store starting position and momenta
        xold = x
        pold = p
        # Recalculate Hamiltonian as momenta have changed
        Eold = E
        # Hold = E + 0.5*(p*p')
        Hold = E + 0.5*(p**2).sum()

        # Decide on window offset, if windowed HMC is used
        if window > 1:
            # window_offset=fix(window*rand(1));
            window_offset = int(window*np.random.rand())

        have_rej = 0
        have_acc = 0
        n = window_offset
        direction = -1 # the default value for direction 
                       # assumes that windowing is used

        while direction == -1 or n != steps:
            # if windowing is not used or we have allready taken
            # window_offset steps backwards...
            if direction == -1 and n==0:
                # Restore, next state should be original start state.
                if window_offset > 0:
                    x = xold
                    p = pold
                    n = window_offset

                # set direction for forward steps
                E = Eold
                H = Hold
                direction = 1
                stps = direction
            else:
                if n*direction+1<window or n > (steps-window):
                    # State in the accept and/or reject window.
                    stps = direction
                else:
                    # State not in the accept and/or reject window. 
                    stps = steps-2*(window-1)

                # First half-step of leapfrog.
                # p = p - direction*0.5*epsilon.*feval(gradf, x, varargin{:});
                p = p - direction*0.5*epsilon*gradf(x, *args)
                x = x + direction*epsilon*p
                
                # Full leapfrog steps.
                # for m = 1:(abs(stps)-1):
                for m in xrange(abs(stps)-1):
                    # p = p - direction*epsilon.*feval(gradf, x, varargin{:});
                    p = p - direction*epsilon*gradf(x, *args)
                    x = x + direction*epsilon*p

                # Final half-step of leapfrog.
                # p = p - direction*0.5*epsilon.*feval(gradf, x, varargin{:});
                p = p - direction*0.5*epsilon*gradf(x, *args)

                # E = feval(f, x, varargin{:});
                E = f(x, *args)
                # H = E + 0.5*(p*p');
                H = E + 0.5*(p**2).sum()

                n += stps

            if window != steps+1 and n < window:
                # Account for state in reject window.  Reject window can be
                # ignored if windows consist of the entire trajectory.
                if not have_rej:
                    rej_free_energy = H
                else:
                    rej_free_energy = -addlogs(-rej_free_energy, -H)

                if not have_rej or np.random.rand() < np.exp(rej_free_energy-H):
                    E_rej = E
                    x_rej = x
                    p_rej = p
                    have_rej = 1

            if n > (steps-window):
                # Account for state in the accept window.
                if not have_acc:
                    acc_free_energy = H
                else:
                    acc_free_energy = -addlogs(-acc_free_energy, -H)

                if not have_acc or  np.random.rand() < np.exp(acc_free_energy-H):
                    E_acc = E
                    x_acc = x
                    p_acc = p
                    have_acc = 1
  
        # Acceptance threshold.
        a = np.exp(rej_free_energy - acc_free_energy)

        if return_diagnostics and k >= 0:
            diagn_pos[k,:] = x_acc
            diagn_mom[k,:] = p_acc
            diagn_acc[k,:] = a

        if display:
            print 'New position is\n',x

        # Take new state from the appropriate window.
        if a > np.random.rand():
            # Accept 
            E = E_acc
            x = x_acc
            p = -p_acc # Reverse momenta
            if display:
                print 'Finished step %4d  Threshold: %g\n'%(k,a)
        else:
            # Reject
            if k >= 0:
                nreject = nreject + 1

            E = E_rej
            x = x_rej
            p = p_rej
            if display:
                print '  Sample rejected %4d.  Threshold: %g\n'%(k,a)

        if k >= 0:
            # Store sample
            samples[k,:] = x;
            if return_energies:
                # Store energy
                energies[k] = E

        # Set momenta for next iteration
        if persistence:
            # Reverse momenta
            p = -p
            # Adjust momenta by a small random amount
            p = alpha*p + salpha*np.random.randn(nparams)
        else:
            # Replace all momenta
            p = np.random.randn(nparams)

        k += 1

    return nreject


def get_state():
    """Return complete state of sampler (including momentum)

            Description
            get_state() returns a state structure that contains the state of
            the internal random number generators and the momentum of the
            dynamic process. These are contained in fields randstate mom
            respectively.
            The momentum state is only used for a persistent momentum update.
    """
    global HMC_MOM
    return dict(randstate = np.random.get_state(),
                mom = HMC_MOM)


def set_state(state):
    """Set complete state of sampler (including momentum).
        
            Description
            set_state(state) resets the state to a given state.
            If state is a dictionary returned by get_state() then it resets the
            generator to exactly the same state.
    """
    global HMC_MOM
    assert type(state) == dict, 'state has to be a state dictionary'
    assert state.has_key('randstate'), 'state does not contain randstate'
    assert state.has_key('mom'), 'state does not contain momentum'
    np.random.set_state(state['randstate'])
    HMC_MOM = state['mom']


def addlogs(a,b):
    """Add numbers represented by their logarithms.
    
            Description
            Add numbers represented by their logarithms.
            Computes log(exp(a)+exp(b)) in such a fashion that it 
            works even when a and b have large magnitude.
    """
    
    if a>b:
        return a + np.log(1+np.exp(b-a))
    else:
        return b + np.log(1+np.exp(a-b))


if __name__ == '__main__':
    from time import time as now
    
    def f_multivariate_normal(x,M):
        """Energy function for multivariate normal distribution
        """
        return .5*np.dot(np.dot(x,M),x)

    def g_multivariate_normal(x,M):
        """Energy gradient for multivariate normal distribution
        """
        return .5*np.dot(x,M+M.T)

    # sample from 5-dimensional unit variance gaussian
    dim = 5
    M = np.eye(dim)
    x0 = np.zeros(dim)    
    t0 = now()
    samples = hmc(f_multivariate_normal, x0, g_multivariate_normal, args=(M,),
                  nsamples=10**3, nomit=10**3, steps=100, stepadj=.05, use_cython=False)
    dt = now()-t0

    # check covariance matrix of sampled data
    C = np.cov(samples,rowvar=0,bias=1)

    print "mean squared error of sample covariance (expect .001): %f" % ((C-np.linalg.inv(M))**2).mean()
    print "time (python): ",dt

    try:
        os.environ['C_INCLUDE_PATH']=np.get_include()
        import pyximport; pyximport.install()
        from f_energyx import f_multivariate_normal, g_multivariate_normal
        t0 = now()
        samples = hmc(f_multivariate_normal, x0, g_multivariate_normal, args=(M,),
                      nsamples=10**3, nomit=10**3, steps=100, stepadj=.05, use_cython=True)
        dt = now()-t0

        # check covariance matrix of sampled data
        C = np.cov(samples,rowvar=0,bias=1)

        print "mean squared error of sample covariance (expect .001): %f" % ((C-np.linalg.inv(M))**2).mean()
        print "time (cython): ",dt
    except:
        print "cython not installed"
