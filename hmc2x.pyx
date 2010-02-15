import numpy as np
cimport numpy as np
cimport cython
import random

cdef extern from "math.h":
    double exp(double x)
    double log(double x)

cdef inline int int_abs(int a): return a if a > 0 else -a


@cython.boundscheck(False)
def hmc_main_loop(f, np.ndarray[np.double_t, ndim=1] x, gradf, args,
                  np.ndarray[np.double_t, ndim=1] p,
                  np.ndarray[np.double_t, ndim=2] samples,
                  np.ndarray[np.double_t, ndim=1] energies,
                  np.ndarray[np.double_t, ndim=2] diagn_pos,
                  np.ndarray[np.double_t, ndim=2] diagn_mom,
                  np.ndarray[np.double_t, ndim=1] diagn_acc,
                  int opt_nsamples, int opt_nomit, int opt_window,
                  int opt_steps, int opt_display,
                  int opt_persistence, int return_energies, int return_diagnostics,
                  double alpha, double salpha, double epsilon):
    cdef int nparams = x.shape[0]
    cdef int nreject = 0              # number of rejected samples
    cdef int window_offset = 0        # window offset initialised to zero
    cdef int k = -opt_nomit       # nomit samples are omitted, so we store
    cdef int n, stps, direction, have_rej, have_acc
    cdef double a, E, Eold, E_acc, E_rej, H, Hold, acc_free_energy, rej_free_energy

    cdef unsigned int i, j, m
    cdef np.ndarray[np.double_t] xold = np.zeros(nparams)
    cdef np.ndarray[np.double_t] pold = np.zeros(nparams)
    cdef np.ndarray[np.double_t] x_acc = np.zeros(nparams)
    cdef np.ndarray[np.double_t] p_acc = np.zeros(nparams)
    cdef np.ndarray[np.double_t] x_rej = np.zeros(nparams)
    cdef np.ndarray[np.double_t] p_rej = np.zeros(nparams)
    cdef np.ndarray[np.double_t] ptmp = np.zeros(nparams)

    rand = random.random
    randn = random.normalvariate

    # Evaluate starting energy.
    E = f(x, *args)

    while k < opt_nsamples:  # samples from k >= 0
        # Store starting position and momenta
        for i in range(nparams): xold[i] = x[i]
        for i in range(nparams): pold[i] = p[i]
        # Recalculate Hamiltonian as momenta have changed
        Eold = E
        # Hold = E + 0.5*(p*p')
        Hold = E
        for i in range(nparams): Hold += .5*p[i]**2

        # Decide on window offset, if windowed HMC is used
        if opt_window > 1:
            # window_offset=fix(opt_window*rand(1));
            window_offset = int(opt_window*rand())

        have_rej = 0
        have_acc = 0
        n = window_offset
        direction = -1 # the default value for direction 
                       # assumes that windowing is used

        while direction == -1 or n != opt_steps:
            # if windowing is not used or we have allready taken
            # window_offset steps backwards...
            if direction == -1 and n == 0:
                # Restore, next state should be original start state.
                if window_offset > 0:
                    for i in range(nparams): x[i] = xold[i]
                    for i in range(nparams): p[i] = pold[i]
                    n = window_offset

                # set direction for forward steps
                E = Eold
                H = Hold
                direction = 1
                stps = direction
            else:
                if n*direction+1<opt_window or n > (opt_steps-opt_window):
                    # State in the accept and/or reject window.
                    stps = direction
                else:
                    # State not in the accept and/or reject window. 
                    stps = opt_steps-2*(opt_window-1)

                # First half-step of leapfrog.
                # p = p - direction*0.5*epsilon.*feval(gradf, x, varargin{:});
                p = p - direction*0.5*epsilon*gradf(x, *args)
                for i in range(nparams): x[i] += direction*epsilon*p[i]
                
                # Full leapfrog steps.
                # for m = 1:(abs(stps)-1):
                for m in range(int_abs(stps)-1):
                    # p = p - direction*epsilon.*feval(gradf, x, varargin{:});
                    p = p - direction*epsilon*gradf(x, *args)
                    for i in range(nparams): x[i] += direction*epsilon*p[i]

                # Final half-step of leapfrog.
                # p = p - direction*0.5*epsilon.*feval(gradf, x, varargin{:});
                p = p - direction*0.5*epsilon*gradf(x, *args)

                # E = feval(f, x, varargin{:});
                E = f(x, *args)
                # H = E + 0.5*(p*p');
                H = E
                for i in range(nparams): H += 0.5*p[i]**2

                n += stps

            if opt_window != opt_steps+1 and n < opt_window:
                # Account for state in reject window.  Reject window can be
                # ignored if windows consist of the entire trajectory.
                if not have_rej:
                    rej_free_energy = H
                else:
                    rej_free_energy = -addlogs(-rej_free_energy, -H)

                if not have_rej or rand() < exp(rej_free_energy-H):
                    E_rej = E
                    for i in range(nparams): x_rej[i] = x[i]
                    for i in range(nparams): p_rej[i] = p[i]
                    have_rej = 1

            if n > (opt_steps-opt_window):
                # Account for state in the accept window.
                if not have_acc:
                    acc_free_energy = H
                else:
                    acc_free_energy = -addlogs(-acc_free_energy, -H)

                if not have_acc or  rand() < exp(acc_free_energy-H):
                    E_acc = E
                    for i in range(nparams): x_acc[i] = x[i]
                    for i in range(nparams): p_acc[i] = p[i]
                    have_acc = 1
  
        # Acceptance threshold.
        a = exp(rej_free_energy - acc_free_energy)

        if return_diagnostics and k >= 0:
            j = k
            for i in range(nparams):
                diagn_pos[j,i] = x_acc[i]
                diagn_mom[j,i] = p_acc[i]
            diagn_acc[j] = a

        if opt_display:
            print 'New position is\n',x

        # Take new state from the appropriate window.
        if a > rand():
            # Accept 
            E = E_acc
            for i in range(nparams): x[i] = x_acc[i]
            for i in range(nparams): p[i] = -p_acc[i] # Reverse momenta
            if opt_display:
                print 'Finished step %4d  Threshold: %g\n'%(k,a)
        else:
            # Reject
            if k >= 0:
                nreject = nreject + 1

            E = E_rej
            for i in range(nparams): x[i] = x_rej[i]
            for i in range(nparams): p[i] = p_rej[i]
            if opt_display:
                print '  Sample rejected %4d.  Threshold: %g\n'%(k,a)

        if k >= 0:
            j = k
            # Store sample
            for i in range(nparams): samples[j,i] = x[i]
            if return_energies:
                # Store energy
                energies[j] = E

        # Set momenta for next iteration
        if opt_persistence:
            # Reverse momenta
            for i in range(nparams): p[i] = -p[i]
            # Adjust momenta by a small random amount
            for i in range(nparams): p[i] = alpha*p[i]+salpha*<double>randn(0,1)
        else:
            # Replace all momenta
            for i in range(nparams): p[i] = randn(0,1)

        k += 1

    return nreject

cdef double addlogs(double a, double b):
    if a>b:
        return a + log(1+exp(b-a))
    else:
        return b + log(1+exp(a-b))
