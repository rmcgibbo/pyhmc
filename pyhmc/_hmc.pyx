from __future__ import print_function
import numpy as np
cimport numpy as np
cimport cython
from numpy import zeros, asarray
from numpy cimport npy_intp, npy_uint8
from libc.math cimport sqrt, log, exp


@cython.boundscheck(False)
def hmc_main_loop(fun, double[::1] x, args, double[::1] p,
                  double[:, ::1] samples, double[::1] logps,
                  double[:, ::1] diagn_pos, double[:, ::1] diagn_mom,
                  double[::1] diagn_acc, npy_intp opt_nsamples,
                  npy_intp opt_nomit, npy_intp opt_window,
                  npy_intp opt_steps, npy_intp opt_display,
                  npy_intp opt_persistence, npy_intp return_logp,
                  npy_intp return_diagnostics, double alpha, double salpha,
                  double epsilon, random):

    cdef npy_intp n_params = x.shape[0]
    cdef npy_intp n_reject = 0         # number of rejected samples
    cdef npy_intp window_offset = 0    # window offset initialised to zero
    cdef npy_intp k = -opt_nomit       # nomit samples are omitted, so we store
    cdef npy_intp n, stps, direction, have_rej, have_acc
    cdef double a, E, E_old, E_acc, logp, E_rej, H, H_old, acc_free_energy, rej_free_energy

    cdef npy_intp i, j, m
    cdef double[::1] randn, grad
    cdef double[::1] x_old = zeros(n_params)
    cdef double[::1] p_old = zeros(n_params)
    cdef double[::1] x_acc = zeros(n_params)
    cdef double[::1] p_acc = zeros(n_params)
    cdef double[::1] x_rej = zeros(n_params)
    cdef double[::1] p_rej = zeros(n_params)
    cdef double[::1] p_tmp = zeros(n_params)

    # Evaluate starting energy.
    x = x.copy()
    logp, grad = fun(asarray(x), *args)
    E = -logp
    if len(grad) != n_params:
        raise ValueError('fun(x, *args) must return (logp, grad)')

    while k < opt_nsamples:  # samples from k >= 0
        # Store starting position and momenta
        for i in range(n_params):
            x_old[i] = x[i]
        for i in range(n_params):
            p_old[i] = p[i]

        # Recalculate Hamiltonian as momenta have changed
        E_old = E
        # Hold = E + 0.5*(p*p')
        H_old = E
        for i in range(n_params):
            H_old += 0.5 * p[i]*p[i]

        # Decide on window offset, if windowed HMC is used
        if opt_window > 1:
            # window_offset=fix(opt_window*rand(1));
            window_offset = int(opt_window * random.rand())

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
                    for i in range(n_params):
                        x[i] = x_old[i]
                    for i in range(n_params):
                        p[i] = p_old[i]
                    n = window_offset

                # set direction for forward steps
                E = E_old
                H = H_old
                direction = 1
                stps = direction
            else:
                if (n*direction+1 < opt_window) or (n > (opt_steps-opt_window)):
                    # State in the accept and/or reject window.
                    stps = direction
                else:
                    # State not in the accept and/or reject window.
                    stps = opt_steps - 2 * (opt_window - 1)

                # First half-step of leapfrog.
                # p = p - direction*0.5*epsilon.*feval(gradf, x, varargin{:});
                logp, grad = fun(asarray(x), *args)
                for i in range(n_params):
                    p[i] +=  direction * 0.5 * epsilon * grad[i]
                for i in range(n_params):
                    x[i] += direction*epsilon*p[i]

                # Full leapfrog steps.
                # for m = 1:(abs(stps)-1):
                for m in range(int_abs(stps)-1):
                    # p = p - direction*epsilon.*feval(gradf, x, varargin{:});
                    logp, grad = fun(asarray(x), *args)
                    for i in range(n_params):
                        p[i] +=  direction * 0.5 * epsilon * grad[i]
                    for i in range(n_params):
                        x[i] += direction * epsilon * p[i]

                # Final half-step of leapfrog.
                # p = p - direction*0.5*epsilon.*feval(gradf, x, varargin{:});
                logp, grad = fun(asarray(x), *args)
                E = -logp
                for i in range(n_params):
                    p[i] += direction * 0.5 * epsilon * grad[i]

                # H = E + 0.5*(p*p');
                H = E
                for i in range(n_params):
                    H += 0.5 * p[i] * p[i]

                n += stps

            if opt_window != opt_steps+1 and n < opt_window:
                # Account for state in reject window.  Reject window can be
                # ignored if windows consist of the entire trajectory.
                if not have_rej:
                    rej_free_energy = H
                else:
                    rej_free_energy = -addlogs(-rej_free_energy, -H)

                if not have_rej or random.rand() < exp(rej_free_energy-H):
                    E_rej = E
                    for i in range(n_params):
                        x_rej[i] = x[i]
                    for i in range(n_params):
                        p_rej[i] = p[i]
                    have_rej = 1

            if n > (opt_steps-opt_window):
                # Account for state in the accept window.
                if not have_acc:
                    acc_free_energy = H
                else:
                    acc_free_energy = -addlogs(-acc_free_energy, -H)

                if not have_acc or random.rand() < exp(acc_free_energy-H):
                    E_acc = E
                    for i in range(n_params):
                        x_acc[i] = x[i]
                    for i in range(n_params):
                        p_acc[i] = p[i]
                    have_acc = 1

        # Acceptance threshold.
        a = exp(rej_free_energy - acc_free_energy)

        if return_diagnostics and k >= 0:
            j = k
            for i in range(n_params):
                diagn_pos[j,i] = x_acc[i]
                diagn_mom[j,i] = p_acc[i]
            diagn_acc[j] = a

        if opt_display:
            print('New position is\n', np.asarray(x))

        # Take new state from the appropriate window.
        if a > random.rand():
            # Accept
            E = E_acc
            for i in range(n_params):
                x[i] = x_acc[i]
            for i in range(n_params):
                p[i] = -p_acc[i]  # Reverse momenta
            if opt_display:
                print('Finished step %4d  Threshold: %g\n'%(k,a))
        else:
            # Reject
            if k >= 0:
                n_reject = n_reject + 1

            E = E_rej
            for i in range(n_params):
                x[i] = x_rej[i]
            for i in range(n_params):
                p[i] = p_rej[i]
            if opt_display:
                print('  Sample rejected %4d.  Threshold: %g\n'%(k,a))

        if k >= 0:
            j = k
            # Store sample
            for i in range(n_params):
                samples[j, i] = x[i]
            if return_logp:
                # Store energy
                logps[j] = -E

        # Set momenta for next iteration
        if opt_persistence:
            # Reverse momenta
            for i in range(n_params):
                p[i] = -p[i]
            # Adjust momenta by a small random amount
            randn = random.randn(n_params)
            for i in range(n_params):
                p[i] = alpha*p[i] + salpha*randn[i]
        else:
            # Replace all momenta
            randn = random.randn(n_params)
            for i in range(n_params):
                p[i] = randn[i]

        k += 1

    return n_reject


cdef inline double addlogs(double a, double b):
    if a > b:
        return a + log(1 + exp(b-a))
    else:
        return b + log(1 + exp(a-b))


cdef inline int int_abs(int a):
    return a if a > 0 else -a


def find_first(npy_uint8[:, :] X):
    """Find the index of the first element of X along axis 0 that
    evaluates to True.
    """
    cdef npy_intp i, j
    cdef npy_intp n = X.shape[0]
    cdef npy_intp m = X.shape[1]
    cdef npy_intp[::1] out = zeros(m, dtype=np.intp)

    for j in range(m):
        out[j] = -1
        for i in range(n):
            if X[i, j]:
                out[j] = i
                break
    return np.asarray(out)
