from __future__ import print_function, division, absolute_import
import numpy as np
import matplotlib.pyplot as pp
from pyhmc.tests.test_autocorr2 import generate_AR1
from pyhmc import integrated_autocorr1, integrated_autocorr2
from pyhmc import integrated_autocorr3, integrated_autocorr4
from pyhmc import integrated_autocorr5, integrated_autocorr6


# https://onlinecourses.science.psu.edu/stat510/node/60
# the autocorrelation function \rho(t) = \phi^t
# \tau = 1 + sum_{t=1}^\infty phi^t
#      = 1 + 2*(1/(1-phi) - 1)
#      = 2/(1-phi) - 1
PHI = 0.98
TRUE = 2/(1-PHI) - 1
n_steps = 1000000
grid = np.logspace(2, np.log10(n_steps), 10)

tau1 = []
tau2 = []
tau3 = []
tau4 = []
tau5 = []
tau6 = []

n_trials = 10
for i in range(n_trials):
    y = generate_AR1(phi=PHI, sigma=1, n_steps=n_steps, c=0, y0=0, random_state=None)

    tau1.append([integrated_autocorr1(y[:n]) for n in grid])
    tau2.append([integrated_autocorr2(y[:n]) for n in grid])
    tau3.append([integrated_autocorr3(y[:n]) for n in grid])
    tau4.append([integrated_autocorr4(y[:n]) for n in grid])
    tau5.append([integrated_autocorr5(y[:n]) for n in grid])
    tau6.append([integrated_autocorr6(y[:n]) for n in grid])

pp.errorbar(grid, y=np.mean(tau1, axis=0), yerr=np.std(tau1, axis=0), c='b',    label='tau 1')
pp.errorbar(grid-1, y=np.mean(tau2, axis=0), yerr=np.std(tau2, axis=0), c='r',    label='tau 2')
pp.errorbar(grid-5, y=np.mean(tau3, axis=0), yerr=np.std(tau3, axis=0), c='g',    label='tau 3')
pp.errorbar(grid-10, y=np.mean(tau4, axis=0), yerr=np.std(tau4, axis=0), c='gold', label='tau 4')
pp.errorbar(grid-20, y=np.mean(tau5, axis=0), yerr=np.std(tau5, axis=0), c='m',    label='tau 5')
pp.errorbar(grid-30, y=np.mean(tau6, axis=0), yerr=np.std(tau6, axis=0), c='cyan', label='tau 6')


pp.plot(grid, [TRUE]*len(grid), 'k-')
pp.xscale('log')
pp.legend(loc=2, fontsize=14)
pp.savefig('Pyplots.pdf')
