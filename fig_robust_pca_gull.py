r"""
Bivariate Gaussian: Robust Parameter Estimation
-----------------------------------------------
Figure 3.23.

An example of computing the components of a bivariate Gaussian using a sample
with 1000 data values (points), with two levels of contamination. The core of
the distribution is a bivariate Gaussian with
:math:`(\mu_x, \mu_y, \sigma_1, \sigma_2, \alpha) = (10, 10, 2, 1, 45^\odot)`
The "contaminating" subsample contributes 5% (left) and 15% (right) of points
centered on the same :math:`(\mu_x, \mu_y)`,
and with :math:`\sigma_1 = \sigma_2 = 5`.
Ellipses show the 1- and 3-sigma contours. The solid lines correspond to the
input distribution. The thin dotted lines show the nonrobust estimate, and the
dashed lines show the robust estimate of the best-fit distribution parameters
(see Section 3.5.3 for details).

*ADDENDUM, December 19, 2013 by gully

I edited the figure to demonstrate some features and try my hand at matplotlib.
Specifically, I wanted to know more about the "robust estimate" and its limits.
Exactly how much better is the robust compared to the nonrobust.

"""
# Author:Jake VanderPlas
# Edited: Michael Gully-Santiago
#         December 19, 2013
# License: BSD
#   The figure produced by this code is published in the textbook
#   "Statistics, Data Mining, and Machine Learning in Astronomy" (2013)
#   For more information, see http://astroML.github.com
#   To report a bug or issue, use the following forum:
#    https://groups.google.com/forum/#!forum/astroml-general
import numpy as np
from scipy import stats
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
from astroML.stats import fit_bivariate_normal
from astroML.stats.random import bivariate_normal

# percent sign needs to be escaped if usetex is activated
import matplotlib
if matplotlib.rcParams.get('text.usetex'):
    pct = r'\%'
else:
    pct = r'%'

#----------------------------------------------------------------------
# This function adjusts matplotlib settings for a uniform feel in the textbook.
# Note that with usetex=True, fonts are rendered with LaTeX.  This may
# result in an error if LaTeX is not installed on your system.  In that case,
# you can set usetex to False.

# from astroML.plotting import setup_text_plots
# setup_text_plots(fontsize=16, usetex=True)

N = 1000

sigma1 = 2.0
sigma2 = 0.2
mu = [10, 10]
# it's easier to visualize the x and y sigmas with alpha=0.0
alpha_deg = 0.0 #used to be 45 deg...

alpha = alpha_deg * np.pi / 180

#------------------------------------------------------------
# Draw N points from a multivariate normal distribution
#
#   we use the bivariate_normal function from astroML.  A more
#   general function for this is numpy.random.multivariate_normal(),
#   which requires the user to specify the full covariance matrix.
#   bivariate_normal() generates this covariance matrix for the
#   given inputs

np.random.seed(0)
X = bivariate_normal(mu, sigma1, sigma2, alpha, N)

#------------------------------------------------------------
# Create the figure showing the fits
fig = plt.figure(figsize=(8, 8))
fig.subplots_adjust(left=0.1, right=0.90, wspace=0.25,
                    bottom=0.1, top=0.9, hspace=0.3)


ax = fig.add_subplot(2, 2, 0)

# add outliers distributed using a bivariate normal.

x, y = X.T

# scatter the points
ax.scatter(x, y, s=2, lw=0, c='k', alpha=0.5)

# Draw elipses showing the fits
for Nsig in [1, 3]:
	# True fit
	E = Ellipse((10, 10), sigma1 * Nsig, sigma2 * Nsig, alpha_deg,
				ec='k', fc='none')
				
	ax.add_patch(E)
	
ax.set_xlim(5.5, 14.5)
ax.set_ylim(5.5, 14.5)
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')


# We'll create ten levels of contamination, but not show each one.
# Instead, we will show the fit sigma versus contamination level.
cont_frac = [0.0001, 0.001, 0.003, 0.006, 0.01, 0.03, 0.05, 0.07, 0.10, 
 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]
n_els = len(cont_frac)
sigfit_rob = np.ones(n_els)
sigfit_nonrob = np.ones(n_els)
sigfit2_rob = np.ones(n_els)
sigfit2_nonrob = np.ones(n_els)
alpfit_rob = np.ones(n_els)
alpfit_nonrob = np.ones(n_els)
ax1 = fig.add_subplot(2, 2, 1)
ax2 = fig.add_subplot(2, 2, 3)
ax3 = fig.add_subplot(2, 2, 2)

for i, f in enumerate(cont_frac):
	
	# add outliers distributed using a bivariate normal.
	X[:int(f * N)] = bivariate_normal((10, 10), 2, 4, 45 * np.pi / 180., int(f * N))
	
	x, y = X.T
	
	# compute the non-robust statistics
	(mu_nr, sigma1_nr, sigma2_nr, alpha_nr) = fit_bivariate_normal(x, y, robust=False)
	sigfit_nonrob[i] = sigma1_nr
	sigfit2_nonrob[i] = sigma2_nr	
	alpfit_nonrob[i] = alpha_nr
	
	# compute the robust statistics
	(mu_r, sigma1_r, sigma2_r, alpha_r) = fit_bivariate_normal(x, y, robust=True)
	sigfit_rob[i] = sigma1_r
	sigfit2_rob[i] = sigma2_r
	alpfit_rob[i] = alpha_r

# scatter the points
for i in range(n_els):
  print "%g\t%g\t%g" %(cont_frac[i], sigfit_rob[i], alpfit_rob[i])

ax1.plot(cont_frac, sigfit_rob, '-k', label='Robust')
ax1.plot(cont_frac, sigfit_nonrob, ':k', label='Nonrobust')
ax3.plot(cont_frac, sigfit2_rob, '-b')
ax3.plot(cont_frac, sigfit2_nonrob,':b')
ax1.plot(cont_frac, sigma1*np.ones(n_els), '--r', label='Actual')
ax3.plot(cont_frac, sigma2*np.ones(n_els), '--g')



ax1.set_xlabel('Contamination Fraction')
ax1.set_ylabel('$\sigma_x$')
ax3.set_xlabel('Contamination Fraction')
ax3.set_ylabel('$\sigma_y$')

leg = ax1.legend(loc='best', fancybox=True)
leg.get_frame().set_alpha(0.5)

ax2.plot(cont_frac, alpfit_rob, '-k', label='Robust')
ax2.plot(cont_frac, alpfit_nonrob, ':k', label='Nonrobust')
ax2.plot(cont_frac, alpha*np.ones(n_els), '--r', label='Actual')

ax2.set_xlabel('Contamination Fraction')
ax2.set_ylabel(r'$\alpha$')

leg = ax2.legend(loc='best', fancybox=True)
leg.get_frame().set_alpha(0.5)
	
plt.show()
