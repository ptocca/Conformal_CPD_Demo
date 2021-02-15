# Conformal CPD Demo
**Work In Progress** A simple interactive demonstration of Conformal Change Point Detection.

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/ptocca/Conformal_CPD_Demo/HEAD?urlpath=voila%2Frender%2FCCPD_Demo.ipynb)

Conformal Change-Point Detection applies the framework of Conformal Predictors (CP) to the problem of detecting a
change in the distribution of a sequence of variates.

Differently from conventional methods, the method does not require the knowledge of the pre-change distribution or of the post-change distribution. 
The CP framework requires however that a *Non Conformity Measure* be defined. The NCM is a function of a test observation 
and a set of observations and expresses how "dissimilar" (or "non conform") to the set of observations the test 
observation is.
The reference set of observations is generally referred to as *calibration set* and is another input to the CP framework.

ConfCPD uses CP to compute p-values (one for each test variate) that have the property of being uniformly distributed when the variates come from the same distribution as the calibration set.

Change Point Detection is then achieved by computing functions of the p-values. The functions have the properties of being martingales if the p-values are uniformly distributed. An alarm is generated when the function exceeds a threshold.
An appropriate value of the threshold is chosen so that the an acceptable rate of false alarms is produced, while detecting quickly a change in distribution.


### References
1. Algorithmic learning in a random world, V. Vovk, A. Gammerman, G. Shafer, Springer, 2005
2. Working papers 24, 26, 29 at [ALRW Site](http://www.alrw.net/)


