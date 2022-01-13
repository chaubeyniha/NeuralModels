from scipy.stats import norm
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt

# Signal Detection Theory

# function to turn 'proportions' into z-scores
Z = norm.ppf 

def sdt(hit_rate, fa_rate):
    """
    This function takes a hit rate and a false alarm rate, and computes d_prime and c
    """
    out = dict()
    out['d_prime'] = Z(hit_rate) - Z(fa_rate)
    out['criterion'] = -(Z(hit_rate) + Z(fa_rate))/2
    return out

# plotting Gaussians for figures
def draw_gauss(m,  s, ax, c, label='', rnge=[-5,5]):
    """
    draws a gaussian at x, with sd s in the specified axis using color c
    """
    x = np.linspace(rnge[0], rnge[1], 1000)
    ax.plot(x, norm.pdf(x, loc=m, scale=s), c=c, label=label)

# Example draw figure
s = sqrt(2)
mean_diff = 2 # if there is no distance (d' = 0) then mean_diff is set to zero
criterion = 1.6

# make a figure
f, ax = plt.subplots(1, 1, figsize=(16,6))
ax.set_title('Hit Rate: , False Alarm Rate: ')

# draw normal distribution
draw_gauss(m=-mean_diff/2, s=s, ax=ax, c='r', label='noise')
draw_gauss(m=mean_diff/2, s=s, ax=ax, c='b', label='signal')

# plot 'c', the criterion value in the plot
ax.axvline(x=criterion,  color='k', linestyle='--', label='criterion') 
plt.legend()

# get Z-scores; reverse calculation for hit rate and false alarm rate
def hr_far(d, c):
    fa = 1-norm.cdf(c, loc=-d/2)
    h = 1-norm.cdf(c, loc=d/2)
    return h, fa

# Plot ROC Curve
cs = np.linspace(-3,3,100,endpoint=True) # evenly space numeber as array

f, ax = plt.subplots(1, 1, figsize=(8,8))
ax.set_title('ROC curve!')
ax.set_xlabel('False alarm Rate')
ax.set_ylabel('Hit Rate')

hs_fas = np.array([hr_far(0.5, c) for c in cs])
plt.plot(hs_fas[:,1], hs_fas[:,0], label="d' = 0.5")
hs_fas = np.array([hr_far(1.0, c) for c in cs])
plt.plot(hs_fas[:,1], hs_fas[:,0], label="d' = 1.0")
hs_fas = np.array([hr_far(2, c) for c in cs])
plt.plot(hs_fas[:,1], hs_fas[:,0], label="d' = 2.0")
hs_fas = np.array([hr_far(3.0, c) for c in cs])
plt.plot(hs_fas[:,1], hs_fas[:,0], label="d' = 3.0")

plt.legend()
