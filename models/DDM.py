from scipy.stats import norm
from math import exp,sqrt
import numpy as np
import pandas as pd
from matplotlib import cm

# Drift Difussion Model

import matplotlib.pyplot as plt
# matplotlib inline

# 'a' is the threshold, 'v' is the drift rate, 't'= time to reach signal in brain
# 'z' is the prepotent bias, 'vs' is the noise
def DDM_trial(v, vs, z, t, a, max_time=5000, full_return=False):
    """
    A very course implementation of a DDM trial simulation.
    Parameters are named according to the figure above.
    Implementation only looks at ssignal data (saying yes and no)
    """
    # z is the static starting point
    drift = z
    
    # if false then trajectory is drift (bias)
    if full_return:
        trajectory = [drift]
        
    # for ms in the range of time - max limit
    for ms in np.arange(t, max_time+t):
        # here we iterate over time so the drift rate is added
        # increase the bias with random value plus drift rate
        drift += np.random.randn()*vs + v
        
        # if false then add new drift to list
        if full_return:
            trajectory.append(drift
                              
        # if the drift is above/equal to threshold or smaller than zero
        # return answer with response time, trajectory
        if drift >= a: # drift hits upper bound
            if full_return:
                return {'answer': 0, 'rt': ms, 'trajectory': trajectory}    
            return {'answer': 0, 'rt': ms}
        elif drift <= 0: # drift hit lower bound
            if full_return:
                return {'answer': 1, 'rt': ms, 'trajectory': trajectory}  
                              
            return {'answer': 1, 'rt': ms}

# Example Plot
a = 1
z = a/2
t = 150
v = 0.00125 
vs = 0.025
n_trials = 250

# set up the figures
f, ss = plt.subplots(2,1, figsize=(16,6), sharex=True)
ss[0].set_xlim([0,1500])
ss[1].set_xlim([0,1500])
ss[1].set_xlabel('time [ms]')
ss[0].set_ylabel('integrated signal')
ss[1].set_ylabel('count')

# simulate a bunch of trials & plot them
trials = []
                              
for x in range(n_trials):
    trial = DDM_trial(v=v, vs=vs, z=z, t=t, a=a, full_return=True)
                              
    if trial['answer'] == 0:
        ss[0].plot(np.arange(len(trial['trajectory']))+t, trial['trajectory'], alpha=0.3, c='b')
    else:
        ss[0].plot(np.arange(len(trial['trajectory']))+t, trial['trajectory'], alpha=0.6, c='r')  
    trials.append({'answer': trial['answer'], 'rt': trial['rt']}) 

# look at influence of drift rate 'v'
different_vs = np.linspace(0.000125, 0.0025, 9)

basic_trials = [pd.DataFrame([DDM_trial(v=v, vs=vs, z=z, t=t, a=a, full_return=False) 
                             for x in range(n_trials)]) for v in different_vs]

f, ss = plt.subplots(len(different_vs),1, figsize=(12,24), sharex=True)
                              
for splt in range(len(different_vs)):
    ss[splt].set_xlim([0,2000])
    ss[splt].hist(basic_trials[splt][basic_trials[splt]['answer']==0]['rt'], alpha=0.5, color='b', bins=np.linspace(0,2000,50))
    ss[splt].hist(basic_trials[splt][basic_trials[splt]['answer']==1]['rt'], alpha=0.75, color='r', bins=np.linspace(0,2000,50))
    ss[splt].set_ylabel('count');
    ss[splt].set_title('Drift Rate is ' +str(different_vs[splt]));
ss[-1].set_xlabel('time [ms]')

# look at influence bias 'z'
different_zs = np.linspace(0, 1, 9)

# for 250 trials, repeat len(z)
basic_trials = [pd.DataFrame([DDM_trial(v=v, vs=vs, z=z, t=t, a=a, full_return=False) 
                             for x in range(n_trials)]) for z in different_zs]

f, ss = plt.subplots(len(different_zs),1, figsize=(12,24), sharex=True)
for splt in range(len(different_zs)):
    ss[splt].set_xlim([0,2000])
    ss[splt].hist(basic_trials[splt][basic_trials[splt]['answer']==0]['rt'], alpha=0.5, color='b', bins=np.linspace(0,2000,50))
    ss[splt].hist(basic_trials[splt][basic_trials[splt]['answer']==1]['rt'], alpha=0.75, color='r', bins=np.linspace(0,2000,50))
    ss[splt].set_ylabel('count');
    ss[splt].set_title('Bias is ' +str(different_zs[splt]));
ss[-1].set_xlabel('time [ms]')
