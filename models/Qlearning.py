import math
import scipy as sp
import scipy.stats as stats 
import numpy as np
import matplotlib 
import matplotlib.pylab as plt
import pandas as pd
import itertools as it
import scipy.optimize as op

class Simulate_QLearn(object): 
    """Simulate choice data using Q-learning algorithm(s) """ 
    
    def __init__(self, beta, alphaG, alphaL, epsilon=1e-6, tau=0):

        # Original Q-learning model with two learning rates and inverse temp param
        self.beta   = beta
        self.alphaG = alphaG
        self.alphaL = alphaL
        # parameters that are in the model but not to be fitted here...
        self.epsilon= epsilon  # forgetting rate
        self.tau = tau            # discounting rate

    def generate_experiment(self, run_count=3, n_reps_per_run=20): 
        """generate an experimental set of trials to use in simulate_RL_train """
        
        runs=[]

        ##### SIMULATE TRIALS #####
        # make simulation trials of choice options and reward -->6 runs of 60 trials 
        # repeat the elements of array 20 times
        good = np.repeat([0,2,4], n_reps_per_run) #80,70,60
        bad  = np.repeat([1,3,5], n_reps_per_run) #20,30,40

        # good and bad choice reward probabilities --> reward=0, no reward=1
        good_prob = list(zip([16,14,12], [4,6,8])) #[(16, 4), (14, 6), (12, 8)]
        # get a list of rewards in line with good_prob
        r_g = np.concatenate([np.concatenate([np.zeros(good_prob[x][0]),
                                np.ones(good_prob[x][1])]) for x in range(len(good_prob))])
        # inverse probability for bad choice
        r_b = 1-r_g 

        # simulated trial types per run 
        # outputs the good rewards & bad research for 60 trials with prob of good/bad at end
        run = pd.DataFrame(np.array([good, bad, r_g, r_b]).T, columns=['good','bad', 'r_g', 'r_b'])
            
        # shuffle and append runs(list)
        for i in range(run_count):
            runs.append(run.sample(frac=1).reset_index(drop=True))
        
        # merge runs to one simulated session
        self.sim_session = pd.concat(runs, ignore_index=True)

    def simulate_RL_train(self): 
        """ simulate choices using the original Q-learning algorithm """
        
        if not hasattr(self, 'sim_session'): # need to generate trials
            self.generate_experiment()
            
        data = self.sim_session

        #####  VARIABLES #####
        # column of 'good' into an array
        choices = np.array(data['good']).astype(int)
        # make new df with reward
        reward = data[['r_g', 'r_b']].astype(int) # 0=reward, 1=noreward
        
        prQ = np.repeat(0.5,6)
        correct = np.zeros(choices.shape[0]).astype(int)
        selection = np.zeros(choices.shape[0])
        q_chosen_sim = np.zeros(choices.shape[0])
        q_unchosen_sim = np.zeros(choices.shape[0])
        rpe_sim = np.zeros(choices.shape[0])
        r = np.zeros(choices.shape[0])
        all_Qvalues = np.zeros((6, choices.shape[0]))
        QChoices = np.zeros((len(choices),2))  

        #-----------------------------------------------------------------------#
        # 				Simulate choices and choice probabilities 				#
        #-----------------------------------------------------------------------#

        for tr in range(choices.shape[0]): 

            #Qvalues stimulus pair
            QChoice = [prQ[choices[tr]], prQ[choices[tr]+1]] 
            QChoices[tr]=QChoice
                    
            #Choice probabilities stimulus pair
            pChoice = 1/(1+np.exp(self.beta*(QChoice[1]-QChoice[0])))
            pChoice = np.array([pChoice, 1-pChoice]) 									
            pChoice = self.epsilon/2+(1-self.epsilon)*pChoice 

            #simulate choices based on stim choice probabilities 
            if tr == 0: 
                correct[tr] = np.random.multinomial(1, [0.5,0.5])[0]
            else: 
                correct[tr] = np.random.multinomial(1, pChoice)[0]

            #the simulated choice given the model; 0 is correct choice 
            simChoice=1-correct[tr] 

            #choice prob. given the model
            selection[tr]=pChoice[simChoice]
            
            #the q-value of the simulated chosen and unchosen stimulus, before updating
            q_chosen_sim[tr]=prQ[choices[tr]+simChoice]
            q_unchosen_sim[tr]=prQ[choices[tr]+1-simChoice]

            #positive learning rate
            if (simChoice==0 and reward['r_g'][tr]==0) or (simChoice==1 and reward['r_b'][tr]==0): 
                alpha = self.alphaG
            #negative learning rate 
            elif (simChoice==0 and reward['r_g'][tr]==1) or (simChoice==1 and reward['r_b'][tr]==1):
                alpha = self.alphaL
            else: 
                print('wrong reinforcement')

            #reinforcement associated with simChoice  
            if simChoice == 0: 
                r[tr]=1-reward['r_g'][tr] #r=1, reward
            else: 
                r[tr]=1-reward['r_b'][tr] #r=0, no reward

            #calculate simulated rpe 
            rpe_sim[tr] = r[tr]-prQ[choices[tr]+simChoice]

            #update stimulus Q-value 
            prQ[choices[tr]+simChoice] = prQ[choices[tr]+simChoice] \
                                            + alpha*(r[tr]-prQ[choices[tr]+simChoice])

            #decay values to initial value 
            prQ = prQ + self.tau * (0.5-prQ)
            all_Qvalues[:,tr]=prQ		
        
        #simulated results, correct simulated choice=1/incorrect=0; rewarded simulated choice=1/noreward=0
        # dataframe with simulation information
        sim_results = pd.DataFrame(np.array([choices, correct, r, selection, q_chosen_sim, 
            q_unchosen_sim, rpe_sim, QChoices[:,0]-QChoices[:,1]]).T, 
            columns=['stim_pair', 'correct_sim','reward_sim', 'select_prob_sim', 'q_chosen_sim', 
            'q_unchosen_sim', 'rpe_sim', 'qdiff_sim'])
        sim_Qvals = pd.DataFrame(np.array(all_Qvalues.T), 
            columns=['sA','sB','sC','sD','sE','sF'])
        self.sim_results = pd.concat([sim_results, sim_Qvals], axis=1)
    
    def plot_simulation(self, data_types=['qdiff_sim', 'rpe_sim'], pairs=[0,2,4], new=False):
        """
        plot the simulated timecourses of `data_types` of a full experiment, for the given pairs. 
        """
        if not hasattr(self, 'sim_results') or new:
            self.generate_experiment()
            self.simulate_RL_train()
        to_be_plotted = [self.sim_results[data_types][self.sim_results['stim_pair']==x] for x in pairs]
        f, ss = plt.subplots(1, len(pairs), figsize=(len(pairs)*6,4), sharey=True)
        ss[0].set_ylabel('Difference in Q-value between choice options\n&\nReward prediction error')
        for i, tbp in enumerate(to_be_plotted):
            tbp.plot(ax=ss[i])
            ss[i].set_title('pair ' + str(pairs[i]))
            ss[i].set_xlabel('trial #')
                            
    def plot_decision_function(self, f=False, line='r:', label=''):
        if not f:
            f = plt.figure()
        plt.plot(np.linspace(-1,1,100), 1/(1+np.exp(self.beta*(np.linspace(-1,1,100)))), line, label=label)
        plt.ylabel('p(incorrect)')
        plt.xlabel('$\Delta$ value between chosen and not-chosen stimulus')
        plt.axvline(0, lw=0.5, color='k')
        

def RL_train_NLL(theta, data): 
    """ Returns the negative log likelihood of the data given the values of theta.
    In this implementation, we fit beta, alpha gain and alpha loss, but not the forgetting and discounting rates."""	

    #parameters to fit
    beta = theta[0] * 100       # inverse gain recoded to [0-100]
    alphaG = theta[1]           # learning rate gain
    alphaL = theta[2]           # learning rate loss
    epsilon = 1e-5 			    # forgetting rate
    tau = 0 					# discounting	

    choices = np.array(data['stim_pair']).astype(int)       #recode into 0,2,4 integers - used for indexing	
    correct = 1-np.copy(data['correct_sim']).astype(int)    #0=correct,1=incorrect -> recoded/reversed to be able to fit. 
    reward = 1-np.copy(data['reward_sim']).astype(int)      #0=reward,1=noreward -> recoded/reversed to be able to fit.

    #start Q-values
    prQ0 = np.repeat(0.5,6) 
    prQ = prQ0

    #initialise Qvalue, probs & prediction error arrays
    QChoices = np.zeros((len(choices),2))  
    selection = np.zeros(len(choices))

    #loop over trials
    for tr in range(choices.shape[0]): 

        #calculate choice prob using soft max
        QChoice = [prQ[choices[tr]], prQ[choices[tr]+1]] 	#Qvalues of stimulus pair
        QChoices[tr]=QChoice

        pChoice = 1/(1+np.exp(beta*(QChoice[1]-QChoice[0])))
        pChoice = np.array([pChoice, 1-pChoice]) 									
        pChoice = epsilon/2+(1-epsilon)*pChoice 	#choice probs of stimulus pair

        selection[tr] = pChoice[correct[tr]]  	    #probability of the chosen stimulus

        #select correct learning rate
        if reward[tr] == 0: 
            alpha = alphaG
        elif reward[tr]==1: 
            alpha = alphaL

        #update stimulus Q-value
        r=1-reward[tr] #1 or 0 			
        
        # update
        prQ[choices[tr]+correct[tr]] = prQ[choices[tr]+correct[tr]] \
                                    + alpha*(r-prQ[choices[tr]+correct[tr]])

        #decay Q-values toward initial value
        prQ=prQ+tau*(0.5-prQ)

    #log likelihood 
    loglik = sum(np.log(selection)) 

    #correct for funny values
    if math.isnan(loglik): 
        loglik = -1e15  
        print('LLH is nan')
    if loglik == float("inf"):
        loglik = 1e15   
        print('LLH is inf')

    return -loglik