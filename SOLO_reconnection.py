# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 15:46:47 2021

@author: naisf
"""

import datetime
import numpy               as np
import matplotlib.gridspec as gridspec
import scipy.constants     as cst

from matplotlib   import pyplot as plt
from tqdm         import tqdm
from astropy.time import Time
from scipy.signal import convolve
from functools    import partial

import fun

as_strided = np.lib.stride_tricks.as_strided 

lw      = .5
alpha   = 1
police  = 14 
i_fig   = 1
plt.close('all')

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
""" Load the data """
#______________________________________________________________________________

data        = np.loadtxt('./data_demo')    
tt_all      = Time(data[0], format='jd')           # Time Vector  
data        = data[1:]                             # Loaded data
                                                               
dt          = tt_all[1:] - tt_all[:-1]              # Time step
dt.format   = 'sec'
dt          = np.mean(dt.value)
tt = tt_all.datetime         
                                                   # Retrieve the relevant variables
B, B_mag, n_p, V, V_mag, T, P, qf = fun.read_data(data) 

                                                   # ! Important ! 
                                                   # Check that the strides of B and v are
                                                   # (8, 24) 
                                                   # (Otherwise, the algorithm does not work)
if B.strides[0]!=8:
    B = B[:, np.ones_like(B[0], dtype=bool)]
    V = V[:, np.ones_like(B[0], dtype=bool)]
        
N_points    = len(tt)                               # Number of measurements

Va          = fun.Alfven_speed(B, B_mag, n_p, P)    # Alfvén speed computation

# Dot products
V2  = np.einsum('k...,k...->...', V , V )
Va2 = np.einsum('k...,k...->...', Va, Va)
VVa = np.einsum('k...,k...->...', V , Va)

# Multi-dimensional convolution operator
conv_valid = partial(convolve, mode='valid')
vconvolve = np.vectorize(conv_valid, signature='(n),(m)->(k)')


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
''' Inputs '''
#______________________________________________________________________________


sigma_0     = 2. #km/s
epsilon     = 0.1 #km/s / points
prior       = 0.01
dV_min      =.3
current_lim = 0.04

print('__________________________________________________')
print('___________________ PARAMETERS ___________________')
print('sigma = ' + str(sigma_0))
print('eps = ' + str(epsilon))
print('prior = ' + str(prior))
print('dV = ' + str(dV_min))
print('current = ' + str(current_lim))



scale   = np.linspace(25,850,100)         # Scales (in seconds) to span
print(
      '\nInvestigating scales from ' 
    + str(scale[0])  + ' to ' 
    + str(scale[-1]) + ' seconds'
)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
""" Test the Walen relation """
#______________________________________________________________________________

tt_jet      = []                          # Initialise the jet list

for j in tqdm(range(len(scale))):         # Sweep the different scales
    
    h = int(scale[j] / dt)                # Number of points in the window (should be ODD)
    if h % 2 == 0:
        h += 1     
   
    # Define the noise profile sigma
    window  = np.linspace(-h/2, h/2, h)
    sigma   = sigma_0 + epsilon * h * (1 - np.exp(-0.5 * window**2 / h**2))
    ker_pp  = 0.5 / sigma**2
    ker_mp  = np.where(window >  0, ker_pp, -ker_pp)
    
    all_V = np.vstack((V, Va, VVa))
    all_V_pp, all_V_mp = [vconvolve(all_V, kernel) for kernel in [ker_pp, ker_mp]]
    
    # Likelihoods computation    
    valid   = np.arange(h//2, N_points - h//2)
    def Likelihood(kernel, V_k, Va_k, VVa_k) :
        return 2 * np.squeeze(VVa_k
        + VVa[valid] * np.sum(kernel)
        - np.einsum('k...,k...->...', V_k, Va[:, valid]) 
        - np.einsum('k...,k...->...', Va_k, V[:, valid])
        )
        
    L_pp = Likelihood(ker_pp, *np.split(all_V_pp, (3, 6)))
    L_pm = Likelihood(ker_mp, *np.split(all_V_mp, (3, 6)))
        
    # Posterior computation
    post_ratio = prior / (1 - prior) * np.hstack(
        (np.ones(h//2),
         np.cosh(L_pm/h) / np.cosh(L_pp/h),
         np.ones(h//2))
    )
        
    #                                       # Reshape the vectors into (..., h, N-h+1) Arrays
    #                                       # to facilitate the analysis
    # Va_reshaped = as_strided(Va, (3,h,N_points-h+1), (8, 24, 24))    
    # V_reshaped  = as_strided(V,  (3,h,N_points-h+1), (8, 24, 24))
    # tt_reshaped = as_strided(tt, (h,N_points-h+1),   (8, 8))
    
    #                                       # Substract the central value for each window
    # dV          = V_reshaped - V_reshaped[:, h//2, :][:, None]
    # dVa         = Va_reshaped - Va_reshaped[:, h//2, :][:, None]       
    
    #                                       # Compute the posterior ratio        
    # post_ratio  = fun.compare_posteriors(prior, dVa, dV, h, sigma)    
    
                                          # Memorise the post ratio of extreme scales 
                                          #(for plotting purposes)
    if j==0:
        post_ratio_min_scale  = post_ratio 
    elif j==len(scale)-1:
        post_ratio_max_scale  = post_ratio 

                                         # find potential jets 
                                         # (post_ratio>1 over two conescutive points)
    i_jet_temp  = np.argwhere(post_ratio > 1)[:, 0]    
    i_jet       = fun.consecutive(i_jet_temp, 2)
    
    n_jets      = len(i_jet)             # Number of potential jets flagged
    if n_jets != 0:
        if len(i_jet[0]) == 0:
            n_jets = 0
            
                                         
    for p in range(n_jets):              # Check the flagged intervals
                                      
        add_points  = h - len(i_jet[p])  # adjust to window length
        i_0         = i_jet[p][0] - add_points//2
        i_f         = np.nanmin((i_jet[p][-1] + add_points//2 + 1, len(tt) - 1))
        
                                         # Rotate to the lmn frame      
        lmn_matrix  = np.asarray(
            fun.mva(B[:, i_0: i_f].T, noprint=True).results['Matrice_Passage']
        )
        B_mva       = np.dot(lmn_matrix.T, B[:,i_0:i_f])
        V_mva       = np.dot(lmn_matrix.T, V[:,i_0:i_f])                     
        
        #If the interval was not detected at lower scales
        if fun.not_detected(tt_jet, tt, i_0, i_f):
            
            #If B_L reverses consistently over the interval
            if fun.Bl_reversal(B_mva[0]):
                if fun.stable_sign(B_mva[0], h):     
                    
                    #If a velocity jet is observed                
                    if fun.velocity_jet(V_mva, dV_min, h,):    
                        
                        #If a current sheet is observed
                        if fun.current_sheet(B_mva[0], V_mva[2], scale[j], current_lim, h):   
                            
                            #Then add the interval to the detection list
                            tt_jet.append(
                                [[Time(tt[i_0]).jd], 
                                 [Time(tt[i_f]).jd], 
                                 [np.log(np.max(post_ratio[i_jet[p]]))]]
                                )              


list.sort(tt_jet, key=lambda x: x[0])     # Sort the detected intervals
tt_jet = np.array(tt_jet)

if len(tt_jet)==0:                       # Case where no jet is detected
    N_JETS=0
    final_tt_jet = []
    
else:                                    # Else, concatenate the overlapping intervals
    tt_jet = tt_jet[:, :, 0]
    
    tt_jet_temp = Time(tt_jet[:, :2], format='jd').datetime
    proba_temp = tt_jet[:, 2]
    
    detection_vect = np.zeros(N_points)  # Creation of a detection vector, size (N,), 
                                         # 1 if in a jet, 0 elsewhere
    proba = np.zeros(N_points)           # initialize the associated log-posterior probability vector
    
    for i in (range(len(tt_jet[:, 0]))):
        detection_vect[
              (tt >= tt_jet_temp[i, 0]) 
            & (tt <= tt_jet_temp[i, 1])
        ] = 1
        
        proba[
              (tt >= tt_jet_temp[i, 0]) 
            & (tt <= tt_jet_temp[i, 1])
        ] = max(
            proba_temp[i], 
            max(
                proba[
                      (tt >= tt_jet_temp[i, 0]) 
                    & (tt <= tt_jet_temp[i, 1])
                ]
            )
        )   
                                         
                                          # Number of jets
    i_detection = fun.consecutive(np.argwhere(detection_vect == 1)[:,0],1)
    N_JETS = len(i_detection)
    
    final_tt_jet = []                     # Final jet timetable
    final_proba  = []                     # Final associated log-posterior probability
    
    for p in range(N_JETS):               # Re-perform the most obvious checks after concatenation:
    
        i_0         = i_detection[p][0]   # Start and End of the interval
        i_f         = i_detection[p][-1]
        
                                          # Transform to the lmn frame        
        lmn_matrix  = np.asarray(
            fun.mva(B[:, i_0: i_f].T, noprint=True).results['Matrice_Passage']
        )
        B_mva       = np.dot(lmn_matrix.T, B[:, i_0:i_f])
        V_mva       = np.dot(lmn_matrix.T, V[:, i_0:i_f]) 
        size = len(B_mva[0])       
        
        # If B_L reverses consistently over the interval
        if fun.Bl_reversal(B_mva[0]):
            if fun.stable_sign(B_mva[0], size):  
                
                # If the slopes of Vl are consistent with a velocity jet 
                pente_1 = V_mva[0, size//2] - V_mva[0,  0]
                pente_2 = V_mva[0, size//2] - V_mva[0, -1] 
                if np.sign(pente_1) * np.sign(pente_2) == 1:
                    
                    # Then add the interval to the detection list
                    final_tt_jet.append(
                        [[Time(tt[i_detection[p][0]]).jd], 
                         [Time(tt[i_detection[p][-1]]).jd]]
                    )
                    final_proba.append(
                        np.max(
                            (proba[i_detection[p][ 0]], 
                             proba[i_detection[p][-1]])
                        )
                    )

    final_tt_jet = np.array(final_tt_jet)
    final_proba  = np.array(final_proba)
    
    #Convert jd to datetime
    final_tt_jet_time = Time(final_tt_jet[:, :, 0], format = 'jd')
    final_tt_jet = final_tt_jet_time.datetime
    N_JETS = len(final_tt_jet)
    
print('__________________________________________________\n')
print('########## ! ' +str(N_JETS) + ' jet(s) detected ! ##########')
print('__________________________________________________\n')

   

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
''' Visualize the results '''
#______________________________________________________________________________

fig = plt.figure(i_fig, figsize = (13,9))
i_fig +=1
gs = gridspec.GridSpec(7,1)

ax1 = fig.add_subplot(gs[0])
# ax1.set_title(str(N_JETS) + ' jet(s) detected !')
ax1.plot(tt, B[0],  lw=lw, c='b', alpha = alpha, label = r'$B_R$')
ax1.plot(tt, B_mag, lw=lw, c='k', alpha = alpha, label = r'$B_{mag}$')
plt.setp(ax1.get_xticklabels(), visible=False)
 
ax2 = fig.add_subplot(gs[2], sharex = ax1)
ax2.plot(tt, B[1],  lw=lw, c='g', alpha = alpha, label = r'$B_T$')
ax2.plot(tt, B_mag, lw=lw, alpha = alpha, c='k')
plt.setp(ax2.get_xticklabels(), visible=False)

ax3 = fig.add_subplot(gs[4], sharex = ax1)
ax3.plot(tt, B[2],  lw=lw, c='r', alpha = alpha, label = r'$B_N$')
ax3.plot(tt, B_mag, lw=lw, alpha = alpha, c='k')
plt.setp(ax3.get_xticklabels(), visible=False)

ax4 = fig.add_subplot(gs[1], sharex = ax1)
ax4.plot(tt, V[0], lw=lw, c='b', alpha = alpha, label = r'$V_R$')
plt.setp(ax4.get_xticklabels(), visible=False)

ax5 = fig.add_subplot(gs[3], sharex = ax1)
ax5.plot(tt, V[1], lw=lw, c='g', alpha = alpha, label = r'$V_T$')
plt.setp(ax5.get_xticklabels(), visible=False)

ax6 = fig.add_subplot(gs[5], sharex = ax1)
ax6.plot(tt, V[2],   lw=lw, c='r', alpha = alpha, label = r'$V_N$')
ax1.set_xlim(tt[0], tt[-1])
plt.setp(ax6.get_xticklabels(), visible=False)

ax7 = fig.add_subplot(gs[6], sharex = ax1)
ax7.plot(tt, post_ratio,   lw=.5, c='k', alpha = alpha, label = r'$n$')
ax7.set_ylabel(r'${cm^{-3}}$', fontsize=police) 
ax7.set_yscale('log')
# ax77 = ax7.twinx()
# ax77.plot(tt, T,   lw=lw, c='purple', alpha = alpha, label = r'$T$')
# ax77.set_ylabel(r'${eV}$', fontsize=police, c='purple') 

ax1.set_xlim(tt[0], tt[-1])

[ax.set_ylabel(r'${nT}$', fontsize=police)   for ax in [ax1, ax2, ax3]]
[ax.set_ylabel(r'${km/s}$', fontsize=police) for ax in [ax4, ax5, ax6]]
[ax.axhline(0,c='grey', lw=lw)          for ax in [ax1, ax2, ax3]]
[ax.legend(prop={'size': police},  loc='center left', 
            bbox_to_anchor=(1,0.5)) for ax in [ax1, ax2, ax3, ax4, ax5, ax6]]    

alpha=.2

if N_JETS>0:    
    for i in range(final_tt_jet[:,0].size):         
        duration = Time(final_tt_jet[i,1]) - Time(final_tt_jet[i,0])
        duration.format = 'sec'       
        print(
            '#' + str(i) + ' - ' 
                + str(int(np.round(duration.value))) + 's - ' 
                + str(Time(final_tt_jet).iso[i, 0])  + ' - '
                + str(Time(final_tt_jet).iso[i, 1])  + ' - ' 
                + str(np.round(final_proba[i], 2)) 
        )
        [ax.axvspan(
            final_tt_jet[i,0],final_tt_jet[i,1], color = 'b', alpha=alpha
        ) for ax in [ax1, ax2, ax3, ax4, ax5, ax6, ax7]]

alpha = 1

[ax.tick_params(labelsize = police) for ax in [ax1, ax2, ax3, ax4, ax5, ax6, ax7]]
plt.tight_layout() 
plt.subplots_adjust(hspace=0.3)
plt.show()
