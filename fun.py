# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 16:56:15 2022

@author: naisf
"""

import numpy as np
import matplotlib
import matplotlib.gridspec as gridspec
import scipy.constants as cst
import datetime
import pandas as pd

from matplotlib import pyplot as plt
from tqdm import tqdm
from astropy.time import Time
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy import interpolate

""" Functions """

class mva(object):
	"""
	Methode de calcul du minimum variance analysis
	Input field : magnetic field or any other 3D field to apply the MVA method in NUMPY ARRAY
	X,Y,Z in rows and data in columns such as B[:,0] = all data, x component
	input field2 : table 3x2 first row is magnetosphere vector, second is magnetosheath then MVA is determined using cross product between magnetosheath and magnetosphere, such as B[0,:] = MSP vector en NUMPY ARRAY
	TO DO : option to input only 1 timeserie and MSP, MSH, current sheet time interval using decorator?)
	Results are provided in self.results
	MVA provide a transport matrix, with L,M,N in each colums of the matrix, such as :
	new_vect = old_vect([x,y,z]) * matrix, or new_vect = inv(matrix) * old_vect[[x], [y], [z]]
	"""
	def __init__(self, field, field2 = None, noprint = False):
		
		self.field = field #np.array([[x,]]) #en entrée : champ avec 3 composantes sur n mesures
		self.results = {}
		self.calc(noprint = noprint)
		#self.noprint = noprint
		if field2 is not None :
			self.field_cross = field2 # en entré : deux vecteurs de 3 composantes déjà moyenné
			self.calc_cross(noprint = noprint)

	def calc(self, noprint = False) :
		self.matrice = np.zeros([3,3])
		for i in range(3) :
			for j in range(3) :
				self.matrice[i,j] = np.mean(self.field[:,i] * self.field[:,j]) - self.field[:,i].mean() * self.field[:,j].mean()
		
		eigVal, eigVect = np.linalg.eig(self.matrice)
		Vec_N, Vec_M, Vec_L = [eigVect[:,i] for i in np.argsort(abs(eigVal))]
        
		if np.linalg.det(np.mat([Vec_L, Vec_M, Vec_N]))==-1 :
			Vec_M = - Vec_M
            
		self.results['eigen_Val'] = eigVal[np.argsort(abs(eigVal))]
		self.results['eigen_Vect'] = {'l' : Vec_L, 'm' : Vec_M, 'n' : Vec_N}
		self.results['Matrice_Passage'] = np.mat([Vec_L, Vec_M, Vec_N]).T
		
		if not noprint :
			print('Resultat par diagonalisation de Matrice : ', '\n', 'L : ', Vec_L, 'Eigen val :', self.results['eigen_Val'][2] , '\n',  'M : ',  Vec_M, 'Eigen val :', self.results['eigen_Val'][1], '\n', 'N : ', Vec_N, 'Eigen val :', self.results['eigen_Val'][0])

	def calc_cross(self, noprint = False) :
		Vec_MSP, Vec_MSH = [self.field_cross[0,:], self.field_cross[1,:]]
		Vec_N2 = np.cross(Vec_MSP, Vec_MSH) / np.linalg.norm(np.cross(Vec_MSP,Vec_MSH))
		tmp = np.cross(Vec_N2, self.results['eigen_Vect']['l'])
		Vec_M2 = (tmp / np.linalg.norm(tmp)) * np.vdot(tmp, Vec_MSP) / abs(np.vdot(tmp, Vec_MSP))
		Vec_L2 = np.cross(Vec_M2, Vec_N2)
		self.results['eigen_Vect2'] = {'l' : Vec_L2, 'm' : Vec_M2, 'n' : Vec_N2}
		self.results['Matrice_Passage2'] = np.mat([Vec_L2, Vec_M2, Vec_N2]).T
		
		if not noprint :
			print('Resultat par produit croisé SH / SP : \n', 'L : ', Vec_L2, '\n',  'M : ',  Vec_M2, '\n', 'N : ', Vec_N2)

	def hodogram(self, system = 1) :
		"""
		project field in MVA new coordinate system and plot the L vs N and M vs N graphs
		""" 
		if system == 1 : matrice = self.results['Matrice_Passage']
		elif system == 2 : matrice = self.results['Matrice_Passage2']
		else : print('Choose MVA 1 or 2')
		proj = np.array([vect *matrice for vect in self.field])[:,0]
		
		f = plt.figure()
		plt.plot(proj[:,0], proj[:,2])
		plt.plot(proj[:,1], proj[:,2])
		plt.show()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def DSplin(xin, xout, f, lis=0.0, n=0, order=1) :
    
    """
    derivation / integration / interpolation function
    credits: Pierre S. Houdayer
    
    Parameters
    ----------
    xin : ARRAY(N,)
        Input grid.
    xout : ARRAY(N,)
        Output grid.
    f : ARRAY(N,)
        Vector to interpolate
    lis : INT
        Smoothing parameter.
    n : INT
        Derivation degree (1 = df/dt, -1 = int(f))
    order : INT
        Interpolation degree (1 = linear).

    Returns
    -------
    ARRAY(N,)
        Interpolated vector

    """
    
    if xin[-1]<xin[0] :
        xin = xin[::-1]
        f = f[::-1]
    
    tck = interpolate.splrep(xin, f, s=lis, k=order)
    
    if (n>-0.5) :
      return interpolate.splev(xout, tck, der=n)
  
    else :
      tckint = interpolate.splantider(tck, abs(n))
      return interpolate.splev(xout, tckint, der=0)
  
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def resample(data, tt_temp, time_step = 4, erase_nan=True):
    """
    Parameters
    ----------
    data : ARRAY(K, N)
        Data array of K parameters containing N measurements
    tt_temp : TIME ARRAY (N,)
        Time vector
    time_step : INT, optional
        Target constant timestep (second) of the new time vector. The default is 4s.
    erase_nan : BOOL, optional
        Erase intervals where no data is available for one parameter values (True) 
        or interpolate linearly in between (False). The default is True.

    Returns
    -------
    tt_new : TIME ARRAY (N1,)
        New time vector with constant timestep
    data_new : ARRAY(K, N1)
        Resampled data array
    """    
    
    t0 = tt_temp[0].jd                                                  # Start date
    n_pts = int(
        (tt_temp[-1].jd - tt_temp[0].jd) * (24*3600//time_step)
    ) + 1                                                               # Number of points in new time vector
    tt_new = Time(
        np.linspace(t0, t0 + (n_pts-1)*(time_step/3600/24), n_pts), 
        format='jd'
    )                                                                   # new time vector
        
    i_nans = np.max(np.isnan(data[:-2]), axis=0)                        # Flag intervals with NaN values
    i_nans_new = DSplin(tt_temp.jd, tt_new.jd, i_nans)                  # Transpose these intervals to 
                                                                        # the new time frame
    i_nans_new[i_nans_new>0] = 1
    i_nans_new = np.array(i_nans_new, dtype=bool) 
    
    data_new = np.zeros((len(data), n_pts))                             # Initiate new data array
    for i in tqdm(range(len(data))):   
        # Resample each parameter
        is_not_nan = np.argwhere(~np.isnan(data[i]))[:, 0]
        data_new[i] = DSplin(
            tt_temp[is_not_nan].jd, tt_new.jd, data[is_not_nan]
        )
        if erase_nan:
            data_new[i, i_nans_new] = 'NaN'                             # Put NaN values where other parameters 
                                                                        # are unavailable
        
    return tt_new, data_new, i_nans_new  

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def read_data(data):
    """
    Parameters
    ----------
    data : ARRAY(K, N)
        Data array of K parameters containing N measurements.
               
    Returns
    -------
    ARRAYS (N,)
        Parameters arrays.
    """
    n_p     = data[0, ]                      #Proton density
    V       = data[1:4, ]                    #Proton velocity vector
    V_mag   = np.linalg.norm(V, axis=0)      #Proton speed
    T       = data[4, ]                      #Proton temperature
    P       = data[5:11, ]/cst.eV            #Pressure tensor
    B       = data[11:14, ]                  #Magnetic field vector
    B_mag   = data[14, ]                     #Magnetic field magnitude
    qf      = data[15, ]                     #PAS quality factor
    return B, B_mag, n_p, V, V_mag, T, P, qf
    
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def rotate_P(B, P):
    """
    Rotates the pressure tensor in the magnetic field frame

    Parameters
    ----------
    B : ARRAY (3, N,)
        Magnetic Field Vector.
    P : ARRAY (6, N,)
        Pressure tensor terms Pxx, Pyy, Pzz, Pxy, Pxz, Pyz.

    Returns
    -------
    P_in_B_frame : ARRAY(N, 3, 3)
        Pressure tensor in the magnetic field frame.

    """    
    #bframe
    b1 = B / np.linalg.norm(B, axis=0)               
    b2 = np.cross(np.array([0, 0, 1]), b1, axisb = 0).T
    b2 = b2 / np.linalg.norm(b2, axis=0)
    b3 = np.cross(b1, b2, axisa = 0, axisb = 0).T
    
    rot_matrix      = np.array([b1, b2, b3]).T
    rot_matrix_inv  = np.linalg.inv(rot_matrix)
    
    P_tensor        = np.array(
        [[P[0], P[3], P[4]],
         [P[3], P[1], P[5]],
         [P[4], P[5], P[2]]]
    ).T
    
    P_in_B_frame = np.einsum(
        '...kj,...jn', 
        rot_matrix_inv, 
        np.einsum(
            '...jk,...jn', P_tensor, rot_matrix
        )
    )
    
    return P_in_B_frame

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
def Alfven_speed(B, B_mag, n_p, P, anisotropy = True):
    """
    Compute the Alfvén speed

    Parameters
    ----------
    B : ARRAY (3, N,)
        Magnetic Field Vector.
    B_mag : ARRAY (N,)
        Magnetic Field Magnitude.
    n_p : ARRAY (N,)
        Proton density
    P : ARRAY (6, N,)
        Pressure tensor terms Pxx, Pyy, Pzz, Pxy, Pxz, Pyz.
    anisotropy : BOOL, optional
        Include the anisotropy correction term. The default is True.

    Returns
    -------
    Va : ARRAY (3, N,)
        Alfvén speed Vector.
    """
    if anisotropy:                
        #Compute the anisotropy correction term
        P_in_B_frame        = rotate_P(B, P)        
        alpha_p             = (
              cst.mu_0 / (B_mag * 1e-9)**2 
            * (P_in_B_frame[:, 0, 0] - P_in_B_frame[:, 1, 1])
            * cst.eV * 1e6
        )
        alpha_p[alpha_p > 1]  = 0
    
        #Alfvén speed
        Va = (B.T * 1e-9 * np.sqrt(1 - alpha_p[:,None]) /
              np.sqrt(n_p[:,None] * 1e6 * cst.m_p * cst.mu_0) * 1e-3).T
    else:
        Va = (B.T * 1e-9 /
              np.sqrt(n_p[:,None] * 1e6 * cst.m_p * cst.mu_0) * 1e-3).T 
    return Va


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def sign_vector(h, theta):
    """
    Create a multiplication vector from theta

    Parameters
    ----------
    h : INT
        size of the output vector.
    theta : ARRAY (2,)
        parameter vector.

    Returns
    -------
    foix : ARRAY (h,)
        multiplication vector.

    """
        
    foix        = np.zeros(h)
    foix[:h//2] = theta[0]
    foix[h//2:] = theta[1]
    return foix 

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def norm_2D(U, V, sign) : 
    result = (
          np.einsum('ij...,ij...->...', U, U) 
        + np.einsum('ij...,ij...->...', V, V)
        + np.einsum('ij...,ij...->...', U, V) * 2*sign
    )
    return result


def likelihood(dVa, dV, h, sigma, theta = np.array([1, 1])):
    """
    Computes the likelihood of theta given V, Va and sigma

    Parameters
    ----------
    dVa : ARRAY (3, h, N-h+1)
        Afvén velocity vector, reshaped and equal to zero at the window's center.
    dV : ARRAY (3, h, N-h+1)
        Velocity vector, reshaped and equal to zero at the window's center.
    h : INT
        window size.
    sigma : ARRAY (N-h+1,)
        noise dispersion vector.
    theta : ARRAY (2,), optional
        parameter vector. The default is np.array([1,1]).

    Returns
    -------
    chi : ARRAY (N-h+1,)
        Likelihood of the data as a function of time.

    """    
    # # Compute the error vector
    eps = (
          (dV - sign_vector(h, theta)[None, :, None] * dVa) 
        / (2**0.5 * sigma[:, None]) 
    )
    
    # Compute the likelihood     
    chi = (
        - 1.0 * np.einsum(
            'ij...,ij...->...', 
            eps, eps, optimize=True
        )
        - 3/2 * np.sum(np.log(2 * np.pi * sigma**2))
    )
    return chi

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def compare_posteriors(prior, dVa, dV, h, sigma):
    """
    Compute the posterior probability of each model and compare them to return 
    the posterior ratio of observing a jet over not observing a jet given the data 

    Parameters
    ----------
    prior : FLOAT
        p(jet) prior probability of observing a jet.
    dVa : ARRAY (3, h, N-h+1)
        Reshaped Alfvén velocity vector, equal to zero at the window's center.
    dV : ARRAY (3, h, N-h+1)
        Reshaped velocity vector, equal to zero at the window's center.
    h : INT
        Window size.
    sigma : ARRAY (N,)
        Noise dispersion vector.

    Returns
    -------
    post_ratio : ARRAY (N,)
        posterior ratio, cf equation XXX in Fargette et al 2023.

    """
    N_points = len(dVa[0,0]) + h - 1
    
    #initialize the likelihood vectors
    Likelihood_PLUS   = np.zeros(N_points)
    Likelihood_MOINS  = np.zeros(N_points)
    Likelihood_JET_pm = np.zeros(N_points)
    Likelihood_JET_mp = np.zeros(N_points)
    
    #Compute the likelihood vectors
    Likelihood_PLUS[  h//2 : N_points-h//2] = likelihood(dVa, dV, h, sigma, np.array([ 1, 1]))          
    Likelihood_MOINS[ h//2 : N_points-h//2] = likelihood(dVa, dV, h, sigma, np.array([-1,-1]))  
    Likelihood_JET_pm[h//2 : N_points-h//2] = likelihood(dVa, dV, h, sigma, np.array([ 1,-1]))  
    Likelihood_JET_mp[h//2 : N_points-h//2] = likelihood(dVa, dV, h, sigma, np.array([-1, 1]))  
       
    #Compute the posterior ratios
    post_ratio = (
          prior / (1 - prior) 
        * (  (np.exp(Likelihood_JET_mp / h) + np.exp(Likelihood_JET_pm / h)) 
           / (np.exp(Likelihood_PLUS   / h) + np.exp(Likelihood_MOINS  / h)))
    )
      
    return post_ratio

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
def consecutive(data, min_size = 1):
    """
    Finds chains of consecutive numbers in a numpy array 

    Parameters
    ----------
    data : ARRAY (N,)
        input vector
    min_size : INT
        minimum size for a chain of consecutive numbers. The default is 1.

    Returns
    -------
    result : LIST OF ARRAYS
        list of numpy array with the consecutive numbers of data.

    """

    chains = np.split(data, np.where(np.diff(data) != 1)[0] + 1)
    result = []
    for i in range(len(chains)):
        if len(chains[i]) >= min_size:
            result.append(chains[i])    
    return result

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def not_detected(tt_jet, tt, i_0, i_f):
    """
    Check if this interval was previously detected at lower scales

    Parameters
    ----------
    tt_jet : LIST OF LISTS
        For each jet, contains the start time (jd format), end time and posterior ratio.
    tt : DATETIME ARRAY
        Time vector.
    i_0 : INT
        Start index in the time vector.
    i_f : INT
        Ending index in the time vector

    Returns
    -------
    bool
        True if not previously detected, False if already detected.

    """
    for k in range(len(tt_jet)):
        if (Time(tt[i_0]).jd < tt_jet[k][0]) & (Time(tt[i_f]).jd > tt_jet[k][1]):
            return False
    return True

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def Bl_reversal(B_l):
    """
    Check that B_l reverses over the interval

    Parameters
    ----------
    B_l : ARRAY (N,)
        B_l component of B.

    Returns
    -------
    bool
        True if B_l reverses (computend on 2 consecutive points at the beginning and end of the window).

    """
    if (np.sign(B_l[0])*np.sign(B_l[-1])==1) or (np.sign(B_l[1])*np.sign(B_l[-2])==1):
        return False
    return True

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def stable_sign(B_l, h):
    """
    Check if the sign of B_l remains the same over 0.1 * h points at the begining and end of the window

    Parameters
    ----------
    B_l : ARRAY (N,)
        B_l component of B.
    h : INT
        Window size.

    Returns
    -------
    bool
        True if the sign is stable.

    """
    for k in range(int(h*0.1)):
        if (np.sign(B_l[k])!=np.sign(B_l[0])) or (np.sign(B_l[len(B_l)-1-k])!=np.sign(B_l[-1])):
            return False
    return True

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def velocity_jet(V_mva, dV_min, h,):
    """  
    Check if a jet is observed over the window
    Parameters
    ----------
    V_mva : ARRAY(3,N)
        Alfvén velocity vector.
    dV_min : INT
        Minimum threshold on velocity increase
    h : INT
        Window length

    Returns
    -------
    bool
        True if the conditions for the observation of a velocity jet are met.
    """
    
    # Compute the slope of V_l on both sides of the jet 
    pente_1 = V_mva[0, h//2] - V_mva[0,  0]
    pente_2 = V_mva[0, h//2] - V_mva[0, -1] 
    
    # Compute the maximum velocity variation in each direction 
    dVl_max = V_mva[0].max() - V_mva[0].min()
    dVm_max = V_mva[1].max() - V_mva[1].min()
    dVn_max = V_mva[2].max() - V_mva[2].min()
      
    # If the slope signs are different, no jet
    if np.sign(pente_1) * np.sign(pente_2) ==-1:
        return False    
    
    # The maximum V_l variation should be superior to that of $V_m$ and $V_n$
    if (dVl_max < 1.*dVm_max) or (dVl_max < 1.*dVn_max):
        return False
    
    # The V_l variation bet. edges and center should be > to 30\% of the maximum $V_l$ variation. 
    if np.min((np.abs(pente_1) / dVl_max, np.abs(pente_2) / dVl_max))< dV_min:
        return False

    return True

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def current_sheet(B_l, V_n, scale, current_lim, h):
    """
    Check if a current sheet is observed

    Parameters
    ----------
    B_l : ARRAY(N, )
        B_l component of the magnetic field.
    V_n : ARRAY(N, )
        V_n component of the velocity.
    scale : FLOAT
        Window size (in seconds).
    current_lim : FLOAT
        Minimum current.
    h : INT
        Window size (in seconds).

    Returns
    -------
    bool
        DESCRIPTION.

    """
    
    #Compute the current on each jet boudary
    delta_n = np.median(V_n) * 1e3 * scale/2
    current1 = np.abs(
        (B_l[h//2] - B_l[ 0]) / (delta_n * cst.mu_0)
    )
    current2 = np.abs(
        (B_l[-1] - B_l[h//2]) / (delta_n * cst.mu_0)
    )
    
    if (current1 >= current_lim) & (current2 >= current_lim): 
        return True
    return False

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
