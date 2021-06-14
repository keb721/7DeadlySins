###########################################################
#                                                         #
#    Code to Determine Percentage of Runs Crystallised    #
#                                                         #
###########################################################


# Import relevant packages
# ------------------------

import numpy as np
import os
import glob
import argparse
import warnings
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Create optional command line arguments
# --------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("-i",  "--filebase",    help="Non-unique part of file name. Default = 'cryst'", type=str)
parser.add_argument("-e", "--entries",      help="Number of data points in the file. Default=50001", type=int)   
parser.add_argument("-hl", "--headernum",   help="Number of header lines to be ignored. Default=375+10N (partition log file)", type=int)
parser.add_argument("-s",  "--crystalsize", help="Number of atoms in fit needed to be regarded as crystalline. Default = num_atoms", type=int)
parser.add_argument("-t",  "--timestep",    help="Timestep of simulation. Default = 0.002", type=float)
parser.add_argument("-o",  "--offset",      help="Shift the start of the fit to the first value where this percentage have crystallised. Default = 0", type=float)
parser.add_argument("-f",  "--folds",       help="Number of folds for bootstrap anaylis. Default = 5", type=int)

args = parser.parse_args()


# Define simulation properties
# ----------------------------

num_atoms    = 4000           
crystal_size = num_atoms if args.crystalsize is None else args.crystalsize
headernum    = 380       if args.headernum   is None else args.headernum
basename     = "cryst"   if args.filebase    is None else args.filebase     # Filename of data
timestep     = 0.002     if args.timestep    is None else args.timestep 
folds        = 5         if args.folds       is None else args.folds      
numentries   = 50001     if args.entries     is None else args.entries      # Number of data points in file


if en_drop < 0.3:
    print("For energy differences less than 0.3, it is likely that systems which did not crystallise can be filtered out.")

scale        = 1000              # Define a scale to prevent numerical overflow
eps          = 1e-16             # Define a scale to prevent finite precision errors 
tolerance    = 50                # Tolerance in where nucleation happens
hlfnum       = 0.5*num_atoms


# Extract simulation data
# -----------------------


f   = glob.glob(basename+"*")     # Lists all files of type basename

print("There are ", len(f), " files.")

for i in range(folds):

    tmp = f[i::folds]

    # Define array of times to determine if crystalline or not
    # --------------------------------------------------------

    times = np.linspace(0, 5000000*timestep/scale, 5001)
    cryst = np.zeros((5001))
    
    outdata = np.vstack((times, cryst))
    
    maxtimes = []
    halfmax  = []
    derivhlf = []

    for q, entry in enumerate(tmp):

        # Generate data
        
        try:
            data = np.genfromtxt(entry, skip_header=headernum, max_rows=numentries)
        except ValueError:
            data = np.genfromtxt(entry, skip_header=headernum, skip_footer=1)
            
        steps, T, Ep, V, sz = np.split(data, 5, axis=1)
    

        if np.isnan(steps[0]):
            data = np.genfromtxt(entry, skip_header=headernum+1, max_rows=numentries)
            steps, T, Ep, V, sz = np.split(data, 5, axis=1)

        steps = steps - steps[0]
        steps = steps.flatten()
        steps = steps*timestep / scale           # Prevent overflow of exponential function

        sz    = sz.flatten()


        # ============ #
        # Produce fits #
        # ============ #

        # Define fitting curve
        # --------------------

        def fit(x, a, b): 
            y = num_atoms/(1+np.exp(-b*(x-a)))  
            return y
 
        dataset   = sz
        midpt     = hlfnum
        post_crit = np.where(dataset > midpt)

        q6_nuc    = False if len(post_crit[0])==0            else True


        if  not q6_nuc:
            continue       #Goes to next value in list - neither method detected an event

        # Fitting only seems to work if a sensible bound is specified for the "a" coordinate
        # Find this
        # ---------

        u_b_num   = post_crit[0][0] + tolerance          # Index of bounds
            
        try:
            u_b_value = steps[u_b_num]                       # x value of bounds
        except IndexError:
            print(entry)
            continue
        l_b_num   = post_crit[0][0] - tolerance          # Index of lower bound
        l_b_value = steps[l_b_num]                       # x value of lower bound
            
        warnings.filterwarnings("error")
            
        try:
            par, pcov = curve_fit(fit, steps, dataset, bounds=([l_b_value, -np.inf],[u_b_value, np.inf]))
            f_dataset = dataset
            f_steps   = steps
            
        except RuntimeWarning:
            print("Overflow in data. Considering only area around the nucleation event")
            par, pcov = curve_fit(fit, steps[l_b_num-1000:u_b_num+1000], dataset[l_b_num-1000:u_b_num+1000], bounds=([l_b_value, -np.inf],[u_b_value, np.inf]))
            
            f_steps   = steps[l_b_num-1000:u_b_num+1000]
            f_dataset = dataset[l_b_num-1000:u_b_num+1000]


            # Define fully crystalline as where the sigmoid reaches 1
            # Find the time it takes to reach this
            # ------------------------------------
            ys = fit(steps, *par)
            halfmax.append(par[0]*scale)
        
            
            q6_steps = steps - par[0]
            np.savetxt("q6_shifted_"+entry, np.transpose(np.vstack((q6_steps*scale, sz))))



    np.savetxt("LJ_068_half_q6_f"+str(i)+"_N"+str(len(tmp))+".dat", halfmax)
