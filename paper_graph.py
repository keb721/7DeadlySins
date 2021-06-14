#################################################
#                                               #
#   Code for side-by-side comparison of data    #
#                                               #
#################################################

# Import relevant packages
# ------------------------

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.optimize import curve_fit
from glob import glob
from matplotlib import patches

# ======================== #
# Define system parameters #
# ======================== #

binnum = 15 # Number of bins for survival probability

# LJ data
# -------

LJT1 = 0.68                  # Temperature 
LJf1 = "LJ_068_half_q6_f*"   # Nucleation times
LJp1 = "collapse_T068/"      # Location for shifted data
LJN1 = 258                   # Number of files

LJT2 = 0.765
LJf2 = "LJ_0765_half_q6_f*"
LJp2 = "collapse_T0765/"
LJN2 = 224

LJT3 = 0.8
LJf3 = "LJ_080_half_q6_f*"
LJp3 = "collapse_T080/"
LJN3 = 823


# ==================== #
#                      # 
#     CREATE PLOTS     #
#                      #
# ==================== #

# Parameters for graph
# --------------------


layout = {'h_pad' : 2.75,
          'w_pad' : 2.25}

fntsz  = 28
mpl.rcParams.update({'text.usetex': True})

fig, ax = plt.subplots(3,2, figsize=([16, 10.6]), tight_layout=layout)


bounding_box = patches.FancyBboxPatch((-280, -2025), 990, 6475, edgecolor='k', facecolor='w', clip_on=False, in_layout=False, boxstyle='round', joinstyle='round', lw=1.5)
bounding_box1 = patches.FancyBboxPatch((-280, -2025), 990, 6475, edgecolor='k', facecolor='w', clip_on=False, in_layout=False, boxstyle='round', joinstyle='round', lw=1.5)
bounding_box2 = patches.FancyBboxPatch((-280, -2025), 990, 6475, edgecolor='k', facecolor='w', clip_on=False, in_layout=False, boxstyle='round', joinstyle='round', lw=1.5)




# -------------------- #
#     Shifted Graph    #
# -------------------- #


fs = glob(LJp1+"q6*")
data = np.genfromtxt(fs[0])
ax[0][0].plot(data[:,0], data[:,1], alpha=0.2, color='tab:blue',  label=r'$T\mbox{*}=$'+str(LJT1))
for i in range(1, len(fs)):
    data = np.genfromtxt(fs[i])
    ax[0][0].plot(data[:,0], data[:,1], alpha=0.2, color='tab:blue')

ax[0][0].set_xlabel(r'$\Delta t\mbox{*}$', fontsize=fntsz)
ax[0][0].set_ylabel(r'$q_6(n)$', fontsize=fntsz)
ax[0][0].tick_params(axis='both', which='major', labelsize=(fntsz-2))
ax[0][0].set_xlim([-200, 200])
ax[0][0].legend(loc='upper left', fontsize=fntsz-2)
transf = ax[0][0].get_yaxis_transform()
ax[0][0].annotate('a)', fontsize=32, xy=(-0.18, 3500), xycoords=transf, annotation_clip=False)
ax[0][0].add_patch(bounding_box)



fs = glob(LJp2+"q6*")
data = np.genfromtxt(fs[0])
ax[1][0].plot(data[:,0], data[:,1], alpha=0.2, color='tab:orange',  label=r'$T\mbox{*}=$'+str(LJT2))
for i in range(1, len(fs)):
    data = np.genfromtxt(fs[i])
    ax[1][0].plot(data[:,0], data[:,1], alpha=0.2, color='tab:orange')

ax[1][0].set_xlabel(r'$\Delta t\mbox{*}$', fontsize=fntsz)
ax[1][0].set_ylabel(r'$q_6(n)$', fontsize=fntsz)
ax[1][0].tick_params(axis='both', which='major', labelsize=(fntsz-2))
ax[1][0].set_xlim([-200, 200])
ax[1][0].legend(loc='upper left', fontsize=fntsz-2)
transf = ax[1][0].get_yaxis_transform()
ax[1][0].annotate('b)', fontsize=32, xy=(-0.18, 3500), xycoords=transf, annotation_clip=False)
ax[1][0].add_patch(bounding_box1)


fs = glob(LJp3+"q6*")
data = np.genfromtxt(fs[0])
ax[2][0].plot(data[:,0], data[:,1], alpha=0.2, color='tab:purple',  label=r'$T\mbox{*}=$'+str(LJT3))
for i in range(1, len(fs)):
    data = np.genfromtxt(fs[i])
    ax[2][0].plot(data[:,0], data[:,1], alpha=0.2, color='tab:purple')
ax[2][0].set_xlabel(r'$\Delta t\mbox{*}$', fontsize=fntsz)
ax[2][0].set_ylabel(r'$q_6(n)$', fontsize=fntsz)
ax[2][0].tick_params(axis='both', which='major', labelsize=(fntsz-2))
ax[2][0].set_xlim([-200, 200])
ax[2][0].legend(loc='upper left', fontsize=fntsz-2)
transf = ax[2][0].get_yaxis_transform()
ax[2][0].annotate('c)', fontsize=32, xy=(-0.18, 3500), xycoords=transf, annotation_clip=False)
ax[2][0].add_patch(bounding_box2)


# -------------------- #
#    Survival Graph    #
# -------------------- #

# Define survival probability eqs
# -------------------------------

def fit_to_J(t, J):
    expon = -1.0*t*J
    P     = np.exp(expon)
    return P


def fit_to_Jg(t, J, gamma):
    expon = pow(t*J, gamma)
    P     = np.exp(-1.0*expon)
    return P

# Errors from decomposition of fits above. Errors come from fitting

def error_Jfit(t, J, sigJ):
    dPdJ = t*np.exp(-1.0*J*t)
    sigP = dPdJ*sigJ
    return abs(sigP)

def error_Jgfit(t, J, gamma, sigJ, sigg):
    expon = pow(t*J, gamma) 
    dPdJ  = pow(t, gamma)*gamma*pow(J, gamma-1)*np.exp(-1.0*expon)
    
    P     = fit_to_Jg(t, J, gamma)
    try:
        natlJ = np.log(P*J*t)
        dPdg  = natlJ*P
    except RuntimeWarning:
        print("Warning")
        dPdg  = np.zeros(len(t))

    sigP2 = (dPdJ*sigJ)**2 + (dPdg*sigg)**2
    sigP  = np.sqrt(sigP2)

    return sigP


def checkpoint(times, binnum, Nruns, checkpoint_times):

    checkpoint_data  = np.zeros(binnum+1)
    
    for entry in times:
        overs = np.where(entry <= checkpoint_times)
        checkpoint_data[overs[0][0]:] += 1

    checkpoint_data = checkpoint_data/Nruns
    checkpoint_data = 1-checkpoint_data

    return checkpoint_data


# Define bootstrapping graph
# --------------------------

def bootstrap(LJf, binnum):
    files = glob(LJf)
    folds = len(files)
    
    max_time = 0

    for fl in files:
        tmp      = np.genfromtxt(fl)
        max_time = max(tmp) if max(tmp) > max_time else max_time
    
    timeseries = np.linspace(0, max_time, binnum+1)    

    data = np.zeros((binnum+1, folds))

    for fn, fl in enumerate(files):
        tmp_times = np.genfromtxt(fl)
        # File names are in the form "LJ_T_OP_fF_Nn.dat", where T is the temperature, OP is the order 
        # parameter, F is the fold number, and n is the number of processed files for this fold
        
        Nruns_tmp = int(fl.split("N")[1].split(".")[0])
        data[:, fn] = checkpoint(tmp_times, binnum, Nruns_tmp, timeseries)


    mean_data = np.mean(data, axis=1)
    err_data  = np.std(data, axis=1)/np.sqrt(folds)

    return timeseries, mean_data, err_data



# Generate Data
# -------------

times1, LJ_mean1, LJ_err1 = bootstrap(LJf1, binnum)
times2, LJ_mean2, LJ_err2 = bootstrap(LJf2, binnum)
times3, LJ_mean3, LJ_err3 = bootstrap(LJf3, binnum)


# Plot LJ
# -------

print("Graph 1")
ax[0][1].errorbar(times1, LJ_mean1, marker='None', linestyle='None', color='tab:blue', yerr=LJ_err1, capsize=2)
ax[0][1].fill_between(times1, LJ_mean1-LJ_err1, LJ_mean1+LJ_err1, color='tab:blue', alpha=0.2)
par1, pcov1 = curve_fit(fit_to_Jg, times1, LJ_mean1, bounds=([0, 0], [np.inf, 20]))
alpha = round(par1[1], 2)
ecov1 = np.sqrt(np.diag(pcov1))
ealph = round(ecov1[1], 2)
print("J =", par1[0], "+/-", ecov1[0])
P = fit_to_Jg(times1, *par1)
ax[0][1].plot(times1, P, color='tab:green', label=r"Fit, $\alpha=$"+str(alpha)+r'$\pm$'+str(ealph))
fit_err = error_Jgfit(times1, par1[0], par1[1], ecov1[0], ecov1[1])
ax[0][1].fill_between(times1, P-fit_err, P+fit_err, color='tab:green', alpha=0.2)
par1, pcov1 = curve_fit(fit_to_J, times1, LJ_mean1, bounds=([0], [np.inf]))
P = fit_to_J(times1, *par1)
ax[0][1].plot(times1, P, color='tab:grey', label=r"Fit, $\alpha$ constrained to 1")
fit_err = error_Jfit(times1, par1[0], np.sqrt(pcov1[0][0]))
ax[0][1].fill_between(times1, P-fit_err, P+fit_err, color='tab:grey', alpha=0.2)
ax[0][1].tick_params(axis='both', which='major', labelsize=(fntsz-2))
ax[0][1].set_xlabel(r'$t\mbox{*}$', fontsize=fntsz)
ax[0][1].set_ylabel(r'$P_{liq}(t\mbox{*})$', fontsize=fntsz)
ax[0][1].legend(fontsize=fntsz-2)
print("constrained J =", par1[0], "+/-", np.sqrt(pcov1[0][0]))




print("Graph 2")
ax[1][1].errorbar(times2, LJ_mean2, marker='None', linestyle='None', color='tab:orange', yerr=LJ_err2, capsize=2)
ax[1][1].fill_between(times2, LJ_mean2-LJ_err2, LJ_mean2+LJ_err2, color='tab:orange', alpha=0.2)
par2, pcov2 = curve_fit(fit_to_Jg, times2[1:-1], LJ_mean2[1:-1], sigma=LJ_err2[1:-1], absolute_sigma=True, bounds=([0, 0], [0.05, 1.5]))
ecov2 = np.sqrt(np.diag(pcov2))
P = fit_to_Jg(times2, *par2)
alpha = round(par2[1], 2)
ealph = round(ecov2[1], 2)
print("J =", par2[0], "+/-", ecov2[0])
ax[1][1].plot(times2, P, color='tab:green', label=r"Fit, $\alpha=$"+str(alpha)+r'$\pm$'+str(ealph))
fit_err = error_Jgfit(times2, par2[0], par2[1], ecov2[0], ecov2[1])
ax[1][1].fill_between(times2, P-fit_err, P+fit_err, color='tab:green', alpha=0.2)
par2, pcov2 = curve_fit(fit_to_J, times2[1:-1], LJ_mean2[1:-1], sigma=LJ_err2[1:-1], absolute_sigma=True, bounds=([0],[0.05]))
P = fit_to_J(times2, *par2)
ax[1][1].plot(times2, P, color='tab:grey', label=r"Fit, $\alpha$ constrained to 1")
fit_err = error_Jfit(times2, par2[0], np.sqrt(pcov2[0][0]))
ax[1][1].fill_between(times2, P-fit_err, P+fit_err, color='tab:grey', alpha=0.2)
ax[1][1].tick_params(axis='both', which='major', labelsize=(fntsz-2))
ax[1][1].set_xlabel(r'$t\mbox{*}$', fontsize=fntsz)
ax[1][1].set_ylabel(r'$P_{liq}(t\mbox{*})$', fontsize=fntsz)
ax[1][1].legend(fontsize=fntsz-2)
print("constrained J =", par2[0], "+/-", np.sqrt(pcov2[0][0]))

print("Graph3")
ax[2][1].errorbar(times3, LJ_mean3, color='tab:purple',  marker='None', linestyle='None', yerr=LJ_err3, capsize=2)
ax[2][1].fill_between(times3, LJ_mean3-LJ_err3, LJ_mean3+LJ_err3, color='tab:purple', alpha=0.2)
par3, pcov3 = curve_fit(fit_to_J, times3[1:], LJ_mean3[1:], sigma=LJ_err3[1:], absolute_sigma=True, bounds=([0],[0.05]))
P = fit_to_J(np.linspace(0, 250000, 10001), *par3)
ax[2][1].plot(np.linspace(0, 250000, 10001), P, color='tab:grey', label=r"Fit, $\alpha$ constrained to 1")
fit_err = error_Jfit(np.linspace(0, 250000, 10001), par3[0], np.sqrt(pcov3[0][0]))
ax[2][1].fill_between(np.linspace(0, 250000, 10001), P-fit_err, P+fit_err, color='tab:grey', alpha=0.2)
ax[2][1].tick_params(axis='both', which='major', labelsize=(fntsz-2))
ax[2][1].set_xlabel(r'$t\mbox{*}$', fontsize=fntsz)
ax[2][1].set_ylabel(r'$P_{liq}(t\mbox{*})$', fontsize=fntsz)
print("constrained J =", par3[0], "+/-", np.sqrt(pcov3[0][0]))




#plt.show()
plt.savefig("Surviv_25Mar.png", dpi=340, format='png')
