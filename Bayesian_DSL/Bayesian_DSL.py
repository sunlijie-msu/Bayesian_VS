import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from random import sample
import scipy.stats as sps
from scipy.stats import poisson
from scipy.ndimage import gaussian_filter1d
from surmise.calibration import calibrator
from surmise.emulation import emulator
import os
import subprocess
from smt.sampling_methods import LHS
import pandas as pd
import matplotlib as mpl
import matplotlib.patches as mpatches
from packaging import version
import seaborn as sns
import pylab
import sys

# sys.stdout = open('output.txt', 'w')

# Read data from CSV files
print("\n[Step 1: Read and check input files.]")
project_path = os.path.abspath("Bayesian_DSL.py")
dir_path = os.path.dirname(project_path)

peak = '23Mg7333' # modify
dataset_Tau = '_S2193' # '_0fs' or '_3fs' or '_5fs'

# Load data fitrange csv files and model fitrange csv files and model parameter values csv files
data_fitrange = np.loadtxt(dir_path + '\DSL_' + peak + dataset_Tau + '\DSL_' + peak + '_data.csv', delimiter=',')  # Read full range data # csv derived from histogram_get_bin_content_error.C
bin_start = 12 # row 12 is 7732.5 # modify
bin_stop = 36 # row 35 is 7847.5 # modify
data_peakrange = data_fitrange[bin_start:bin_stop, :] # Select data in the peak range by rows. bin_start is included and bin_stop is excluded, therefore, +1.

data_x_values_fitrange = data_fitrange[:, 0]
data_y_values_fitrange = data_fitrange[:, 1]
data_y_varlow_fitrange = data_fitrange[:, 2]
data_y_varhigh_fitrange = data_fitrange[:, 3]

data_x_values_peakrange = data_peakrange[:, 0]
data_y_values_peakrange = data_peakrange[:, 1]
data_y_varlow_peakrange = data_peakrange[:, 2]
data_y_varhigh_peakrange = data_peakrange[:, 3]

model_parameter_values = np.loadtxt(dir_path + '\DSL_' + peak + dataset_Tau + '\DSL_' + peak + '_model_parameter_values.csv', delimiter=',')  # csv derived from Comparison_DSL2.C
model_y_values_fitrange = np.loadtxt(dir_path + '\DSL_' + peak + dataset_Tau + '\DSL_' + peak + '_model_y_values.csv', delimiter=',')   # csv derived from Comparison_DSL2.C
model_y_var_fitrange = np.loadtxt(dir_path + '\DSL_' + peak + dataset_Tau + '\DSL_' + peak + '_model_y_values_var.csv', delimiter=',')   # csv derived from Comparison_DSL2.C
# model_y_var_fitrange = np.loadtxt(dir_path + '\DSL_' + peak + dataset_Tau + '\DSL_' + peak + '_model_y_values_var_All0.csv', delimiter=',')   # csv derived from Comparison_DSL2.C

model_y_values_peakrange = model_y_values_fitrange[:,bin_start:bin_stop] # Select model in the peak range by columns
model_y_var_peakrange = model_y_var_fitrange[:,bin_start:bin_stop] # Select model in the peak range by columns


print("Path: ", dir_path + '\DSL_' + peak + dataset_Tau + '\DSL_' + peak + '_data.csv')
print("\nDimensions of data_fitrange:")
print("Shape:", data_fitrange.shape)
print("Rows:", len(data_fitrange))
print("Columns:", len(data_fitrange[0]))

print("\nDimensions of data_peakrange:")
print("Shape:", data_peakrange.shape)
print("Rows:", len(data_peakrange))
print("Columns:", len(data_peakrange[0]))

print("\nDimensions of data_x_values_fitrange:")
print("Shape:", data_x_values_fitrange.shape)

print("\nDimensions of data_y_values_fitrange:")
print("Shape:", data_y_values_fitrange.shape)

print("\nDimensions of data_y_values_var_low_fitrange:")
print("Shape:", data_y_varlow_fitrange.shape)

print("\nDimensions of data_y_values_var_high_fitrange:")
print("Shape:", data_y_varhigh_fitrange.shape)

print("\nDimensions of data_x_values_peakrange:")
print("Shape:", data_x_values_peakrange.shape)

print("\nDimensions of data_y_values_peakrange:")
print("Shape:", data_y_values_peakrange.shape)

print("\nDimensions of data_y_values_var_low_peakrange:")
print("Shape:", data_y_varlow_peakrange.shape)

print("\nDimensions of data_y_values_var_high_peakrange:")
print("Shape:", data_y_varhigh_peakrange.shape)

print("\nDimensions of model_parameter_values:")
print("Shape:", model_parameter_values.shape)
print("Rows:", len(model_parameter_values))
print("Columns:", len(model_parameter_values[0]))

print("\nDimensions of model_y_values_fitrange:")
print("Shape:", model_y_values_fitrange.shape)
print("Rows:", len(model_y_values_fitrange))
print("Columns:", len(model_y_values_fitrange[0]))

print("\nDimensions of model_y_values_peakrange:")
print("Shape:", model_y_values_peakrange.shape)
print("Rows:", len(model_y_values_peakrange))
print("Columns:", len(model_y_values_peakrange[0]))

bin_size = data_x_values_fitrange[1] - data_x_values_fitrange[0]  # bin_size = 5 keV
num_bins_peak = len(data_x_values_peakrange)  # the number of unique values in the first column of data_x_values, which corresponds to the number of different x values in the data = 59
num_bins_fitrange = len(data_x_values_fitrange)
peakrange_min = data_x_values_peakrange[0]
peakrange_max = data_x_values_peakrange[num_bins_peak-1]
fitrange_min = data_x_values_fitrange[0]
fitrange_max = data_x_values_fitrange[num_bins_fitrange-1]
num_model_runs = len(model_parameter_values)  # len(2Darray) will give you the number of rows in the 2D array.

print("\npeakrange_min: ", peakrange_min,", peakrange_max: ", peakrange_max)
print("fitrange_min: ", fitrange_min, ", fitrange_max: ", fitrange_max)
print("\nnum_bins_peakrange: ", num_bins_peak, ",  num_bins_fitrange: ", num_bins_fitrange)

num_bins_peak = len(model_y_values_peakrange[0])  # len(2Darray[0]) will give you the number of elements in the first row of the 2D array, i.e., the number of columns in the 2D array.
num_bins_fitrange = len(model_y_values_fitrange[0])
print("num_bins_peakrange: ", num_bins_peak, ",  num_bins_fitrange: ", num_bins_fitrange)

# Select some model runs as training runs. You may select all the runs.
# rndsample = sample(range(0, 324), 324)  # generates a list of 324 random numbers from the range 0 to 324 and assigns it to the variable rndsample.
if peak== '23Mg7333':
    number_of_fullmodel_runs = 201
    ymax = round(max(data_y_values_fitrange) + max(data_y_varhigh_fitrange) * 2.5)
    emulator_diagnostics = 'False' # 'True' or 'False'. If 'True', the emulator will be trained on a subset of the full model runs for evaluating emulator performance. If 'False', the emulator will be trained on all the full model runs for MCMC parameter estimation.
if peak == '31S1248':
    number_of_fullmodel_runs = 324
    ymax = 90
if peak == '31S3076':
    number_of_fullmodel_runs = 2106
    ymax = 12
if peak == '31S4156':
    number_of_fullmodel_runs = 297
    ymax = 30


if emulator_diagnostics == 'True':
    training_size = int(0.75 * number_of_fullmodel_runs) # Calculate the number of training samples (80% of the total)
    rndsample_train = sample(range(number_of_fullmodel_runs), training_size) # Randomly select indices for the training set
    rndsample_test = [x for x in range(number_of_fullmodel_runs) if x not in rndsample_train]

if emulator_diagnostics == 'False':
    rndsample_train = list(range(0, number_of_fullmodel_runs, 1)) # 1 means select all runs; 2 means select every other run, 0, 2, 4, 6, 8, etc.
    rndsample_test = list(range(1, number_of_fullmodel_runs, 2)) # 1 means select all runs; 2 means select every other run, 1, 3, 5, 7, 9, etc.

print('\nSelect training runs: ', rndsample_train)
model_y_values_peakrange_train = model_y_values_peakrange[rndsample_train, :]  # model_y_values_train is a subset of model_y_values where the rows are selected using the rndsample list and all columns are included by specifying : for the second index.
model_y_values_fitrange_train = model_y_values_fitrange[rndsample_train, :]  # model_y_values_train is a subset of model_y_values where the rows are selected using the rndsample list and all columns are included by specifying : for the second index. # for visualization purpose only
model_y_var_peakrange_train = model_y_var_peakrange[rndsample_train, :]
model_y_var_fitrange_train = model_y_var_fitrange[rndsample_train, :]
model_parameter_values_train = model_parameter_values[rndsample_train, :]


print('\nSelect test runs: ', rndsample_test)
model_y_values_peakrange_test = model_y_values_peakrange[rndsample_test, :]  # model_y_values_train is a subset of model_y_values where the rows are selected using the rndsample list and all columns are included by specifying : for the second index.
model_y_var_peakrange_test = model_y_var_peakrange[rndsample_test, :]
model_parameter_values_test = model_parameter_values[rndsample_test, :]

# plots a list of profiles in the same figure. Each profile corresponds to a simulation replica for the given instance.
plt.rcParams['axes.linewidth'] = 3.0
plt.rcParams['font.size'] = 60
font_family_options = ['Times New Roman', 'Georgia', 'Cambria', 'Courier New', 'serif']
plt.rcParams['font.family'] = font_family_options
# plt.rcParams['mathtext.default'] = 'regular'
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Times New Roman'
plt.rcParams['mathtext.it'] = 'Times New Roman:italic'

print("\n[Step 2: Plot model training runs (prior) vs data.]")
fig, ax_prior = plt.subplots(figsize=(36, 12))
fig.subplots_adjust(left=0.08, bottom=0.18, right=0.98, top=0.96)

p1 = ax_prior.errorbar(data_x_values_fitrange, data_y_values_fitrange, yerr=[data_y_varlow_fitrange,data_y_varhigh_fitrange], fmt='s', color='black', linewidth=3, markersize=7, label='Data', ecolor='black', zorder=2)  # zorder 2 appears on top of the zorder = 1.
# Get min and max model counts in each bin
model_lowcount = model_y_values_fitrange_train.min(axis=0)
model_highcount = model_y_values_fitrange_train.max(axis=0)
# Smooth boundaries of the filled area
smoothed_model_lowcount = gaussian_filter1d(model_lowcount, sigma=0.7)
smoothed_model_highcount = gaussian_filter1d(model_highcount, sigma=0.7)
# Plot shaded region between min and max
p2 = ax_prior.fill_between(data_x_values_fitrange, smoothed_model_lowcount, smoothed_model_highcount,
                     color='red', alpha=0.3, linewidth=0, label='Prior', zorder=1)
ax_prior.tick_params(axis='both', which='major', labelsize=60, length=9, width=2)
# ax.tick_params(direction='in')
ax_prior.set_ylabel('Counts per 5 keV', fontsize=60, labelpad=30)
ax_prior.set_xlabel('Energy (keV)', fontsize=60, labelpad=20)
ax_prior.legend([p1, p2], ['Data', 'Prior'], fontsize=60, loc='upper right', bbox_to_anchor=(1.007, 1.025))
xmin = min(data_x_values_fitrange) - bin_size * 0.2
xmax = max(data_x_values_fitrange) + bin_size * 0.2
ax_prior.set_xlim(xmin, xmax)
ax_prior.set_ylim(0, ymax)

# Adjust the width of the frame
for spine in ax_prior.spines.values():
    spine.set_linewidth(2)  # Set the linewidth to make the frame wider
plt.savefig(peak + dataset_Tau + '_prior.png')
# plt.show()


print("\n[Step 3: Principal Component Analysis (PCA).]")

def plot_explained_variance(singular_values): # singular_values is related to the eigenvalues of the covariance matrix.
    # Compute the individual explained variances
    explained_variances = (singular_values ** 2) / np.sum(singular_values ** 2)
    # Compute the cumulative explained variances
    cumulative_variances = np.cumsum(explained_variances)

    # Limit to first 20 components
    # max_components = 20
    # explained_variances = explained_variances[:max_components]
    # cumulative_variances = cumulative_variances[:max_components]
    
    # Create the plot with the specified size
    fig, ax_variance = plt.subplots(figsize=(36, 12))
    fig.subplots_adjust(left=0.09, bottom=0.18, right=0.98, top=0.96)
    
    # Create the bar plot for individual variances
    bar = ax_variance.bar(range(1, len(explained_variances) + 1), explained_variances, alpha=0.5, color='g', label='Individual Explained Variance', width=0.5)

    # Create the line plot for cumulative variance
    line = ax_variance.plot(range(1, len(cumulative_variances) + 1), cumulative_variances, marker='o', linestyle='-', color='r', label='Cumulative Explained Variance', linewidth=3, markersize=10)

    # Adding percentage values on top of bars and dots
    for i, (bar, cum_val) in enumerate(zip(bar, cumulative_variances)):
         ax_variance.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{explained_variances[i]*100:.1f}%', ha='center', va='bottom', fontsize=32)
         ax_variance.text(i+1, cum_val, f'{cum_val*100:.1f}%', ha='center', va='bottom', fontsize=32)

    # Aesthetics for the plot
    ax_variance.set_xlabel('Principal Components', fontsize=60, labelpad=20)
    ax_variance.set_ylabel('Explained Variance', fontsize=60, labelpad=20)
    # ax_variance.set_title('Explained Variance by Different Principal Components')
    ax_variance.set_xticks(range(1, len(explained_variances) + 1))
    # ax_variance.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    ax_variance.legend(loc='lower left', bbox_to_anchor=(0.55, 0.25)) # (x, y)
    ax_variance.tick_params(axis='both', which='major', labelsize=60, length=9, width=2)
    ax_variance.set_xlim(0.3, 24.7)
    ax_variance.set_ylim(3e-4, 1.9)
    ax_variance.set_yscale('log')
    ax_variance.grid(True)
    # plt.show()
    plt.savefig(peak + dataset_Tau + '_explained_variance.png')
    

# (No filter) Fit an emulator via 'PCGP'
print("\n[Step 4: Model emulation.]")
# C:\Users\sun\AppData\Local\Programs\Python\Python311\Lib\site-packages\surmise\emulationmethods\PCGP_numPCs.py # modify
# PCGP_numPCs.py is a modified version of PCGP.py that allows the user to specify the number of principal components to be used in the emulator. The number of principal components is specified using the args dictionary with the key 'num_pcs'. If num_pcs is not specified, it falls back to epsilon = 0.1.

emulator_1 = emulator(x=data_x_values_peakrange,
                      theta=model_parameter_values_train,
                      f=model_y_values_peakrange_train.T,
                      method='PCGP_numPCs',
                      args={'num_pcs': 24})  # Specify the number of principal components

# emulator_1 = emulator(x=data_x_values_peakrange,
#                       theta=model_parameter_values_train,
#                       f=model_y_values_peakrange_train.T,
#                       method='PCGP',
#                       args={'epsilon': 1e-50}) # Typically, we want to keep the principal components that capture the most variance. The epsilon parameter is used to set a threshold for filtering PCs with explained variances > epsilon.

# prior_min = [0, 7330, 0.4, 0.4]
# prior_max = [30, 7338, 2.0, 2.0]
# prior_dict = {'min': prior_min, 'max': prior_max}
# emulator_1 = emulator(x=data_x_values_peakrange,
#                       theta=model_parameter_values_train,
#                       f=model_y_values_peakrange_train.T,
#                       method='PCGPR',
#                       args={'epsilon': 1e-10, 'prior': prior_dict})

# emulator_1 = emulator(x=data_x_values_peakrange,
#                       theta=model_parameter_values_train,
#                       f=model_y_values_peakrange_train.T,
#                       method='indGP')

# emulator_1 = emulator(x=data_x_values_peakrange,
#                       theta=model_parameter_values_train,
#                       f=model_y_values_peakrange_train.T,
#                       method='PCSK',
#                       # args={'epsilonPC': 0.0000000001, 'simsd': model_y_var_peakrange_train.T, 'verbose': 1})
#                       args={'warnings': True, 'numpcs': 24, 'simsd': model_y_var_peakrange_train.T, 'verbose': 1})

# f can be from an analytic function too
# model_y_values, m runs/rows, n bins/columns, need to be transposed in this case cuz each column in f should correspond to a row in x.
# /usr/local/lib/python3.8/dist-packages/surmise/emulationmethods/PCGP.py
# C:\Users\sun\AppData\Local\Programs\Python\Python311\Lib\site-packages\surmise\emulationmethods

# {Model emulation methods}

# 1) `indGP': skips the PCA dimensionality reduction step and builds independent emulators directly on the original outputs, hence, no epsilon is needed. However, the subsequent MCMC process incurs significant computational costs (CPU and RAM) associated with large covariance matrices. MCMC results look similar to PCGP.

# 2) `PCGP': Principal Component Gaussian Process emulator uses PCA to reduce the dimensionality of the simulation output before fitting a Gaussian Process model to each Principal Component separately. Default emulation method; recommended for DSL analysis.

# 3) `PCGPR': Principal Component Gaussian Process Regression utilizes scikit-learn's GaussianProcessRegressor for GP modeling.

# 4) `PCGPRG': Principal Component Gaussian Process Regression with Grouped observables. PCGPR emulation are performed for each subgroup separately.

# Ref.~\cite{Liyanage_PRC2023} found that PCSK>PCGPR>PCGPR in terms of $R^2$ scores.

# 5) `PCGPwM': Principal Component Gaussian Process with Missingness properly handle missing points in the simulation output due to simulation failures through imputation and incorporating additional variance estimates at each point requiring imputation, before performing PCA and fitting Gaussian Process models.

# 6) `PCGPwImpute': Principal Component Gaussian Process with Imputation: extends PCGPwM by first externally imputing missing values using iterative imputation techniques (such as K-Nearest Neighbors, Bayesian Ridge Regression, or Random Forest) from scikit-learn, and then applies the PCGPwM method on the completed dataset.

# 7) `PCSK': Principal Component Stochastic Kriging.

# Both PCGP and PCSK are Gaussian Process-based emulators. Both use PCA to reduce high-dimensional outputs into manageable dimensions for modeling. PCGP is developed to handle deterministic simulation models, where the simulator returns the same output every time given the same input parameters. PCGP considers mean but ignores any variance of simulation data. Our \textsc{geant}4 simulation model is stochastic, meaning that for the same input parameters, the simulator can return different outputs when being run multiple times. Instead of only taking into account the mean values of the emulator training data, the PCSK method also considers the standard deviation at the training points (`simsd'). The PCSK emulation incorporates this variation in the accuracy between the resulting data points, known as heteroscedasticity, by using stochastic kriging. 

# PCSK considers mean but ignores variance of simulation data during hyperparameter optimization, but considers variance in posterior predictive distribution.

# PCGP and PCSK with or without the uncertainties of the training data give consistent posterior results and both PCGP and PCSK outperform standard Gaussian Process implementations like Scikit GP in predictive accuracy and uncertainty estimation~\cite{Roch_PRC2024}.

# 8) `LCGP': Latent Component Gaussian Process handles stochastic multivariate outputs with heterogeneous errors~\cite{Chan_Thesis2023}. Unavailable in \textsc{surmise} version 0.3.0.



# after fitting the emulator
# print(emulator_1._info['param_desc'])
print("\nEmulation method: ", emulator_1.method)
print("\nfitinfo keys:", emulator_1._info.keys())

method_str = str(emulator_1.method)
if 'PCGP_numPCs' in method_str:
    singular_values = emulator_1._info['singular_values']
    plot_explained_variance(singular_values)



print("\n[Step 5: Diagnostics plots.]")
pred_emu = emulator_1.predict(data_x_values_peakrange, model_parameter_values_test)  # predict at some parameter values and x values
pred_m, pred_var = pred_emu.mean(), pred_emu.var()  # Returns the mean and variance at theta and x in the prediction. pred_var represents the diagonal elements of the emulator covariance matrix, namely, the predictive variances.
print("\npred_m:", pred_m.shape, "pred_var:", pred_var.shape)  # pred_m: (59, 162) pred_var: (59, 162)
pred_emu_tr = emulator_1.predict(data_x_values_peakrange, model_parameter_values_train)
pred_m_tr, pred_var_tr = pred_emu_tr.mean(), pred_emu_tr.var()
print("pred_m_tr:", pred_m_tr.shape, "pred_var_tr:", pred_var_tr.shape)  # pred_m_tr: (59, 324) pred_var_tr: (59, 324)

n = pred_m.shape[1]  # returns the number of columns in pred_m = 162
std_err_f_test = ((pred_m - model_y_values_peakrange_test.T)/np.sqrt(pred_var)).flatten()
std_err_nf_test = np.mean((pred_m - model_y_values_peakrange_test.T)/np.sqrt(pred_var), axis=0)
fig, axs_emu1 = plt.subplots(1, 2, figsize=(36, 12))
fig.subplots_adjust(left=0.07, bottom=0.15, right=0.97, top=0.95)
xs = np.repeat(data_x_values_peakrange, n)
axs_emu1[0].scatter(xs, std_err_f_test, alpha=0.5, s=70)
axs_emu1[0].plot(xs, np.repeat(0, len(xs)), color='red')
axs_emu1[0].set_xlabel(r'Energy (keV)')
axs_emu1[0].set_ylabel(r'${(\hat{\mu}_{test} - \mu_{test})}/{\hat{\sigma}_{test}}$')
axs_emu1[0].set_ylim(-5, 5)
axs_emu1[1].scatter(np.arange(0, n), std_err_nf_test, s=200)
axs_emu1[1].plot(np.arange(0, n), np.repeat(0, n), color='red')
axs_emu1[1].set_xlabel(r'Test run')
axs_emu1[1].set_ylim(-0.6, 0.6)
plt.savefig(peak + dataset_Tau + '_residual.png')
#  plt.show()

# Calculate R^2 for test set
errors_test = (pred_m - model_y_values_peakrange_test.T).flatten()
sst_test = np.sum((model_y_values_peakrange_test.T.flatten() - np.mean(model_y_values_peakrange_test.T.flatten()))**2)
sse_test = np.sum(errors_test**2)
rsq_test = 1 - (sse_test / sst_test)
rsq_test_rounded = np.round(rsq_test, 3)

print(f"\nRsq (test) = {rsq_test_rounded}")

# Calculate R^2 for training set
errors_train = (pred_m_tr - model_y_values_peakrange_train.T).flatten()
sst_train = np.sum((model_y_values_peakrange_train.T.flatten() - np.mean(model_y_values_peakrange_train.T.flatten()))**2)
sse_train = np.sum(errors_train**2)
rsq_train = 1 - (sse_train / sst_train)
rsq_train_rounded = np.round(rsq_train, 3)

print(f"Rsq (train) = {rsq_train_rounded}")


# Visualization with the correct R^2
fig, axs_emu2 = plt.subplots(1, 2, figsize=(36, 12))
fig.subplots_adjust(left=0.07, bottom=0.15, right=0.97, top=0.89)

# Scatter plot for test set
axs_emu2[0].scatter(model_y_values_peakrange_test.T, pred_m, alpha=0.3)
axs_emu2[0].plot(range(2, ymax-15), range(2, ymax-15), color='red')
axs_emu2[0].set_xlabel('Simulator bin counts (test)')
axs_emu2[0].set_ylabel('Emulator bin counts (test)')
axs_emu2[0].set_title(r'$R^2=$' + str(rsq_test_rounded))

# Scatter plot for training set
axs_emu2[1].scatter(model_y_values_peakrange_train.T, pred_m_tr, alpha=0.3)
axs_emu2[1].plot(range(2, ymax-15), range(2, ymax-15), color='red')
axs_emu2[1].set_xlabel('Simulator bin counts (training)')
axs_emu2[1].set_ylabel('Emulator bin counts (training)')
axs_emu2[1].set_title(r'$R^2=$' + str(rsq_train_rounded))

plt.savefig(peak + dataset_Tau + '_R2.png')
# plt.show()

mu = 0
variance = 1
sigma = np.sqrt(variance)
x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
fig, axs_emu3 = plt.subplots(1, 3, figsize=(36, 12))
fig.subplots_adjust(left=0.07, bottom=0.13, right=0.97, top=0.88)
axs_emu3[0].hist((pred_m - model_y_values_peakrange_test.T).flatten(), bins=100)
axs_emu3[0].set_title(r'$\hat{\mu}_{test} - \mu_{test}$', pad=20)
axs_emu3[0].set_xlim([-1.5, 1.5])
axs_emu3[1].hist(std_err_f_test, density=True, bins=100)  # standardized error
axs_emu3[1].plot(x, sps.norm.pdf(x, mu, sigma), color='red')
axs_emu3[1].set_title(r'${(\hat{\mu}_{test} - \mu_{test})}/{\hat{\sigma}_{test}}$', pad=20)
axs_emu3[1].set_xlim([-4, 4])
axs_emu3[2].hist(((pred_m - model_y_values_peakrange_test.T)/model_y_values_peakrange_test.T).flatten(), bins=100)  # relative error
axs_emu3[2].set_title(r'${(\hat{\mu}_{test} - \mu_{test})}/{\mu_{test}}$', pad=20)
l = np.arange(-2, 3, 1)/10
axs_emu3[2].set(xticks=l, xticklabels=l)
axs_emu3[2].axvline(x=0, ls='--', color='red')
axs_emu3[2].set_xlim([-0.15, 0.15])
plt.savefig(peak + dataset_Tau + '_error.png')
# plt.show()


print('\nMSE =', np.round(np.mean(np.sum(np.square(pred_m - model_y_values_peakrange_test.T), axis=1)), 2))  # calculates the mean squared error (MSE), which is another measure of the accuracy of the emulator.
print('SSE =', np.round(np.sum((pred_m - model_y_values_peakrange_test.T)**2), 2))


# Define a class for prior of 4 parameters
print("\n[Step 6: Prior class specification.]")

# Mixture weights (should sum to 1)
w1, w2, w3 = 0.333333333333, 0.333333333333, 0.333333333333

# Parameters for the two normal distributions
mu, sigma = 7333.2, 1.1 # Basunia_NDS2021 ENSDF
mu1, sigma1 = 7332.7, 1.2 # Sallaska_PRC2011
mu2, sigma2 = 7333.7, 1.1 # Jenkins_PRL2004
mu3, sigma3 = 7335.09, 0.92  # Goldberg_Thesis2024

class Prior_DSL23Mg_7333_ENSDF:
    """ This defines the class instance of priors provided to the method. """
    def lpdf(theta):  # log-probability density function of the prior for a given set of parameters theta ['Tau', 'Eg', 'Bkg', 'SP']
        return (sps.uniform.logpdf(theta[:, 0], 0, 30) +
                    sps.norm.logpdf(theta[:, 1], mu, sigma) +
                    sps.norm.logpdf(theta[:, 2], 1.0, 0.1) +
                    sps.norm.logpdf(theta[:, 3], 1.0, 0.1)).reshape((len(theta), 1))
    

    def rnd(n):  # Generates n random samples from a prior distribution.
        return np.vstack((sps.uniform.rvs(0, 30, size=n),
                          sps.norm.rvs(mu, sigma, size=n),
                          sps.norm.rvs(1.0, 0.1, size=n),
                          sps.norm.rvs(1.0, 0.1, size=n))).T


class Prior_DSL23Mg_7333:

    def lpdf(theta):
        # Compute individual PDFs
        pdf1 = sps.norm.pdf(theta[:, 1], mu1, sigma1)
        pdf2 = sps.norm.pdf(theta[:, 1], mu2, sigma2)
        pdf3 = sps.norm.pdf(theta[:, 1], mu3, sigma3)

        # Mixture PDF
        pdf_mixture = w1 * pdf1 + w2 * pdf2 + w3 * pdf3
        pdf_mixture = np.maximum(pdf_mixture, 1e-300)  # Prevent log(0)

        # Compute log-PDF
        logpdf_theta1 = np.log(pdf_mixture)

        # Log-PDFs for other parameters
        logpdf_theta0 = sps.uniform.logpdf(theta[:, 0], 0, 30)
        logpdf_theta2 = sps.norm.logpdf(theta[:, 2], 1.0, 0.1)
        logpdf_theta3 = sps.norm.logpdf(theta[:, 3], 1.0, 0.1)

        # Total log-PDF
        return (logpdf_theta0 + logpdf_theta1 + logpdf_theta2 + logpdf_theta3).reshape((len(theta), 1))

     
    def rnd(n):
        # Randomly choose which distribution to sample from
        components = np.array([0, 1, 2])
        mus = np.array([mu1, mu2, mu3])
        sigmas = np.array([sigma1, sigma2, sigma3])
        choices = np.random.choice(components, size=n, p=[w1, w2, w3])

        # Generate samples for theta[:, 1]
        samples_theta1 = sps.norm.rvs(loc=mus[choices], scale=sigmas[choices])

        # Generate samples for other parameters
        samples_theta0 = sps.uniform.rvs(0, 30, size=n)
        samples_theta2 = sps.norm.rvs(1.0, 0.1, size=n)
        samples_theta3 = sps.norm.rvs(1.0, 0.1, size=n)

        # Combine all samples
        return np.column_stack((samples_theta0, samples_theta1, samples_theta2, samples_theta3))



class Prior_DSL31S_1248:
    """ This defines the class instance of priors provided to the method. """
    def lpdf(theta):  # log-probability density function of the prior for a given set of parameters theta
        return (sps.uniform.logpdf(theta[:, 0], 200, 1800) +
                sps.norm.logpdf(theta[:, 1], 1248.4, 0.2) +
                sps.norm.logpdf(theta[:, 2], 1.0, 0.1) +
                sps.norm.logpdf(theta[:, 3], 1.0, 0.2)).reshape((len(theta), 1))

    def rnd(n):  # Generates n random variable from a prior distribution.
        return np.vstack((sps.uniform.rvs(200, 1800, size=n),
                          sps.norm.rvs(1248.4, 0.2, size=n),
                          sps.norm.rvs(1.0, 0.1, size=n),
                          sps.norm.rvs(1.0, 0.2, size=n))).T


class Prior_DSL31S_3076:
    """ This defines the class instance of priors provided to the method. """
    def lpdf(theta):  # log-probability density function of the prior for a given set of parameters theta
        return (sps.uniform.logpdf(theta[:, 0], 0, 30) +
                sps.norm.logpdf(theta[:, 1], 3076.24, 0.20) +
                sps.norm.logpdf(theta[:, 2], 1.0, 0.1) +
                sps.norm.logpdf(theta[:, 3], 1.0, 0.1) +
                sps.uniform.logpdf(theta[:, 4], -1.0, 2.0)).reshape((len(theta), 1))

    def rnd(n):  # Generates n random variable from a prior distribution.
        return np.vstack((sps.uniform.rvs(0, 30, size=n),
                          sps.norm.rvs(3076.24, 0.20, size=n),
                          sps.norm.rvs(1.0, 0.1, size=n),
                          sps.norm.rvs(1.0, 0.1, size=n),
                          sps.uniform.rvs(-1.0, 2.0, size=n))).T

class Prior_DSL31S_4156:
    """ This defines the class instance of priors provided to the method. """
    def lpdf(theta):  # log-probability density function of the prior for a given set of parameters theta ['Tau', 'Eg', 'Bkg', 'SP']
        return (sps.uniform.logpdf(theta[:, 0], 0, 30) +
                sps.norm.logpdf(theta[:, 1], 4155.84, 0.31) +
                sps.norm.logpdf(theta[:, 2], 1.0, 0.2) +
                sps.norm.logpdf(theta[:, 3], 1.0, 0.2)).reshape((len(theta), 1))
    
    def rnd(n):  # Generates n random variables (rvs) from a prior distribution.
        return np.vstack((sps.uniform.rvs(0, 30, size=n),
                          sps.norm.rvs(4155.84, 0.31, size=n),
                          sps.norm.rvs(1.0, 0.2, size=n),
                          sps.norm.rvs(1.0, 0.2, size=n))).T


# obsvar = np.maximum(0.2 * data_y_values, 1.4)
obsvar = (data_y_varlow_peakrange + data_y_varhigh_peakrange)/2
# print(obsvar)

# Calibrator 1
print("\n[Step 7: MCMC sampling.]")
total_mcmc_samples = 1000000
plot_mcmc_samples = 100000  # for slow 2D corner plots
if peak == '23Mg7333':
    calibrator_1 = calibrator(emu=emulator_1,
                                           y=data_y_values_peakrange,
                                           x=data_x_values_peakrange,
                                           thetaprior=Prior_DSL23Mg_7333,
                                           # method='directbayes',
                                           method='directbayeswoodbury',
                                           # method='mlbayeswoodbury',
                                           yvar=obsvar,
                                           args={'theta0': np.array([[7.0, 7333.2, 1.0, 1.0]]),  # initial guess ['Tau', 'Eg', 'Bkg', 'SP']
                                                     'sampler': 'metropolis_hastings',
                                                     # 'sampler': 'LMC',
                                                     # 'sampler': 'PTMC',
                                                     # 'sampler': 'PTLMC',
                                                     'numsamp': total_mcmc_samples,
                                                     'numchain': 10,
                                                     'stepType': 'normal',
                                                     'burnSamples': 1000,
                                                     'nburnin': 1000,
                                                     'verbose': True
                                                     # 'stepParam': np.array([0.0, 0.0, 0.0, 0.0]) # ['Tau', 'Eg', 'Bkg', 'SP'] somehow stepParams don't work well
                                                     }
                                            )

# {Model calibration methods}
# 1) `directbayes': default calibration method. It builds a log-posterior by combining a prior (fitinfo[`'thetaprior']) with a likelihood (loglik) comparing emulator�\predicted means/covariances to observed data. `directbayes' uses standard matrix operations. The final posterior samples are stored in fitinfo['thetarnd'].

# 2) `directbayeswoodbury': leverages the Woodbury identity to handle Gaussian covariance manipulations, which can be more efficient or numerically stable for large datasets. The final posterior samples are stored in fitinfo['thetarnd']. `directbayeswoodbury' is recommended for DSL analysis.

# 3) `mlbayeswoodbury': clf\_method must be a trained classifier (e.g., a scikit-learn model) that implements .predict\_proba(...) on each sampled theta to get acceptance probabilities, which it folds into the posterior. If this ML-based weighting is not needed, either pass clf\_method=None or use a different calibration method (directbayes or directbayeswoodbury).

# {Utilities methods (samplers)}

# 1) `metropolis\_hastings': default sampler that draws from a target posterior distribution; recommended for DSL analysis.

# 2) `LMC': Langevin Monte Carlo uses gradient-based proposals to guide samples toward high-posterior regions, often improving acceptance over plain metropoli\_hastings. Users do not need to set stepParam explicitly: by default, LMC will estimate an initial scale from the starting samples and adapt from there. Users do not need to set burn-in explicitly. Instead, LMC uses an iterative procedure (e.g., maxiters=10, numsamppc=200) and tries to adapt acceptance rates. It then returns a single final chain (theta) with size numsamp.

# 3) `PTLMC': Parallel Tempering Langevin Monte Carlo combines Parallel Tempering (running multiple chains at different temperatures and exchanging states periodically) and Langevin Monte Carlo (using gradient-based proposals). PTLMC has the advantages of faster convergence, especially for complex or multimodal distributions, and reduced risk of trapping in local minima. In the analysis of the S2193 7333-keV $\gamma$-ray data, the posterior distribution exhibits numerous nearby local minima compared to that obtained using the Metropolis-Hastings sampler, indicating it is less suitable for DSL lineshape analysis. `PTLMC' does not perform better than the default `metropolis\_hastings' when it comes to handling multimodal distributions. The overall credible intervals by integrating the posterior distributions are consistent with those from `metropolis\_hastings' sampling. The acceptance rate for PTLMC is approximately 0.004, while Metropolis-Hastings has an acceptance rate of about 0.22.

# 4) `PTMC': required positional arguments: `log\_likelihood' and `log\_prior'. PTMC is not supported in the version 0.3.0 of \textsc{surmise}.


if peak == '31S1248':
    calibrator_1 = calibrator(emu=emulator_1,
                                           y=data_y_values_peakrange,
                                           x=data_x_values_peakrange,
                                           thetaprior=Prior_DSL31S_1248,
                                           # method='directbayes',
                                           method='directbayeswoodbury',
                                           # method='mlbayeswoodbury',
                                           yvar=obsvar,
                                           args={'theta0': np.array([[1100, 1248.4, 1.0, 1.0]]),  # initial guess
                                                     'sampler': 'metropolis_hastings',
                                                     # 'sampler': 'LMC',
                                                     # 'sampler': 'PTLMC',
                                                     'numsamp': total_mcmc_samples,
                                                     'numchain': 8,
                                                     'stepType': 'normal',
                                                     # 'burnSamples': 1000
                                                     # 'stepParam': np.array([1, 0.01, 0.01, 0.01])
                                                     }
                                            )

if peak == '31S3076':
    calibrator_1 = calibrator(emu=emulator_1,
                                           y=data_y_values_peakrange,
                                           x=data_x_values_peakrange,
                                           thetaprior=Prior_DSL31S_3076,
                                           method='directbayeswoodbury',
                                           yvar=obsvar,
                                           args={'theta0': np.array([[1.0, 3076.4, 1.0, 1.0, 0.0]]),  # initial guess
                                                     'sampler': 'metropolis_hastings',
                                                     'numsamp': total_mcmc_samples,
                                                     'numchain': 8,
                                                     'stepType': 'normal'
                                                     # 'burnSamples': 1000,
                                                     # 'stepParam': np.array([0.1, 0.02, 0.01, 0.01, 0.1])
                                                     }
                                            )
if peak == '31S4156':
    calibrator_1 = calibrator(emu=emulator_1,
                                           y=data_y_values_peakrange,
                                           x=data_x_values_peakrange,
                                           thetaprior=Prior_DSL31S_4156,
                                           # method='directbayes',
                                           # method='directbayeswoodbury',
                                           method='mlbayeswoodbury',
                                           yvar=obsvar,
                                           args={'theta0': np.array([[0.0, 4155.84, 1.0, 1.0]]),  # initial guess ['Tau', 'Eg', 'Bkg', 'SP']
                                                     'sampler': 'metropolis_hastings',
                                                     # 'sampler': 'LMC',
                                                     # 'sampler': 'PTMC', # sampler() missing 2 required positional arguments: 'log_likelihood' and 'log_prior'
                                                     # 'sampler': 'PTLMC',
                                                     'numsamp': total_mcmc_samples,
                                                     'numchain': 10,
                                                     'stepType': 'normal',
                                                     'burnSamples': 1000,
                                                     'nburnin': 1000,
                                                     'verbose': True
                                                     # 'stepParam': np.array([0.0, 0.0, 0.0, 0.0]) # ['Tau', 'Eg', 'Bkg', 'SP'] somehow stepParams don't work well
                                                     }
                                            )
    # C:\Users\sun\AppData\Local\Programs\Python\Python311\Lib\site-packages\surmise\utilitiesmethods\metropolis_hastings.py
    # for i in range(1, burnSamples + numsamp):
    #     if verbose:
    #         if i % 5000 == 0: # Changed by Lijie Sun
    #             print("At sample {}, acceptance rate is {}.".format(i, n_acc/i))


print("\n[Step 8-1: Plot transparent uncertainty band predictions with calibrated parameters.]")

def plot_pred_interval(calib):
    pred = calib.predict(data_x_values_peakrange)
    rndm_m = pred.rnd(s=total_mcmc_samples)
    fig, ax_post_predict = plt.subplots(figsize=(36, 12))
    fig.subplots_adjust(left=0.08, bottom=0.18, right=0.98, top=0.96)
    
    p1 = ax_post_predict.errorbar(data_x_values_fitrange, data_y_values_fitrange, yerr=[data_y_varlow_fitrange,data_y_varhigh_fitrange], fmt='s', color='black', linewidth=3, markersize=7, label='Data', ecolor='black', zorder=3)  # zorder 2 appears on top of the zorder = 1.

    posterior_y_upper = np.percentile(rndm_m[:, 0: num_bins_peak], 97.7, axis=0)
    posterior_y_lower = np.percentile(rndm_m[:, 0: num_bins_peak], 2.3, axis=0)
    posterior_y_median = np.percentile(rndm_m[:, 0: num_bins_peak], 50, axis=0)
    print("\nposterior_prediction_y_median: ", posterior_y_median)
    
    # p2 = ax_post_predict.fill_between(data_x_values_fitrange, smoothed_model_lowcount, smoothed_model_highcount,
    #                  color='red', alpha=0.3, linewidth=0, zorder=1)
    
    slope_value = (posterior_y_median[ num_bins_peak-1] - posterior_y_median[0]) / (peakrange_max - peakrange_min)
    intercept_value = posterior_y_median[0] - slope_value * peakrange_min
    
    p4 = ax_post_predict.fill_between(data_x_values_peakrange, posterior_y_lower, posterior_y_upper, color='blue', label='95% Credible Interval', alpha=0.3, linewidth=0, zorder=1)
    p3_line, = ax_post_predict.plot(data_x_values_peakrange, posterior_y_median, color='blue', label='Prediction Median', alpha=1.0, linewidth=2, zorder=2)
    ax_post_predict.tick_params(axis='both', which='major', labelsize=60, length=9, width=2)
    # ax_post_predict.tick_params(direction='out')
    ax_post_predict.set_ylabel('Counts per 5 keV', fontsize=60, labelpad=30)
    ax_post_predict.set_xlabel('Energy (keV)', fontsize=60, labelpad=20)
    ax_post_predict.legend([p1, p3_line, p4], ['Data', 'Prediction Median', '95% Credible Interval'], fontsize=60, loc='upper right', bbox_to_anchor=(1.007, 1.025))
    # ax_post_predict.set_xticks(np.arange(xmin-0.5, xmax+1, step=25))
    ax_post_predict.set_xlim(xmin, xmax)
    ax_post_predict.set_ylim(0, ymax)

    
    # Add a linear background out of peak range for visualization only


    linear_x_values = np.linspace(fitrange_min, peakrange_min, 200)
    linear_y_values_middle = slope_value * linear_x_values + intercept_value
    linear_y_values_upper = linear_y_values_middle * (posterior_y_upper[0] / posterior_y_median[0])
    linear_y_values_lower = linear_y_values_middle * (posterior_y_lower[0] / posterior_y_median[0])

    ax_post_predict.fill_between(linear_x_values, linear_y_values_lower, linear_y_values_upper, color='blue', alpha=0.3, linewidth=0, zorder=1)
    ax_post_predict.plot(linear_x_values, linear_y_values_middle, label='Linear Function Left', color='blue', linewidth=2, zorder=2)

    linear_x_values = np.linspace(peakrange_max, fitrange_max, 200)
    linear_y_values_middle = slope_value * linear_x_values + intercept_value
    linear_y_values_upper = linear_y_values_middle * (posterior_y_upper[num_bins_peak-1] / posterior_y_median[num_bins_peak-1])
    linear_y_values_lower = linear_y_values_middle * (posterior_y_lower[num_bins_peak-1] / posterior_y_median[num_bins_peak-1])

    ax_post_predict.fill_between(linear_x_values, linear_y_values_lower, linear_y_values_upper, color='blue', alpha=0.3, linewidth=0, zorder=1)
    ax_post_predict.plot(linear_x_values, linear_y_values_middle, label='Linear Function Right', color='blue', linewidth=2, zorder=2)
        
    # Adjust the width of the frame
    for spine in ax_post_predict.spines.values():
        spine.set_linewidth(2)  # Set the linewidth to make the frame wider

    plt.savefig(peak + dataset_Tau + '_prediction.png')
    # plt.show()

plot_pred_interval(calibrator_1)

print("\n[Step 8-2: Plot posterior samples.]")
def plot_theta(calib, whichtheta):
    fig, axs_trace = plt.subplots(3, 1, figsize=(30, 30))
    fig.subplots_adjust(left=0.11, bottom=0.07, right=0.96, top=0.97, hspace=0.3)
    cal_theta = calib.theta.rnd(total_mcmc_samples)
    axs_trace[0].plot(cal_theta[:, whichtheta])
    axs_trace[0].set_xlabel("Iteration", fontsize=60)
    axs_trace[0].set_ylabel(r"$\tau$ (fs)", fontsize=60)
    axs_trace[0].set_xlim([0, total_mcmc_samples/100]) # plots the first 1% samples?
    axs_trace[0].set_ylim([0, 30])

    axs_trace[1].boxplot(cal_theta[:, whichtheta], orientation='horizontal')
    axs_trace[1].set_xlabel(r"$\tau$ (fs)", fontsize=60)  
    axs_trace[1].set_ylabel(r"$\tau$", fontsize=60)
    axs_trace[1].set_yticklabels([])
    axs_trace[1].set_xlim([0, 30])

    axs_trace[2].hist(cal_theta[:, whichtheta], bins=300, range=[0, 30])
    axs_trace[2].set_xlabel(r"$\tau$ (fs)", fontsize=60)
    axs_trace[2].set_ylabel("Counts per 0.1 fs", fontsize=60, labelpad=20)
    axs_trace[2].set_xlim([0, 30])
    
    axs_trace[0].tick_params(axis='both', which='major', labelsize=60)
    axs_trace[1].tick_params(axis='both', which='major', labelsize=60)
    axs_trace[2].tick_params(axis='both', which='major', labelsize=60)
    
    samples = cal_theta[:, whichtheta]
    percentiles = [16, 50, 84, 90, 95]

    # Calculate percentiles
    values = np.percentile(samples, percentiles)
    percentile_values = dict(zip(percentiles, values))

    sigma_minus = percentile_values[50] - percentile_values[16]
    sigma_plus = percentile_values[84] - percentile_values[50]
    print(
        f"\nTau = {percentile_values[50]:.3f} +{sigma_plus:.3f} -{sigma_minus:.3f} fs")

    # write the samples to a file
    with open(f"{peak}{dataset_Tau}_samples.dat", 'w') as samples_file:
        np.savetxt(samples_file, samples, delimiter='\t')

    # write the percentiles to a file
    with open(f"{peak}{dataset_Tau}_percentiles.txt", 'w') as file:
        for percentile, value in percentile_values.items():
            file.write(f"{percentile}th percentile: {value:.3f}\n")

    
    # Save the plot
    plt.savefig(peak + dataset_Tau + '_trace.png')

    # plt.show()

plot_theta(calibrator_1, 0)


print("\n[Step 8-3: Plot 2D posterior distributions of parameters.]")
if peak == '23Mg7333':
    theta_prior = Prior_DSL23Mg_7333.rnd(plot_mcmc_samples)  # draw a number of random parameters from the prior
    theta_post = calibrator_1.theta.rnd(plot_mcmc_samples)
    dfpost = pd.DataFrame(theta_post, columns=['Lifetime', '$\gamma$-ray Energy', 'Background', 'Stopping Power'])
    dfprior = pd.DataFrame(theta_prior, columns=['Lifetime', '$\gamma$-ray Energy', 'Background', 'Stopping Power'])

if peak == '31S1248':
    theta_prior = Prior_DSL31S_1248.rnd(total_mcmc_samples)  # draw a number of random parameters from the prior
    theta_post = calibrator_1.theta.rnd(total_mcmc_samples)
    dfpost = pd.DataFrame(theta_post, columns=['Tau', 'Eg', 'Bkg', 'SP'])
    dfprior = pd.DataFrame(theta_prior, columns=['Tau', 'Eg', 'Bkg', 'SP'])

if peak == '31S3076':
    theta_prior = Prior_DSL31S_3076.rnd(total_mcmc_samples)  # draw a number of random parameters from the prior
    theta_post = calibrator_1.theta.rnd(total_mcmc_samples)
    dfpost = pd.DataFrame(theta_post, columns=['Tau', 'Eg', 'Bkg', 'SP', 'AC'])
    dfprior = pd.DataFrame(theta_prior, columns=['Tau', 'Eg', 'Bkg', 'SP', 'AC'])

if peak == '31S4156':
    theta_prior = Prior_DSL31S_4156.rnd(total_mcmc_samples)  # draw a number of random parameters from the prior
    theta_post = calibrator_1.theta.rnd(total_mcmc_samples)
    dfpost = pd.DataFrame(theta_post, columns=['Lifetime', '$\gamma$-ray Energy', 'Background', 'Stopping Power'])
    dfprior = pd.DataFrame(theta_prior, columns=['Lifetime', '$\gamma$-ray Energy', 'Background', 'Stopping Power'])

np.set_printoptions(precision=3, suppress=True) # Set the precision to 3 decimal places and suppress scientific notation
print("One prior sample:", theta_prior[0])
print("One posterior sample:", theta_post[0])

# the code below generates a Seaborn PairGrid with kernel density estimation (KDE) plots for a large dataset, which takes a considerable amount of time!
#  a function from the Pandas library that is used to create a dataframe object. Dataframe is a two-dimensional, size-mutable, and tabular data structure with columns of potentially different types.

df = pd.concat([dfprior, dfpost])
#  It is used to combine multiple DataFrames or Series into a single DataFrame or Series. It can combine the dfprior and dfpost DataFrames vertically.
pr = ['Prior' for i in range(plot_mcmc_samples)]
ps = ['Posterior' for i in range(plot_mcmc_samples)]
pr.extend(ps)
df['Distributions'] = pr
#  The pr and ps lists are created to add a new column 'distribution' to the combined DataFrame indicating whether a row belongs to the prior or posterior distribution.

font_family_options = ['Times New Roman', 'Georgia', 'Cambria', 'Courier New', 'serif']
plt.rcParams['font.family'] = font_family_options
# Set Seaborn style
sns.set(style="white")

# Create a PairGrid
g = sns.PairGrid(df, palette=["red", "blue"], corner=True, diag_sharey=False, hue='Distributions')

# Adjust the figure size to make it wider
fig = g.fig
fig.set_size_inches(40, 30)

# Iterate through all axes
for ax in g.axes.flat:
   if ax:
       ax.xaxis.set_ticks_position('bottom')
       ax.yaxis.set_ticks_position('left')
       ax.set_xlabel(ax.get_xlabel(), fontsize=1) 
       ax.set_ylabel(ax.get_ylabel(), fontsize=1)


# Set font for all tick labels   
for ax in g.fig.axes:
    for label in ax.get_xticklabels():
        label.set_fontname("Times New Roman")
        label.set_fontsize(50) 
    for label in ax.get_yticklabels():
        label.set_fontname("Times New Roman")
        label.set_fontsize(50)


# Add yaxes to 1st column
for ax in g.axes[:,0]:
    ax.yaxis.set_visible(True)
    ax.set_ylabel(ax.get_ylabel(), fontsize=50, fontname='Times New Roman', labelpad=25)

# Add xaxes to 4th row
for ax in g.axes[3,:]: 
    ax.xaxis.set_visible(True)
    ax.set_xlabel(ax.get_xlabel(), fontsize=50, fontname='Times New Roman', labelpad=25)


g.axes[0, 0].set(xlim=(0, 30), xticks=np.arange(0, 31, 5))
g.axes[1, 1].set(xlim=(7329.9, 7337.8), xticks=np.arange(7330, 7337, 2.0))
# g.axes[1, 1].set(xlim=(7331.9, 7338.5), xticks=np.arange(7332, 7339, 2.0))
# g.axes[1, 1].set(xlim=(7331.5, 7336.0), xticks=np.arange(7332, 7337, 2.0))
g.axes[2, 2].set(xlim=(0.6, 1.4), xticks=np.arange(0.6, 1.6, 0.3))
g.axes[3, 3].set(xlim=(0.6, 1.4), xticks=np.arange(0.6, 1.6, 0.3))

g.axes[1, 0].set(ylim=(7329.9, 7337.8), yticks=np.arange(7330, 7337, 2.0))
# g.axes[1, 0].set(ylim=(7331.9, 7336.5), yticks=np.arange(7332, 7339, 2.0))
# g.axes[1, 0].set(ylim=(7331.5, 7336.0), yticks=np.arange(7332, 7337, 2.0))
g.axes[2, 0].set(ylim=(0.6, 1.4), yticks=np.arange(0.6, 1.6, 0.3))
g.axes[3, 0].set(ylim=(0.6, 1.4), yticks=np.arange(0.6, 1.6, 0.3))


# Map the diagonal with kernel density plots. Rather than using discrete bins, a Kernel density estimation (KDE) plot smooths the observations with a Gaussian kernel, producing a continuous density estimate:
g.map_diag(sns.histplot, kde=True, bins=100, fill=True)

# Map the lower triangle with kernel density plots
g.map_lower(sns.kdeplot, fill=True)

# Adjust the layout
g.fig.subplots_adjust(top=0.96, bottom=0.09, left=0.07, right=0.95, hspace=0.3, wspace=0.35)

# Create the legend manually with specific colors
legend = plt.legend([plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=50), plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=50)], ['Posterior', 'Prior'], fontsize=100, loc='upper right', bbox_to_anchor=(1.2, 5.0))

# Set the font size and properties of the legend
for text in legend.get_texts():
    text.set_fontsize(100)
    text.set_color("blue" if text.get_text() == "Posterior" else "red")
    text.set_fontname("Times New Roman")

# Add the legend to the figure
g.fig.add_artist(legend)
# g.add_legend(fontsize=100, loc='upper right', title='Distributions', title_fontsize=100)

# Save the figure
plt.savefig(peak + dataset_Tau + '_posterior.png')

# Show the plot (uncomment if you want to display the plot)
# plt.show()
    

print("\n[The End]")
# sys.stdout.close()
