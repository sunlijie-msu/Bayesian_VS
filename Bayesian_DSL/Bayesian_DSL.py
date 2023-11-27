import numpy as np
import matplotlib.pyplot as plt
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


# Read data from CSV files
print("[Step 1: Read and check input files.]")
project_path = os.path.abspath("Bayesian_DSL.py")
dir_path = os.path.dirname(project_path)

peak = '31S4156'
fakeTau = '_0fs' # '_0fs' or '_5fs' or '_10fs'

data_fullrange = np.loadtxt(dir_path + '\DSL_' + peak + fakeTau + '\DSL_' + peak + '_data.csv', delimiter=',')  # Read full range data # csv derived from histogram_get_bin_content_error.C
data_peakrange = data_fullrange[130:240, :] # Select data in the peak range by rows

data_x_values_fullrange = data_fullrange[:, 0]
data_y_values_fullrange = data_fullrange[:, 1]
data_y_varlow_fullrange = data_fullrange[:, 2]
data_y_varhigh_fullrange = data_fullrange[:, 3]

data_x_values_peakrange = data_peakrange[:, 0]
data_y_values_peakrange = data_peakrange[:, 1]
data_y_varlow_peakrange = data_peakrange[:, 2]
data_y_varhigh_peakrange = data_peakrange[:, 3]

model_parameter_values = np.loadtxt(dir_path + '\DSL_' + peak + fakeTau + '\DSL_' + peak + '_model_parameter_values.csv', delimiter=',')  # csv derived from Comparison_DSL2.C
model_y_values_fullrange = np.loadtxt(dir_path + '\DSL_' + peak + fakeTau + '\DSL_' + peak + '_model_y_values.csv', delimiter=',')   # csv derived from Comparison_DSL2.C
model_y_values_peakrange = model_y_values_fullrange[:,130:240] # Select model in the peak range by columns


print("Path: ", dir_path + '\DSL_' + peak + fakeTau + '\DSL_' + peak + '_data.csv')
print("Dimensions of data_fullrange:")
print("Shape:", data_fullrange.shape)
print("Rows:", len(data_fullrange))
print("Columns:", len(data_fullrange[0]))

print("\nDimensions of data_peakrange:")
print("Shape:", data_peakrange.shape)
print("Rows:", len(data_peakrange))
print("Columns:", len(data_peakrange[0]))

print("\nDimensions of data_x_values_fullrange:")
print("Shape:", data_x_values_fullrange.shape)

print("\nDimensions of data_y_values_fullrange:")
print("Shape:", data_y_values_fullrange.shape)

print("\nDimensions of data_y_values_var_low_fullrange:")
print("Shape:", data_y_varlow_fullrange.shape)

print("\nDimensions of data_y_values_var_high_fullrange:")
print("Shape:", data_y_varhigh_fullrange.shape)

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

print("\nDimensions of model_y_values_fullrange:")
print("Shape:", model_y_values_fullrange.shape)
print("Rows:", len(model_y_values_fullrange))
print("Columns:", len(model_y_values_fullrange[0]))

print("\nDimensions of model_y_values_peakrange:")
print("Shape:", model_y_values_peakrange.shape)
print("Rows:", len(model_y_values_peakrange))
print("Columns:", len(model_y_values_peakrange[0]))

bin_size = 1
num_bins_peak = len(data_x_values_peakrange)  # the number of unique values in the first column of data_x_values, which corresponds to the number of different x values in the data = 59
num_bins_fullrange = len(data_x_values_fullrange)
peakrange_min = data_x_values_peakrange[0]
peakrange_max = data_x_values_peakrange[num_bins_peak-1]
fitrange_min = data_x_values_fullrange[0]
fitrange_max = data_x_values_fullrange[num_bins_fullrange-1]
num_model_runs = len(model_parameter_values)  # len(2Darray) will give you the number of rows in the 2D array.

print("peakrange_min: ", peakrange_min,", peakrange_max: ", peakrange_max)
print("fitrange_min: ", fitrange_min, ", fitrange_max: ", fitrange_max)
print("num_bins_peakrange: ", num_bins_peak, ",  num_bins_fullrange: ", num_bins_fullrange)

num_bins_peak = len(model_y_values_peakrange[0])  # len(2Darray[0]) will give you the number of elements in the first row of the 2D array, i.e., the number of columns in the 2D array.
num_bins_fullrange = len(model_y_values_fullrange[0])
print("num_bins_peakrange: ", num_bins_peak, ",  num_bins_fullrange: ", num_bins_fullrange)

# Select some model runs as training runs. You may select all the runs.
# rndsample = sample(range(0, 324), 324)  # generates a list of 324 random numbers from the range 0 to 324 and assigns it to the variable rndsample.
if peak == '31S1248':
    number_of_fullmodel_runs = 324
if peak == '31S3076':
    number_of_fullmodel_runs = 2106
if peak == '31S4156':
    number_of_fullmodel_runs = 297
rndsample = list(range(0, number_of_fullmodel_runs, 1))
print('Select training runs: ', rndsample)
model_y_values_peakrange_train = model_y_values_peakrange[rndsample, :]  # model_y_values_train is a subset of model_y_values where the rows are selected using the rndsample list and all columns are included by specifying : for the second index.
model_y_values_fullrange_train = model_y_values_fullrange[rndsample, :]  # model_y_values_train is a subset of model_y_values where the rows are selected using the rndsample list and all columns are included by specifying : for the second index. # for visualization purpose only
model_parameter_values_train = model_parameter_values[rndsample, :]
rndsample = list(range(1, number_of_fullmodel_runs, 2))
# rndsample = [250]  # only select run 251 for test
print('Select test runs: ', rndsample)
model_y_values_peakrange_test = model_y_values_peakrange[rndsample, :]  # model_y_values_train is a subset of model_y_values where the rows are selected using the rndsample list and all columns are included by specifying : for the second index.
model_parameter_values_test = model_parameter_values[rndsample, :]

# plots a list of profiles in the same figure. Each profile corresponds to a simulation replica for the given instance.

plt.rcParams['font.size'] = 19
plt.rcParams['font.family'] = 'Times New Roman'
# plt.rcParams['font.weight'] = 'bold'

print("[Step 2: Plot model training runs (prior) vs data.]")
fig, ax_prior = plt.subplots(figsize=(15, 5))
fig.subplots_adjust(left=0.07, bottom=0.16, right=0.97, top=0.95)

p1 = ax_prior.errorbar(data_x_values_fullrange, data_y_values_fullrange, yerr=[data_y_varlow_fullrange,data_y_varhigh_fullrange], fmt='s', color='black', markersize=1, label='Data', ecolor='black', zorder=2)  # zorder 2 appears on top of the zorder = 1.

# Get min and max model counts in each bin
model_lowcount = model_y_values_fullrange_train.min(axis=0)
model_highcount = model_y_values_fullrange_train.max(axis=0)
# Smooth boundaries of the filled area
smoothed_model_lowcount = gaussian_filter1d(model_lowcount, sigma=1.5)
smoothed_model_highcount = gaussian_filter1d(model_highcount, sigma=1.5)
# Plot shaded region between min and max
p2 = ax_prior.fill_between(data_x_values_fullrange, smoothed_model_lowcount, smoothed_model_highcount,
                     color='red', alpha=0.3, linewidth=0, zorder=1)
ax_prior.set_ylabel('Counts per 1 keV', fontsize=22)
ax_prior.set_xlabel('Energy (keV)', fontsize=22)
ax_prior.legend(['Prior', 'Data'], fontsize=24, loc='upper left')
ax_prior.tick_params(axis='both', which='major', labelsize=22)
# ax.tick_params(direction='in')
xmin = min(data_x_values_fullrange) + 70
xmax = max(data_x_values_fullrange) - 89.5
ax_prior.set_xlim(xmin, xmax)
ymax = max(data_y_values_fullrange) + 7
ax_prior.set_ylim(0, ymax)
plt.savefig(peak + '_prior.png')
# plt.show()


# (No filter) Fit an emulator via 'PCGP'
print("[Step 3: Model emulation.]")
emulator_1 = emulator(x=data_x_values_peakrange, theta=model_parameter_values_train, f=model_y_values_peakrange_train.T, method='PCGP', args={'epsilon': 0.0000000001})
# this takes time. f can be from an analytic function too
# model_y_values, m runs/rows, n bins/columns, need to be transposed in this case cuz each column in f should correspond to a row in x.
# /usr/local/lib/python3.8/dist-packages/surmise/emulationmethods/PCGP.py
# C:\Users\sun\AppData\Local\Programs\Python\Python311\Lib\site-packages\surmise\emulationmethods
print("[Step 4: Diagnostics plots.]")
pred_emu = emulator_1.predict(data_x_values_peakrange, model_parameter_values_test)  # predict at some parameter values and x values
pred_m, pred_var = pred_emu.mean(), pred_emu.var()  # Returns the mean and variance at theta and x in the prediction. pred_var represents the diagonal elements of the emulator covariance matrix, namely, the predictive variances.
print("pred_m:", pred_m.shape, "pred_var:", pred_var.shape)  # pred_m: (59, 162) pred_var: (59, 162)
pred_emu_tr = emulator_1.predict(data_x_values_peakrange, model_parameter_values_train)
pred_m_tr, pred_var_tr = pred_emu_tr.mean(), pred_emu_tr.var()
print("pred_m_tr:", pred_m_tr.shape, "pred_var_tr:", pred_var_tr.shape)  # pred_m_tr: (59, 324) pred_var_tr: (59, 324)

n = pred_m.shape[1]  # returns the number of columns in pred_m = 162
std_err_f_test = ((pred_m - model_y_values_peakrange_test.T)/np.sqrt(pred_var)).flatten()
std_err_nf_test = np.mean((pred_m - model_y_values_peakrange_test.T)/np.sqrt(pred_var), axis=0)
fig, axs_emu1 = plt.subplots(1, 2, figsize=(15, 5))
fig.subplots_adjust(left=0.07, bottom=0.15, right=0.97, top=0.95)
xs = np.repeat(data_x_values_peakrange, n)
axs_emu1[0].scatter(xs, std_err_f_test, alpha=0.5)
axs_emu1[0].plot(xs, np.repeat(0, len(xs)), color='red')
axs_emu1[0].set_xlabel(r'Energy (keV)')
axs_emu1[0].set_ylabel(r'${(\hat{\mu}_{test} - \mu_{test})}/{\hat{\sigma}_{test}}$')
axs_emu1[1].scatter(np.arange(0, n), std_err_nf_test)
axs_emu1[1].plot(np.arange(0, n), np.repeat(0, n), color='red')
axs_emu1[1].set_xlabel(r'Test id')
plt.savefig(peak + '_residual.png')
#  plt.show()


errors_test = (pred_m - model_y_values_peakrange_test.T).flatten()
errors_tr = (pred_m_tr - model_y_values_peakrange_train.T).flatten()
sst_test = np.sum((model_y_values_peakrange_test.T.flatten() - np.mean(model_y_values_peakrange_test.T.flatten()))**2)
sst_tr = np.sum((model_y_values_peakrange_train.T.flatten() - np.mean(model_y_values_peakrange_train.T.flatten()))**2)

fig, axs_emu2 = plt.subplots(1, 2, figsize=(15, 5))
fig.subplots_adjust(left=0.07, bottom=0.15, right=0.97, top=0.91)
axs_emu2[0].scatter(model_y_values_peakrange_test.T, pred_m, alpha=0.5)
if peak == '31S1248':
    bin_count_max = 90
if peak == '31S3076':
    bin_count_max = 12
if peak == '31S4156':
    bin_count_max = 18
axs_emu2[0].plot(range(2, bin_count_max), range(2, bin_count_max), color='red')
axs_emu2[0].set_xlabel('Simulator counts in each bin (test)')
axs_emu2[0].set_ylabel('Emulator counts in each bin (test)')
axs_emu2[0].set_title(r'$r^2=$' + str(np.round(1 - np.sum(errors_test**2)/sst_test, 3)))
axs_emu2[1].scatter(model_y_values_peakrange_train.T, pred_m_tr, alpha=0.5)
axs_emu2[1].plot(range(2, bin_count_max), range(2, bin_count_max), color='red')
axs_emu2[1].set_xlabel('Simulator counts in each bin (training)')
axs_emu2[1].set_ylabel('Emulator counts in each bin (training)')
axs_emu2[1].set_title(r'$r^2=$' + str(np.round(1 - np.sum(errors_tr**2)/sst_tr, 3)))
plt.savefig(peak + '_R2.png')
# plt.show()


mu = 0
variance = 1
sigma = np.sqrt(variance)
x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
fig, axs_emu3 = plt.subplots(1, 3, figsize=(15, 5))
fig.subplots_adjust(left=0.07, bottom=0.13, right=0.97, top=0.90)
axs_emu3[0].hist((pred_m - model_y_values_peakrange_test.T).flatten(), bins=100)
axs_emu3[0].set_title(r'$\hat{\mu}_{test} - \mu_{test}$')
axs_emu3[1].hist(std_err_f_test, density=True, bins=80)  # standardized error
axs_emu3[1].plot(x, sps.norm.pdf(x, mu, sigma), color='red')
axs_emu3[1].set_title(r'${(\hat{\mu}_{test} - \mu_{test})}/{\hat{\sigma}_{test}}$')
axs_emu3[2].hist(((pred_m - model_y_values_peakrange_test.T)/model_y_values_peakrange_test.T).flatten(), bins=100)  # relative error
axs_emu3[2].set_title(r'${(\hat{\mu}_{test} - \mu_{test})}/{\mu_{test}}$')
l = np.arange(-2, 3, 1)/10
axs_emu3[2].set(xticks=l, xticklabels=l)
axs_emu3[2].axvline(x=0, ls='--', color='red')
axs_emu3[2].set_xlim([-0.25, 0.25])
plt.savefig(peak + '_error.png')
# plt.show()


print("Rsq = ", 1 - np.round(np.sum(np.square(pred_m - model_y_values_peakrange_test.T)) / np.sum(np.square(model_y_values_peakrange_test - np.mean(model_y_values_peakrange_test.T, axis=1))), 3))  # calculates the R-squared value, which is a measure of how well the emulator fits the simulation model.
print('MSE = ', np.round(np.mean(np.sum(np.square(pred_m - model_y_values_peakrange_test.T), axis=1)), 2))  # calculates the mean squared error (MSE), which is another measure of the accuracy of the emulator.
print('SSE = ', np.round(np.sum((pred_m - model_y_values_peakrange_test.T)**2), 2))

# Define a class for prior of 4 parameters
print("[Step 5: Prior class specification.]")
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
                sps.norm.logpdf(theta[:, 2], 1.0, 0.1) +
                sps.norm.logpdf(theta[:, 3], 1.0, 0.2)).reshape((len(theta), 1))
    
    def rnd(n):  # Generates n random variables (rvs) from a prior distribution.
        return np.vstack((sps.uniform.rvs(0, 30, size=n),
                          sps.norm.rvs(4155.84, 0.31, size=n),
                          sps.norm.rvs(1.0, 0.1, size=n),
                          sps.norm.rvs(1.0, 0.2, size=n))).T


# obsvar = np.maximum(0.2 * data_y_values, 1.4)
obsvar = (data_y_varlow_peakrange + data_y_varhigh_peakrange)/2
# print(obsvar)

# Calibrator 1
print("[Step 6: MCMC sampling.]")
number_of_mcmc_samples = 80000
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
                                                          'numsamp': number_of_mcmc_samples,
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
                                                          'numsamp': number_of_mcmc_samples,
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
                                               method='directbayeswoodbury',
                                               # method='mlbayeswoodbury',
                                               yvar=obsvar,
                                               args={'theta0': np.array([[0.0, 4155.84, 1.0, 1.0]]),  # initial guess ['Tau', 'Eg', 'Bkg', 'SP']
                                                          'sampler': 'metropolis_hastings',
                                                          # 'sampler': 'LMC',
                                                          # 'sampler': 'PTLMC',
                                                          'numsamp': number_of_mcmc_samples,
                                                          'numchain': 8,
                                                          'stepType': 'normal',
                                                          'burnSamples': 1000,
                                                          'verbose': True
                                                          # 'stepParam': np.array([1, 0.01, 0.01, 0.01])
                                                          }
                                              )


print("[Step 7-1: Plot predictions with calibrated parameters.]")

def plot_pred_interval(calib):
    pred = calib.predict(data_x_values_peakrange)
    rndm_m = pred.rnd(s=number_of_mcmc_samples//2)
    fig, ax_post_predict = plt.subplots(figsize=(15, 5))
    fig.subplots_adjust(left=0.07, bottom=0.16, right=0.97, top=0.95)

    p1 = ax_post_predict.errorbar(data_x_values_fullrange, data_y_values_fullrange, yerr=[data_y_varlow_fullrange,data_y_varhigh_fullrange], fmt='s', color='black', markersize=1, label='Data', ecolor='black', zorder=2)  # zorder 2 appears on top of the zorder = 1.

    posterior_y_upper = np.percentile(rndm_m[:, 0: num_bins_peak], 97.5, axis=0)
    posterior_y_lower = np.percentile(rndm_m[:, 0: num_bins_peak], 2.5, axis=0)
    posterior_y_median = np.percentile(rndm_m[:, 0: num_bins_peak], 50, axis=0)

    p2 = ax_post_predict.plot(data_x_values_peakrange, posterior_y_median, color='blue', alpha=1.0, zorder=1)
    p3 = ax_post_predict.fill_between(data_x_values_peakrange, posterior_y_lower, posterior_y_upper, color='blue', alpha=0.3, linewidth=0, zorder=1)
    ax_post_predict.tick_params(axis='both', which='major', labelsize=22)
    ax_post_predict.set_ylabel('Counts per 1 keV', fontsize=22)
    ax_post_predict.set_xlabel('Energy (keV)', fontsize=22)
    ax_post_predict.legend(['Prediction Mean', '95% Credible Interval', 'Data'], fontsize=24, loc='upper left')
    xmin = min(data_x_values_fullrange) + 70
    xmax = max(data_x_values_fullrange) - 89.5
    # ax_post_predict.tick_params(direction='out')
    # ax_post_predict.set_xticks(np.arange(xmin-0.5, xmax+1, step=25))
    ymax = max(data_y_values_fullrange) + 7
    ax_post_predict.set_xlim(xmin, xmax)
    ax_post_predict.set_ylim(0, ymax)
    plt.savefig(peak + '_prediction.png')
    # plt.show()

plot_pred_interval(calibrator_1)

print("[Step 7-2: Plot posterior samples.]")
def plot_theta(calib, whichtheta):
    fig, axs_trace = plt.subplots(1, 3, figsize=(15, 5))
    fig.subplots_adjust(left=0.07, bottom=0.16, right=0.97, top=0.95, wspace=0.3)
    cal_theta = calib.theta.rnd(number_of_mcmc_samples//2)
    axs_trace[0].plot(cal_theta[:, whichtheta])
    axs_trace[0].set_xlabel("Iteration", fontsize=22)
    axs_trace[0].set_ylabel(r"$\tau$ (fs)", fontsize=22)

    axs_trace[1].boxplot(cal_theta[:, whichtheta])
    axs_trace[1].set_xlabel(r"$\tau$", fontsize=22)
    axs_trace[1].set_ylabel(r"$\tau$ (fs)", fontsize=22)

    axs_trace[2].hist(cal_theta[:, whichtheta], bins=200, range=[0, 20])
    axs_trace[2].set_xlabel(r"$\tau$ (fs)", fontsize=22)
    axs_trace[2].set_ylabel("Counts per 0.1 fs", fontsize=22)
    axs_trace[2].set_xlim([0, 20])
    
    axs_trace[0].tick_params(axis='both', which='major', labelsize=22)
    axs_trace[1].tick_params(axis='both', which='major', labelsize=22)
    axs_trace[2].tick_params(axis='both', which='major', labelsize=22)
    
    samples = cal_theta[:, whichtheta]
    sorted_samples = np.sort(samples)
    percentiles = [16, 50, 84, 90]
    for percentile in percentiles:
        index = int(np.round(len(sorted_samples) * (percentile / 100.0)) - 1)
        value = sorted_samples[index]
        print(f"{percentile}th percentile: {value:.2f}")
    plt.savefig(peak + '_trace.png')
    # plt.show()

plot_theta(calibrator_1, 0)


print("[Step 7-3: Plot 2D posteriors of parameters.]")
if peak == '31S1248':
    theta_prior = Prior_DSL31S_1248.rnd(number_of_mcmc_samples)  # draw 1000 random parameters from the prior
    theta_post = calibrator_1.theta.rnd(number_of_mcmc_samples)
    dfpost = pd.DataFrame(theta_post, columns=['Tau', 'Eg', 'Bkg', 'SP'])
    dfprior = pd.DataFrame(theta_prior, columns=['Tau', 'Eg', 'Bkg', 'SP'])

if peak == '31S3076':
    theta_prior = Prior_DSL31S_3076.rnd(number_of_mcmc_samples)  # draw 1000 random parameters from the prior
    theta_post = calibrator_1.theta.rnd(number_of_mcmc_samples)
    dfpost = pd.DataFrame(theta_post, columns=['Tau', 'Eg', 'Bkg', 'SP', 'AC'])
    dfprior = pd.DataFrame(theta_prior, columns=['Tau', 'Eg', 'Bkg', 'SP', 'AC'])

if peak == '31S4156':
    theta_prior = Prior_DSL31S_4156.rnd(number_of_mcmc_samples)  # draw 1000 random parameters from the prior
    theta_post = calibrator_1.theta.rnd(number_of_mcmc_samples)
    dfpost = pd.DataFrame(theta_post, columns=['Tau', 'Eg', 'Bkg', 'SP'])
    dfprior = pd.DataFrame(theta_prior, columns=['Tau', 'Eg', 'Bkg', 'SP'])

print("One prior sample: ", theta_prior[0])
print("One posterior sample:", theta_post[0])

#  a function from the Pandas library that is used to create a dataframe object. Dataframe is a two-dimensional, size-mutable, and tabular data structure with columns of potentially different types.

df = pd.concat([dfprior, dfpost])
#  It is used to combine multiple DataFrames or Series into a single DataFrame or Series. It can combine the dfprior and dfpost DataFrames vertically.
pr = ['prior' for i in range(number_of_mcmc_samples)]
ps = ['posterior' for i in range(number_of_mcmc_samples)]
pr.extend(ps)
df['Distributions'] = pr
#  The pr and ps lists are created to add a new column 'distribution' to the combined DataFrame indicating whether a row belongs to the prior or posterior distribution.
plt.rcParams["font.family"] = "Times New Roman"
# Set Seaborn style
sns.set(style="white")

# Create a PairGrid
g = sns.PairGrid(df, palette=["red", "blue"], corner=True, diag_sharey=False, hue='Distributions')

# Adjust the figure size to make it wider
fig = g.fig
fig.set_size_inches(16, 12)

# Iterate through all axes 
for ax in g.axes.flat:
   if ax:
       ax.xaxis.set_ticks_position('bottom')
       ax.yaxis.set_ticks_position('left')
       ax.set_xlabel(ax.get_xlabel(), fontsize=19) 
       ax.set_ylabel(ax.get_ylabel(), fontsize=19)


# Set font for all tick labels   
for ax in g.fig.axes:
    for label in ax.get_xticklabels():
        label.set_fontname("Times New Roman")
        label.set_fontsize(19) 
    for label in ax.get_yticklabels():
        label.set_fontname("Times New Roman")
        label.set_fontsize(19)


# Add yaxes to 1st column 
for ax in g.axes[:,0]:
    ax.yaxis.set_visible(True)
    ax.set_ylabel(ax.get_ylabel(), fontsize=19)

# Add xaxes to 4th row
for ax in g.axes[3,:]: 
    ax.xaxis.set_visible(True)
    ax.set_xlabel(ax.get_xlabel(), fontsize=19)


g.axes[0, 0].set(xlim=(0, 30), xticks=np.arange(0, 31, 5))
g.axes[1, 1].set(xlim=(4154.8, 4157), xticks=np.arange(4155, 4158, 1.0))
g.axes[2, 2].set(xlim=(0.5, 1.5), xticks=np.arange(0.5, 1.6, 0.25))
g.axes[3, 3].set(xlim=(0, 2.1), xticks=np.arange(0, 2.1, 0.5))

g.axes[1, 0].set(ylim=(4154.8, 4157), yticks=np.arange(4155, 4158, 1.0))
g.axes[2, 0].set(ylim=(0.5, 1.5), yticks=np.arange(0.5, 1.6, 0.25))
g.axes[3, 0].set(ylim=(0, 2.1), yticks=np.arange(0, 2.1, 0.5))

# # Set ticks for 1st column yaxes  
# g.axes[0,0].set_yticks(np.arange(0, 31, 5))
# g.axes[1,0].set_yticks(np.arange(4154.8, 4157, 1.0)) 
# g.axes[2,0].set_yticks(np.arange(0.5, 1.6, 0.5))
# g.axes[3,0].set_yticks(np.arange(0, 2.1, 0.5))

# # Set ticks for 4th row xaxes
# g.axes[3,0].set_xticks(np.arange(0, 31, 5))
# g.axes[3,1].set_xticks(np.arange(4154.8, 4157, 1.0))
# g.axes[3,2].set_xticks(np.arange(0.5, 1.6, 0.5)) 
# g.axes[3,3].set_xticks(np.arange(0, 2.1, 0.5))

# Map the diagonal with kernel density plots
g.map_diag(sns.kdeplot, fill=True)

# Map the lower triangle with kernel density plots
g.map_lower(sns.kdeplot, fill=True)

# Adjust the layout
g.fig.subplots_adjust(top=0.96, bottom=0.09, left=0.07, right=0.95, hspace=0.3, wspace=0.3)

g.fig.legend(labels=['Posterior', 'Prior'], fontsize=28)

# Save the figure
plt.savefig(peak + '_posterior.png')

# Show the plot (uncomment if you want to display the plot)
# plt.show()
    

print("[The End]")
