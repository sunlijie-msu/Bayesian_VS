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
data_x_values = np.loadtxt(dir_path + '/DSL_' + peak + '_data_x_values.csv', delimiter=',')  # x values are common for data and model
data_y_values = np.loadtxt(dir_path + '/DSL_' + peak + '_data_y_values.csv', delimiter=',')  # csv derived from h_measured_spec->Print("all"); >>data.txt
data_y_values_var = np.loadtxt(dir_path + '/DSL_' + peak + '_data_y_values_var.csv', delimiter=',')
model_y_values = np.loadtxt(dir_path + '/DSL_' + peak + '_model_y_values.csv', delimiter=',')
model_parameter_values = np.loadtxt(dir_path + '/DSL_' + peak + '_model_parameter_values.csv', delimiter=',')
data_x_values_fullrange = np.loadtxt(dir_path + '/DSL_' + peak + '_data_x_values_fullrange.csv', delimiter=',')  # for visualization only
data_y_values_fullrange = np.loadtxt(dir_path + '/DSL_' + peak + '_data_y_values_fullrange.csv', delimiter=',')  # for visualization only
model_y_values_fullrange = np.loadtxt(dir_path + '/DSL_' + peak + '_model_y_values_fullrange.csv', delimiter=',')  # for visualization only

print("The list data_x_values has", len(data_x_values), "rows and", 1, "column.")
print("The list data_y_values has", len(data_y_values), "rows and", 1, "column.")
print("The list model_y_values has", len(model_y_values), "rows (runs) and", len(model_y_values[0]), "columns (bins).")
print("The list model_parameter_values has", len(model_parameter_values), "rows (runs) and", len(model_parameter_values[0]), "columns (parameters).")
print("The list data_x_values_fullrange has", len(data_x_values_fullrange), "rows and", 1, "column.")
print("The list data_y_values_fullrange has", len(data_y_values_fullrange), "rows and", 1, "column.")
print("The list model_y_values_fullrange has", len(model_y_values_fullrange), "rows (runs) and", len(model_y_values_fullrange[0]), "columns (bins).")

bin_size = 1
num_bins_peak = len(data_x_values)  # the number of unique values in the first column of data_x_values, which corresponds to the number of different x values in the data = 59
num_bins_fullrange = len(data_x_values_fullrange)
peakrange_min = data_x_values[0]
peakrange_max = data_x_values[num_bins_peak-1]
fitrange_min = data_x_values_fullrange[0]
fitrange_max = data_x_values_fullrange[num_bins_fullrange-1]
num_model_runs = len(model_parameter_values)  # The function len() in this case returns the number of rows in the numpy array model_parameter_values_train. This corresponds to the number of model training runs = 324

print("peakrange_min: ", peakrange_min,", peakrange_max: ", peakrange_max)
print("fitrange_min: ", fitrange_min, ", fitrange_max: ", fitrange_max)
print("num_bins_peak: ", num_bins_peak, ",  num_bins_fullrange: ", num_bins_fullrange)


print('How many rows in data_y_values? num_data_y:', data_y_values.shape[0])
print('How many rows in data_x_values? num_bins_peak:', data_x_values.shape[0])
print('How many columns in model_y_values? num_bins_peak_model:', model_y_values.shape[1])
print('How many rows in model_y_values? num_training_model_runs:', model_y_values.shape[0])
print('How many columns in model_parameter_values? num_model_param:', model_parameter_values.shape[1])
print('How many rows in model_parameter_values? num_training_model_runs_param:', model_parameter_values.shape[0])


# Select some model runs as training runs. You may select all the runs.
# rndsample = sample(range(0, 324), 324)  # generates a list of 324 random numbers from the range 0 to 324 and assigns it to the variable rndsample.
if peak == '31S1248':
    number_of_fullmodel_runs = 324
if peak == '31S3076':
    number_of_fullmodel_runs = 2106
if peak == '31S4156':
    number_of_fullmodel_runs = 243
rndsample = list(range(0, number_of_fullmodel_runs, 1))
print('Select training runs: ', rndsample)
model_y_values_train = model_y_values[rndsample, :]  # model_y_values_train is a subset of model_y_values where the rows are selected using the rndsample list and all columns are included by specifying : for the second index.
model_y_values_fullrange_train = model_y_values_fullrange[rndsample, :]  # model_y_values_train is a subset of model_y_values where the rows are selected using the rndsample list and all columns are included by specifying : for the second index.
model_parameter_values_train = model_parameter_values[rndsample, :]
rndsample = list(range(1, number_of_fullmodel_runs, 2))
# rndsample = [250]  # only select run 251 for test
print('Select test runs: ', rndsample)
model_y_values_test = model_y_values[rndsample, :]  # model_y_values_train is a subset of model_y_values where the rows are selected using the rndsample list and all columns are included by specifying : for the second index.
model_parameter_values_test = model_parameter_values[rndsample, :]

# plots a list of profiles in the same figure. Each profile corresponds to a simulation replica for the given instance.
print("[Step 2: Plot model training runs (prior) vs data.]")
plt.rcParams["font.size"] = "18"

fig, ax = plt.subplots(figsize=(15, 5))
fig.subplots_adjust(left=0.07, bottom=0.15, right=0.97, top=0.95)

# Lower Poisson errors
lower_errors = data_y_values_fullrange - poisson.ppf(0.16, data_y_values_fullrange)
# Upper Poisson errors
upper_errors = poisson.ppf(0.84, data_y_values_fullrange) - data_y_values_fullrange

p1 = ax.errorbar(data_x_values_fullrange, data_y_values_fullrange, yerr=[lower_errors, upper_errors], fmt='s', color='black', markersize=2, label='Data', ecolor='black', zorder=2)  # zorder 2 appears on top of the zorder = 1.
# Get min and max model counts in each bin
model_y_value_low = model_y_values_fullrange_train.min(axis=0)
model_y_value_high = model_y_values_fullrange_train.max(axis=0)
# Smooth boundaries of the filled area
smoothed_model_y_value_low = gaussian_filter1d(model_y_value_low, sigma=1.5)
smoothed_model_y_value_high = gaussian_filter1d(model_y_value_high, sigma=1.5)
# Plot shaded region between min and max
p2 = ax.fill_between(data_x_values_fullrange, smoothed_model_y_value_low, smoothed_model_y_value_high,
                     color='red', alpha=0.3, label='Prior (Model)', linewidth=0, zorder=1)
ax.set_ylabel('Counts per 1 keV')
ax.set_xlabel('Energy (keV)')
ax.legend(['Prior', 'Data'], fontsize=22)
ax.tick_params(direction='in')
xmin = min(data_x_values_fullrange) + 70
xmax = max(data_x_values_fullrange) - 89.5
ax.set_xlim(xmin, xmax)
plt.savefig(peak + '_prior.png')
#  plt.show()


# (No filter) Fit an emulator via 'PCGP'
print("[Step 3: Model emulation.]")
emulator_1 = emulator(x=data_x_values, theta=model_parameter_values_train, f=model_y_values_train.T, method='PCGP', args={'epsilon': 0.0000000001})
# this takes time. f can be from an analytic function too
# model_y_values, m runs/rows, n bins/columns, need to be transposed in this case cuz each column in f should correspond to a row in x.
# /usr/local/lib/python3.8/dist-packages/surmise/emulationmethods/PCGP.py

print("[Step 4: Diagnostics plots.]")
pred_emu = emulator_1.predict(data_x_values, model_parameter_values_test)  # predict at some parameter values and x values
pred_m, pred_var = pred_emu.mean(), pred_emu.var()  # Returns the mean and variance at theta and x in the prediction. pred_var represents the diagonal elements of the emulator covariance matrix, namely, the predictive variances.
print("pred_m:", pred_m.shape, "pred_var:", pred_var.shape)  # pred_m: (59, 162) pred_var: (59, 162)
pred_emu_tr = emulator_1.predict(data_x_values, model_parameter_values_train)
pred_m_tr, pred_var_tr = pred_emu_tr.mean(), pred_emu_tr.var()
print("pred_m_tr:", pred_m_tr.shape, "pred_var_tr:", pred_var_tr.shape)  # pred_m_tr: (59, 324) pred_var_tr: (59, 324)

n = pred_m.shape[1]  # returns the number of columns in pred_m = 162
std_err_f_test = ((pred_m - model_y_values_test.T)/np.sqrt(pred_var)).flatten()
std_err_nf_test = np.mean((pred_m - model_y_values_test.T)/np.sqrt(pred_var), axis=0)
fig, axs = plt.subplots(1, 2, figsize=(15, 5))
fig.subplots_adjust(left=0.07, bottom=0.15, right=0.97, top=0.95)
xs = np.repeat(data_x_values, n)
axs[0].scatter(xs, std_err_f_test, alpha=0.5)
axs[0].plot(xs, np.repeat(0, len(xs)), color='red')
axs[0].set_xlabel(r'Energy (keV)')
axs[0].set_ylabel(r'${(\hat{\mu}_{test} - \mu_{test})}/{\hat{\sigma}_{test}}$')
axs[1].scatter(np.arange(0, n), std_err_nf_test)
axs[1].plot(np.arange(0, n), np.repeat(0, n), color='red')
axs[1].set_xlabel(r'Test id')
plt.savefig(peak + '_residual.png')
#  plt.show()


errors_test = (pred_m - model_y_values_test.T).flatten()
errors_tr = (pred_m_tr - model_y_values_train.T).flatten()
sst_test = np.sum((model_y_values_test.T.flatten() - np.mean(model_y_values_test.T.flatten()))**2)
sst_tr = np.sum((model_y_values_train.T.flatten() - np.mean(model_y_values_train.T.flatten()))**2)

fig, axs = plt.subplots(1, 2, figsize=(15, 5))
fig.subplots_adjust(left=0.07, bottom=0.15, right=0.97, top=0.91)
axs[0].scatter(model_y_values_test.T, pred_m, alpha=0.5)
if peak == '31S1248':
    bin_count_max = 90
if peak == '31S3076':
    bin_count_max = 12
if peak == '31S4156':
    bin_count_max = 125
axs[0].plot(range(1, bin_count_max), range(1, bin_count_max), color='red')
axs[0].set_xlabel('Simulator counts in each bin (test)')
axs[0].set_ylabel('Emulator counts in each bin (test)')
axs[0].set_title(r'$r^2=$' + str(np.round(1 - np.sum(errors_test**2)/sst_test, 3)))
axs[1].scatter(model_y_values_train.T, pred_m_tr, alpha=0.5)
axs[1].plot(range(1, bin_count_max), range(1, bin_count_max), color='red')
axs[1].set_xlabel('Simulator counts in each bin (training)')
axs[1].set_ylabel('Emulator counts in each bin (training)')
axs[1].set_title(r'$r^2=$' + str(np.round(1 - np.sum(errors_tr**2)/sst_tr, 3)))
plt.savefig(peak + '_R2.png')
# plt.show()


mu = 0
variance = 1
sigma = np.sqrt(variance)
x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
fig.subplots_adjust(left=0.07, bottom=0.15, right=0.97, top=0.92)
axs[0].hist((pred_m - model_y_values_test.T).flatten(), bins=100)
axs[0].set_title(r'$\hat{\mu}_{test} - \mu_{test}$')
axs[1].hist(std_err_f_test, density=True, bins=80)  # standardized error
axs[1].plot(x, sps.norm.pdf(x, mu, sigma), color='red')
axs[1].set_title(r'${(\hat{\mu}_{test} - \mu_{test})}/{\hat{\sigma}_{test}}$')
axs[2].hist(((pred_m - model_y_values_test.T)/model_y_values_test.T).flatten(), bins=100)  # relative error
axs[2].set_title(r'${(\hat{\mu}_{test} - \mu_{test})}/{\mu_{test}}$')
l = np.arange(-2, 3, 1)/10
axs[2].set(xticks=l, xticklabels=l)
axs[2].axvline(x=0, ls='--', color='red')
plt.savefig(peak + '_error.png')
# plt.show()


print("Rsq = ", 1 - np.round(np.sum(np.square(pred_m - model_y_values_test.T)) / np.sum(np.square(model_y_values_test - np.mean(model_y_values_test.T, axis=1))), 3))  # calculates the R-squared value, which is a measure of how well the emulator fits the simulation model.
print('MSE = ', np.round(np.mean(np.sum(np.square(pred_m - model_y_values_test.T), axis=1)), 2))  # calculates the mean squared error (MSE), which is another measure of the accuracy of the emulator.
print('SSE = ', np.round(np.sum((pred_m - model_y_values_test.T)**2), 2))

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
    def lpdf(theta):  # log-probability density function of the prior for a given set of parameters theta
        return (sps.uniform.logpdf(theta[:, 0], 0, 30) +
                sps.norm.logpdf(theta[:, 1], 4155.84, 0.31) +
                sps.norm.logpdf(theta[:, 2], 1.0, 0.2) +
                sps.norm.logpdf(theta[:, 3], 1.0, 0.4)).reshape((len(theta), 1))

    def rnd(n):  # Generates n random variable from a prior distribution.
        return np.vstack((sps.uniform.rvs(0, 30, size=n),
                          sps.norm.rvs(4155.84, 0.31, size=n),
                          sps.norm.rvs(1.0, 0.2, size=n),
                          sps.norm.rvs(1.0, 0.4, size=n))).T


# obsvar = np.maximum(0.2 * data_y_values, 1.4)
obsvar = data_y_values_var
# print(obsvar)

# Calibrator 1
print("[Step 6: MCMC sampling.]")
number_of_mcmc_samples = 80000
if peak == '31S1248':
    calibrator_1 = calibrator(emu=emulator_1,
                                               y=data_y_values,
                                               x=data_x_values,
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
                              y=data_y_values,
                              x=data_x_values,
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
                                               y=data_y_values,
                                               x=data_x_values,
                                               thetaprior=Prior_DSL31S_4156,
                                               # method='directbayes',
                                               method='directbayeswoodbury',
                                               # method='mlbayeswoodbury',
                                               yvar=obsvar,
                                               args={'theta0': np.array([[5.0, 4155.84, 1.0, 1.0]]),  # initial guess
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


print("[Step 7-1: Plot posteriors of parameters.]")
if peak == '31S1248':
    theta_prior = Prior_DSL31S_1248.rnd(number_of_mcmc_samples)  # draw 1000 random parameters from the prior
    theta_post = calibrator_1.theta.rnd(number_of_mcmc_samples)
    dfpost = pd.DataFrame(theta_post, columns=['Tau', 'Eg', 'bkg', 'SP'])
    dfprior = pd.DataFrame(theta_prior, columns=['Tau', 'Eg', 'bkg', 'SP'])

if peak == '31S3076':
    theta_prior = Prior_DSL31S_3076.rnd(number_of_mcmc_samples)  # draw 1000 random parameters from the prior
    theta_post = calibrator_1.theta.rnd(number_of_mcmc_samples)
    dfpost = pd.DataFrame(theta_post, columns=['Tau', 'Eg', 'bkg', 'SP', 'AC'])
    dfprior = pd.DataFrame(theta_prior, columns=['Tau', 'Eg', 'bkg', 'SP', 'AC'])

if peak == '31S4156':
    theta_prior = Prior_DSL31S_4156.rnd(number_of_mcmc_samples)  # draw 1000 random parameters from the prior
    theta_post = calibrator_1.theta.rnd(number_of_mcmc_samples)
    dfpost = pd.DataFrame(theta_post, columns=['Tau', 'Eg', 'bkg', 'SP'])
    dfprior = pd.DataFrame(theta_prior, columns=['Tau', 'Eg', 'bkg', 'SP'])

print("One prior sample: ", theta_prior[0])
print("One posterior sample:", theta_post[0])

#  a function from the Pandas library that is used to create a dataframe object. Dataframe is a two-dimensional, size-mutable, and tabular data structure with columns of potentially different types.

df = pd.concat([dfprior, dfpost])
#  It is used to combine multiple DataFrames or Series into a single DataFrame or Series. It can combine the dfprior and dfpost DataFrames vertically.
pr = ['prior' for i in range(number_of_mcmc_samples)]
ps = ['posterior' for i in range(number_of_mcmc_samples)]
pr.extend(ps)
df['distribution'] = pr
#  The pr and ps lists are created to add a new column 'distribution' to the combined DataFrame indicating whether a row belongs to the prior or posterior distribution.
sns.set(style="white")
g = sns.PairGrid(df, palette=["red", "blue"], corner=True, diag_sharey=False, hue='distribution')
g.map_diag(sns.kdeplot, shade=True)
g.map_lower(sns.kdeplot, fill=True)
g.add_legend()
g.fig.subplots_adjust(top=0.96, bottom=0.08, left=0.10, right=0.95)
plt.savefig(peak + '_posterior.png')
#  plt.show()


print("[Step 7-2: Plot predictions with calibrated parameters.]")

def plot_pred_interval(calib):
    pred = calib.predict(data_x_values)
    rndm_m = pred.rnd(s=number_of_mcmc_samples//2)
    fig, ax = plt.subplots(figsize=(15, 5))
    fig.subplots_adjust(left=0.07, bottom=0.15, right=0.97, top=0.95)
    plt.rcParams["font.size"] = "18"
    # Lower Poisson errors
    lower_errors = data_y_values_fullrange - poisson.ppf(0.16, data_y_values_fullrange)
    # Upper Poisson errors
    upper_errors = poisson.ppf(0.84, data_y_values_fullrange) - data_y_values_fullrange
    p1 = ax.errorbar(data_x_values_fullrange, data_y_values_fullrange, yerr=[lower_errors, upper_errors], fmt='s', color='black', markersize=2, label='Data', ecolor='black', zorder=2)  # zorder 2 appears on top of the zorder = 1.

    posterior_y_upper = np.percentile(rndm_m[:, 0: num_bins_peak], 97.5, axis=0)
    posterior_y_lower = np.percentile(rndm_m[:, 0: num_bins_peak], 2.5, axis=0)
    posterior_y_median = np.percentile(rndm_m[:, 0: num_bins_peak], 50, axis=0)

    p2 = ax.plot(data_x_values, posterior_y_median, color='blue', alpha=1.0, zorder=3)
    p3 = ax.fill_between(data_x_values, posterior_y_lower, posterior_y_upper,
                         color='blue', alpha=0.3, label='Prior (Model)', linewidth=0, zorder=1)
    ax.set_ylabel('Counts per 1 keV')
    ax.set_xlabel('Energy (keV)')
    #  ax.legend(['Posterior', 'Data'], fontsize=22)
    ax.legend([p1[0], p2[0], p3], ['Data', 'Prediction Mean', '95% Credible Interval'], fontsize=19)
    ax.tick_params(direction='in')
    xmin = min(data_x_values_fullrange) + 70
    xmax = max(data_x_values_fullrange) - 89.5
    ax.set_xlim(xmin, xmax)
    plt.savefig(peak + '_prediction.png')
    #  plt.show()

plot_pred_interval(calibrator_1)

print("[Step 7-3: Plot posterior samples.]")
def plot_theta(calib, whichtheta):
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    fig.subplots_adjust(left=0.07, bottom=0.15, right=0.97, top=0.95)
    cal_theta = calib.theta.rnd(number_of_mcmc_samples//2)
    axs[0].plot(cal_theta[:, whichtheta])
    axs[1].boxplot(cal_theta[:, whichtheta])
    axs[2].hist(cal_theta[:, whichtheta], bins=75)
    axs[2].set_xlim([0, 15])
    plt.savefig(peak + '_trace.png')
    #  plt.show()


plot_theta(calibrator_1, 0)

print("[The End]")
