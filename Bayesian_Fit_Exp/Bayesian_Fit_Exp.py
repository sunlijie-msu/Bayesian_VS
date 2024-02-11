import numpy as np
import matplotlib.pyplot as plt
import emcee
import corner
import seaborn as sns
import pandas as pd
import scipy.stats as sps



print("[Step 1: Read data from CSV input files and plot data.]")

# Load data fullrange csv files and model fullrange csv files and model parameter values csv files
data_fullrange = np.loadtxt('F:/e21010/pxct/timing_msdtotal_e_5361_5481_msd26_t_bin1ns.csv', delimiter=',')  # Read full range data # csv derived from histogram
bin_start = 1500 + 160 # don't change the 1500, which is an offset
# bin 0 is the first; bin 1650 is the 1651st = 150.5 ns;  bin 2040 is the 2041st = 540 ns
bin_stop = 1500 + 1420 # don't change the 1500, which is an offset
# bin 2920 is the 2921st = 1420.5 ns (not included), so the last included bin is 1420.5 - 1 = 1419.5 ns
data_peakrange = data_fullrange[bin_start:bin_stop, :] # Select data in the peak range by rows
data_x_values_peakrange = data_peakrange[:, 0]
data_y_values_peakrange = data_peakrange[:, 1]
data_y_varlow_peakrange = data_peakrange[:, 2]
data_y_varhigh_peakrange = data_peakrange[:, 3]

data_y_errors_peakrange = (data_y_varlow_peakrange + data_y_varhigh_peakrange)/2
data_x_errors_peakrange = 0.5

num_bins_peak = len(data_x_values_peakrange)  # the number of unique values in the first column of data_x_values, which corresponds to the number of different x values in the data
peakrange_min = data_x_values_peakrange[0]
peakrange_max = data_x_values_peakrange[num_bins_peak-1]
print("peakrange_min: ", peakrange_min,", peakrange_max: ", peakrange_max)

plt.rcParams['axes.linewidth'] = 3.0
plt.rcParams['font.size'] = 60
font_family_options = ['Times New Roman', 'Georgia', 'Cambria', 'Courier New', 'serif']
plt.rcParams['font.family'] = font_family_options
# plt.rcParams['mathtext.default'] = 'regular'
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Times New Roman'
plt.rcParams['mathtext.it'] = 'Times New Roman:italic'

fig, ax_prior = plt.subplots(figsize=(36, 12))
fig.subplots_adjust(left=0.08, bottom=0.18, right=0.98, top=0.96)

ax_prior.errorbar(data_x_values_peakrange, data_y_values_peakrange, yerr=[data_y_varlow_peakrange,data_y_varhigh_peakrange], fmt='s', color='black', linewidth=2, markersize=2, label='Data', ecolor='black', zorder=2)  # zorder 2 appears on top of the zorder = 1.
ax_prior.tick_params(axis='both', which='major', labelsize=60, length=9, width=2)
# ax.tick_params(direction='in')
ax_prior.set_xlabel("Time difference LEGe - MSD26 (ns)", fontsize=60, labelpad=20)
ax_prior.set_ylabel("Counts per 1 ns", fontsize=60, labelpad=11)
xmin = min(data_x_values_peakrange) - 0.5
xmax = max(data_x_values_peakrange) + 0.5
print("xmin: ", xmin,", xmax: ", xmax)
ax_prior.set_xlim(xmin, xmax)
ymax = max(data_y_values_peakrange) + max(data_y_varhigh_peakrange) * 1.5
ax_prior.set_ylim(0, ymax)

# Adjust the width of the frame
for spine in ax_prior.spines.values():
    spine.set_linewidth(3)  # Set the linewidth to make the frame wider
plt.savefig('Exp_data.png')


print("[Step 2: Define the exponential decay function (Model).]")
def exponential_model(parameters, x_values):
    total_decays, half_life, background = parameters
    return total_decays / half_life * 0.693147 * np.exp(x_values / half_life * -0.693147) + background


print("[Step 3: Define the log-likelihood function (Likelihood).]")
def log_likelihood(parameters, x_values, y_values, x_errors, y_errors):
    total_decays, half_life, background = parameters
    model_values = exponential_model(parameters, x_values)
    model_values_1 = exponential_model(parameters, x_values + x_errors / 2)
    model_values_2 = exponential_model(parameters, x_values - x_errors / 2)
    total_error_squared = y_errors**2 + ((model_values_1 - model_values_2) / 2)**2
    return -0.5 * np.sum((y_values - model_values)**2 / total_error_squared + np.log(total_error_squared))


print("[Step 4: Define the log-prior function (Prior).]")
def log_prior(parameters):
    total_decays, half_life, background = parameters
    if 6e6 < total_decays < 9e6 and 57.0 < half_life < 80.0 and 2.0 < background < 16.0:
        return 0.0
    return -np.inf


print("[Step 5: Define the log-posterior function (Posterior).]")
def log_posterior(parameters, x_values, y_values, x_errors, y_errors):
    log_prior_val = log_prior(parameters)
    if not np.isfinite(log_prior_val):
        return -np.inf
    return log_prior_val + log_likelihood(parameters, x_values, y_values, x_errors, y_errors)


print("[Step 6: Set up the MCMC sampler.]")
num_walkers = 100
num_dimensions = 3

sampler = emcee.EnsembleSampler(num_walkers, num_dimensions, log_posterior, args=(data_x_values_peakrange, data_y_values_peakrange, data_x_errors_peakrange, data_y_errors_peakrange))


print("[Step 7: Initialize the MCMC walkers.]")
initial_positions = np.zeros((num_walkers, num_dimensions))
initial_positions[:, 0] =  (0.9 + 0.2 * np.random.rand(num_walkers)) * 7e6    # initial slope between 1 and 10
initial_positions[:, 1] =  (0.9 + 0.2 * np.random.rand(num_walkers)) * 68.0  # initial intercept between -150 and 200
initial_positions[:, 2] =  (0.9 + 0.2 * np.random.rand(num_walkers)) * 9  # initial intercept between -150 and 200
# these ranges are just for the initial positions of the walkers in the MCMC sampler. The walkers are free to explore beyond these ranges during the sampling process, constrained only by the prior distribution defined in the log_prior function.
#  np.random.rand(num_walkers) generates an array of num_walkers random numbers uniformly distributed in the half-open interval [0.0, 1.0).


print("[Step 8: Run MCMC sampling.]")
num_steps = 10200
sampler.run_mcmc(initial_positions, num_steps, progress=True)

print("\n[Step 9: Get the chain and discard burn-in.]")
chain = sampler.get_chain(discard=200, flat=True) # combining the chains from all walkers into a single chain.

print("[Step 10: Plot 2D posterior distributions of parameters.]")
# labels = ["Total Decays", "Half-life", "Background"]
labels = ["N", "T", "B"]
fig, axes = plt.subplots(len(labels), len(labels), figsize=(30, 30))
fig = corner.corner(chain, labels=labels, fig=fig, 
                    quantiles=[0.16, 0.5, 0.84], 
                    color='dodgerblue', 
                    use_math_text=True,
                    hist_bin_factor=2,
                    show_titles=True, 
                    title_fmt=".2f", 
                    title_kwargs={"fontsize": 60}, 
                    label_kwargs={"fontsize": 60},
                    hist_kwargs={"linewidth": 4},
                    #smooth=1.4
                    )

fig.subplots_adjust(left=0.12, bottom=0.12, right=0.97, top=0.95)

# Adjusting tick label sizes
for ax in fig.axes:
    ax.tick_params(axis='both', labelsize=50)  # Adjust tick label size as needed

fig.savefig("Exp_corner_plot.png")


print("[Step 11: Plot transparent uncertainty band predictions with calibrated parameters.]")
fig, ax_post_predict = plt.subplots(figsize=(36, 13))
fig.subplots_adjust(left=0.08, bottom=0.18, right=0.98, top=0.96)
    
p1 = ax_post_predict.errorbar(data_x_values_peakrange, data_y_values_peakrange, yerr=[data_y_varlow_peakrange,data_y_varhigh_peakrange], fmt='s', color='black', linewidth=2, markersize=2, label='Data', ecolor='black', zorder=1)  # zorder 2 appears on top of the zorder = 1.

y_values_for_each_params = [exponential_model(params, data_x_values_peakrange) for params in chain] # predictions of our calibrated model at 1000 x points for each set of parameters in the chain
y_values_for_each_params = np.array(y_values_for_each_params)  # Convert list to numpy array
print(y_values_for_each_params.shape) # (total number of samples, number of x points)
# Calculate the median, upper and lower percentiles
posterior_y_upper = np.percentile(y_values_for_each_params, 97.7, axis=0)
posterior_y_lower = np.percentile(y_values_for_each_params, 2.3, axis=0)
posterior_y_median = np.percentile(y_values_for_each_params, 50, axis=0)

# Plot the median, upper and lower percentiles with band
p3 = ax_post_predict.fill_between(data_x_values_peakrange, posterior_y_lower, posterior_y_upper, color='deepskyblue', alpha=0.4, linewidth=0, zorder=2)
p2 = ax_post_predict.plot(data_x_values_peakrange, posterior_y_median, color='dodgerblue', alpha=1.0, linewidth=3, zorder=2)
ax_post_predict.tick_params(axis='both', which='major', labelsize=60, length=9, width=2)
# ax.tick_params(direction='in')
ax_post_predict.set_xlabel("Time difference LEGe - MSD26 (ns)", fontsize=60, labelpad=20)
ax_post_predict.set_ylabel("Counts per 1 ns", fontsize=60, labelpad=11)
ax_post_predict.legend(['95% Credible Interval', 'Prediction Median', 'Data'], fontsize=60, loc='upper right')
xmin = min(data_x_values_peakrange) - 0.5
xmax = max(data_x_values_peakrange) + 0.5
ax_post_predict.set_xlim(xmin, xmax)
ymax = max(data_y_values_peakrange) + max(data_y_varhigh_peakrange) * 1.3
ax_post_predict.set_ylim(0, ymax)
ax_post_predict.set_ylim(2, ymax*1.5)
ax_post_predict.set_yscale('log')

# Adjust the width of the frame
for spine in ax_prior.spines.values():
    spine.set_linewidth(3)  # Set the linewidth to make the frame wider

plt.savefig("Exp_fit_prediction.png")


print("[The End]")
