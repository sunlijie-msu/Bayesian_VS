import numpy as np
import matplotlib.pyplot as plt
import emcee
import corner
import seaborn as sns
import pandas as pd
import scipy.stats as sps
import os

print("[Bayesian Fit of Exponential Decay Model to PXCT Lifetime Data]\n")
print("[Step 0: Set up paths.]")
project_path = os.path.abspath("Bayesian_Fit_Exp.py")
dir_path = os.path.dirname(project_path)
parameter_output_txt_name = dir_path + "\Parameters_percentiles.txt"
offset = 1500
Which_Dataset = 'timing_msdtotal_e' # Modify run 0079-0091
# Which_Dataset = 'timing_msd26_e' # run 0091-0095, 0100-0107
Which_MSD = 26 # Modify: 12 for MSD12; 26 for MSD26

# The run_analysis() function now encapsulates the main code logic, taking bin_start and bin_stop as arguments.
def run_analysis(fit_start, fit_stop, Ea_gate):
    
    print("[Step 1: Read data from CSV input files and plot data.]")
    # Which_Dataset = 'timing_msdtotal_e' # Modify run 0079-0091
    # Which_Dataset = 'timing_msd26_e' # run 0091-0095, 0100-0107

    if Which_Dataset == 'timing_msdtotal_e':
        # Which_MSD = 26 # Modify: 12 for MSD12; 26 for MSD26
        Ea_central = 5421 # 5421 for MSDtotal, based on LISE++ calculation
    
        # if Which_MSD == 12:
            # bin_start = 1500 + 240 # don't change the 1500, which is an offset
    
        # if Which_MSD == 26:
            # bin_start = 1500 + 160 # don't change the 1500, which is an offset
            # bin 0 is the first; bin 1650 is the 1651st = 150.5 ns;  bin 2040 is the 2041st = 540 ns
    
        # Ea_gate = 60 # Modify: 3 means +/-3 keV = 6 keV; 20 means +/-20 keV = 40 keV



    if Which_Dataset == 'timing_msd26_e':
        # Which_MSD = 26 # 26 for MSD26
        Ea_central = 5479 # 5479 for MSD26; based on LISE++ calculation
        # Ea_gate = 20 # 3; 3 means +/-3 keV = 6 keV; 20 means +/-20 keV = 40 keV
    

    # bin_stop = 1500 + 1400 # don't change the 1500, which is an offset
    # bin 2920 is the 2921st = 1420.5 ns (not included), so the last included bin is 1420.5 - 1 = 1419.5 ns

    Ea_start = Ea_central - Ea_gate
    Ea_stop = Ea_central + Ea_gate

    filename = "\PXCT_237Np_" + Which_Dataset + '_Eastart' + str(Ea_start) + '_Eastop' + str(Ea_stop) + '_MSD' + str(Which_MSD) + '_Fitstart' + str(fit_start - offset) + '_Fitstop' + str(fit_stop - offset) + '_'
    print(dir_path + filename)
    corner_png_name = dir_path + filename + "Posterior2D.png"
    prediction_png_name = dir_path + filename + "Prediction.png"


    # Load data fullrange csv files and model fullrange csv files and model parameter values csv files
    print('F:/e21010/pxct/' + Which_Dataset + '_' + str(Ea_start) + '_' + str(Ea_stop) + '_msd' + str(Which_MSD) + '_t_bin1ns.csv')
    data_fullrange = np.loadtxt('F:/e21010/pxct/' + Which_Dataset + '_' + str(Ea_start) + '_' + str(Ea_stop) + '_msd' + str(Which_MSD) + '_t_bin1ns.csv', delimiter=',')  # Read full range data # csv derived from histogram

    data_peakrange = data_fullrange[fit_start:fit_stop, :] # Select data in the peak range by rows
    data_x_values_peakrange = data_peakrange[:, 0]
    data_y_values_peakrange = data_peakrange[:, 1]
    data_y_varlow_peakrange = data_peakrange[:, 2]
    data_y_varhigh_peakrange = data_peakrange[:, 3]

    # Set values where data_y_varlow_peakrange is 0 to 1.841 (for empty bins)
    data_y_varlow_peakrange[data_y_varlow_peakrange == 0] = 1.841

    data_y_errors_peakrange = (data_y_varlow_peakrange + data_y_varhigh_peakrange) / 2
    data_x_errors_peakrange = 0.5 # bin size is 1 ns

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

    fig, ax_prior = plt.subplots(figsize=(24, 15))
    fig.subplots_adjust(left=0.14, bottom=0.16, right=0.96, top=0.96)

    ax_prior.errorbar(data_x_values_peakrange, data_y_values_peakrange, yerr=[data_y_varlow_peakrange,data_y_varhigh_peakrange], fmt='s', color='black', linewidth=2, markersize=2, label='Data', ecolor='black', zorder=2)  # zorder 2 appears on top of the zorder = 1.
    ax_prior.tick_params(axis='both', which='major', labelsize=60, length=9, width=2)
    # ax.tick_params(direction='in')
    ax_prior.set_xlabel("Time difference LEGe - MSD" + str(Which_MSD) + " (ns)", fontsize=60, labelpad=27)
    ax_prior.set_ylabel("Counts per 1 ns", fontsize=60, labelpad=15)
    xmin = min(data_x_values_peakrange) - 0.5
    xmax = max(data_x_values_peakrange) + 0.5
    print("xmin: ", xmin,", xmax: ", xmax)
    ax_prior.set_xlim(xmin, xmax)
    ymax = max(data_y_values_peakrange) + max(data_y_varhigh_peakrange) * 1.5
    ax_prior.set_ylim(0, ymax)

    # Adjust the width of the frame
    for spine in ax_prior.spines.values():
        spine.set_linewidth(3)  # Set the linewidth to make the frame wider

    # fig.savefig('Exp_data_' + Which_Dataset + '_' + str(Which_MSD) + '_t.png')


    print("[Step 2: Define the exponential decay function (Model).]")
    def exponential_model(parameters, x_values):
        total_decays, half_life, background = parameters
        return total_decays / half_life * 0.693147 * np.exp(x_values / half_life * -0.693147) + background


    print("[Step 3: Define the log-likelihood function (Likelihood).]")
    def log_likelihood(parameters, x_values, y_values, x_errors, y_errors):
        total_decays, half_life, background = parameters
        model_values = exponential_model(parameters, x_values)
        # Errors are not used in the maximum likelihood function, but they are used in the chi-squared function
        likelihood = 0
        # Ensure that model values are positive to avoid log of zero or negative numbers
        model_values = np.maximum(model_values, 1e-9)
    
        for y_data, y_model in zip(y_values, model_values):
            if y_data > 0:
                # The Poisson log-likelihood component for observed counts. Same as ROOT ML fit method. Highly recommended!
                likelihood += y_model - y_data + y_data * np.log(y_data / y_model)
            else:
                # If observed count is zero, simplify the Poisson likelihood component because log(0) is -infinity
                likelihood += y_model - y_data
            
        return -likelihood  # Return the negative likelihood for minimization
    
        # model_values_1 = exponential_model(parameters, x_values + x_errors / 2)
        # model_values_2 = exponential_model(parameters, x_values - x_errors / 2)
        # total_error_squared = y_errors**2 + ((model_values_1 - model_values_2) / 2)**2
        # return -0.5 * np.sum((y_values - model_values)**2 / total_error_squared + np.log(total_error_squared)) # Similar to ROOT Chi2 fit method. Not recommended for low statistics data.


    print("[Step 4: Define the log-prior function (Prior).]")
    def log_prior(parameters):
        total_decays, half_life, background = parameters
        if 1.5e6 < total_decays < 7.9e6 and 57.0 < half_life < 80.0 and 0.0 < background < 16.0: # Modify the prior ranges based on previous ROOT fit results
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
    if Ea_stop - Ea_start == 20:
        N_guess = 2e6
        T_guess = 68.0
        B_guess = 3.0
    
    if Ea_stop - Ea_start == 40:
        N_guess = 4e6
        T_guess = 68.0
        B_guess = 5.0
    
    if Ea_stop - Ea_start == 60:
        N_guess = 5e6
        T_guess = 68.0
        B_guess = 7.0
        
    if Ea_stop - Ea_start == 80:
        N_guess = 6e6
        T_guess = 68.0
        B_guess = 8.0
        
    if Ea_stop - Ea_start == 100:
        N_guess = 6.6e6
        T_guess = 68.0
        B_guess = 9.0
        
    if Ea_stop - Ea_start == 120:
        N_guess = 7e6
        T_guess = 68.0
        B_guess = 9.4
    


    initial_positions = np.zeros((num_walkers, num_dimensions))
    initial_positions[:, 0] =  (0.9 + 0.2 * np.random.rand(num_walkers)) * N_guess    # initial slope between 1 and 10 # Modify the initial positions based on previous ROOT fit results
    initial_positions[:, 1] =  (0.9 + 0.2 * np.random.rand(num_walkers)) * T_guess  # initial intercept between -150 and 200
    initial_positions[:, 2] =  (0.9 + 0.2 * np.random.rand(num_walkers)) * B_guess  # initial intercept between -150 and 200
    # these ranges are just for the initial positions of the walkers in the MCMC sampler. The walkers are free to explore beyond these ranges during the sampling process, constrained only by the prior distribution defined in the log_prior function.
    #  np.random.rand(num_walkers) generates an array of num_walkers random numbers uniformly distributed in the half-open interval [0.0, 1.0).


    print("[Step 8: Run MCMC sampling.]")
    num_steps = 1200
    sampler.run_mcmc(initial_positions, num_steps, progress=True)

    print("\n[Step 9: Get the chain and discard burn-in.]")
    chain = sampler.get_chain(discard=200, flat=True) # combining the chains from all walkers into a single chain.

    print("[Step 10-1: Plot 2D posterior distributions of parameters.]\n")
    # labels = ["Total Decays", "Half-life", "Background"]
    labels = ["N", "T", "B"]
    fig, axes = plt.subplots(len(labels), len(labels), figsize=(27, 27))
    fig = corner.corner(chain, labels=labels, fig=fig, 
                        quantiles=[0.16, 0.5, 0.84], 
                        color='blue', 
                        use_math_text=True,
                        hist_bin_factor=2,
                        show_titles=True, 
                        title_fmt=".2f", 
                        title_kwargs={"fontsize": 60}, 
                        label_kwargs={"fontsize": 60},
                        hist_kwargs={"linewidth": 4},
                        #smooth=1.4
                        )

    fig.subplots_adjust(left=0.12, bottom=0.12, right=0.97, top=0.94)

    # Adjusting tick label sizes
    for ax in fig.axes:
        ax.tick_params(axis='both', labelsize=50)  # Adjust tick label size as needed

    fig.savefig(corner_png_name)
    print(f"Posterior 2D distributions saved to {corner_png_name}\n")

    print("[Step 10-2: Save the parameter quantiles to a text file.]\n")
    # Calculate the quantiles for each parameter
    quantiles = np.percentile(chain, [16, 50, 84], axis=0)

    # Open a text file to write the quantiles
    with open(parameter_output_txt_name, 'a') as file:
        # Initialize an empty string to accumulate parameter details
        parameters_info = ""
        for i, label in enumerate(labels):
            parameters_info += f"{label}\t{quantiles[0][i]:.4f}\t{quantiles[1][i]:.4f}\t{quantiles[2][i]:.4f}\t"
    
        # Write the accumulated parameter details followed by other details in one row
        file.write(f"{parameters_info}\tPXCT_237Np\t{Which_Dataset}\tEastart\t{Ea_start}\tEastop\t{Ea_stop}\tMSD\t{Which_MSD}\tFitstart\t{(fit_start - offset)}\tFitstop\t{(fit_stop - offset)}\n")


    print(f"Quantiles saved to {parameter_output_txt_name}\n")


    print("[Step 11: Plot transparent uncertainty band predictions with calibrated parameters.]")
    fig, ax_post_predict = plt.subplots(figsize=(24, 15))
    fig.subplots_adjust(left=0.11, bottom=0.16, right=0.96, top=0.96)
    
    p1 = ax_post_predict.errorbar(data_x_values_peakrange, data_y_values_peakrange, yerr=[data_y_varlow_peakrange,data_y_varhigh_peakrange], fmt='s', color='black', linewidth=1, markersize=2, label='Data', ecolor='black', zorder=1)  # zorder 2 appears on top of the zorder = 1.

    y_values_for_each_params = [exponential_model(params, data_x_values_peakrange) for params in chain] # predictions of our calibrated model at 1000 x points for each set of parameters in the chain
    y_values_for_each_params = np.array(y_values_for_each_params)  # Convert list to numpy array
    print(y_values_for_each_params.shape) # (total number of samples, number of x points)
    # Calculate the median, upper and lower percentiles
    posterior_y_upper = np.percentile(y_values_for_each_params, 97.7, axis=0)
    posterior_y_lower = np.percentile(y_values_for_each_params, 2.3, axis=0)
    posterior_y_median = np.percentile(y_values_for_each_params, 50, axis=0)

    # Plot the median, upper and lower percentiles with band
    p3 = ax_post_predict.fill_between(data_x_values_peakrange, posterior_y_lower, posterior_y_upper, color='deepskyblue', alpha=0.4, linewidth=0, zorder=2)
    p2 = ax_post_predict.plot(data_x_values_peakrange, posterior_y_median, color='dodgerblue', alpha=1.0, linewidth=4, zorder=2)
    ax_post_predict.tick_params(axis='both', which='major', labelsize=60, length=9, width=2)
    # ax.tick_params(direction='in')
    ax_post_predict.set_xlabel("Time difference LEGe - MSD" + str(Which_MSD) + " (ns)", fontsize=60, labelpad=27)
    ax_post_predict.set_ylabel("Counts per 1 ns", fontsize=60, labelpad=12)
    ax_post_predict.legend(['95% Credible Interval', 'Prediction Median', 'Data'], fontsize=60, loc='upper right')
    xmin = min(data_x_values_peakrange) - 0.5
    xmax = max(data_x_values_peakrange) + 0.5
    ax_post_predict.set_xlim(xmin, xmax)
    ymax = max(data_y_values_peakrange) + max(data_y_varhigh_peakrange) * 1.3
    # ax_post_predict.set_ylim(0, ymax)
    ax_post_predict.set_ylim(0.1, ymax * 1.5)
    ax_post_predict.set_yscale('log')

    # Adjust the width of the frame
    for spine in ax_prior.spines.values():
        spine.set_linewidth(3)  # Set the linewidth to make the frame wider

    # plt.show()
    plt.savefig(prediction_png_name)
    print(f"\nPrediction plot saved to {prediction_png_name}")



# The main() function handles the loop, iterating through different bin_start and bin_stop values and calling run_analysis() for each combination.
def main():

    fit_start_values = np.array([160, 220, 280, 340, 400]) + offset
    fit_stop_values = np.array([1100, 1200, 1300, 1400]) + offset
    Ea_gate_values = np.array([10, 20, 30, 40, 50, 60])
    
    # Iterate through all combinations of parameters using nested loops
    for fitstart in fit_start_values:
        for fitstop in fit_stop_values:
            for Eagate in Ea_gate_values:
                run_analysis(fitstart, fitstop, Eagate)
            

if __name__ == "__main__":
    main()

print("[The End]")

