import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.lines as mlines
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


# Constants
Jr = 1.5  # 31S 6390 spin 3/2+
Jp = 0.5  # proton spin 1/2
JT = 1  # 30P gs spin 1+
hbar = 6.582119514e-1  # eV*fs
Bp_mean = 2.5e-4
Bp_positive_sigma = 0.4e-4
Bp_negative_sigma = 0.3e-4

print("[Step 1: Read Tau sample file.]")

fakeTau = '_3fs' # modify

file_path = f'D:\\X\\out\\Bayesian_VS\\Bayesian_DSL\\png{fakeTau}_scaled_0.5k\\31S4156{fakeTau}_samples.dat'

print("Load MCMC samples from file:", file_path)

Tau = np.loadtxt(file_path)

print("Tau_0: ", Tau[0], "fs")

# Number of samples
num_samples = len(Tau)
print("Number of samples: ", num_samples)

# fakeTau2 = '_0fs' # modify

# file_path2 = f'D:\\X\\out\\Bayesian_VS\\Bayesian_DSL\\png{fakeTau2}_scaled_1k\\31S4156{fakeTau2}_samples.dat'

# print("Load MCMC samples from file:", file_path2)

# Tau2 = np.loadtxt(file_path2)

# print("Tau2_0: ", Tau2[0], "fs")

# Number of samples
# num_samples2 = len(Tau2)
# print("Number of samples: ", num_samples2)

Bp = np.random.normal(loc=Bp_mean, scale=Bp_positive_sigma, size=num_samples)

# Convert lists to arrays
Bp = np.array(Bp)
Tau = np.array(Tau)
# Tau2 = np.array(Tau2)

# Calculate resonance strength for each sample
OmegaGamma = (2 * Jr + 1) / (2 * Jp + 1) / (2 * JT + 1) * Bp * (1 - Bp) * hbar / Tau *1e6 # Central value: 21.935 ueV if Tau = 5 fs; 21.675 ueV if Tau = 5.06 fs
# OmegaGamma2 = (2 * Jr + 1) / (2 * Jp + 1) / (2 * JT + 1) * Bp * (1 - Bp) * hbar / Tau2 *1e6 # Central value: 21.935 ueV if Tau = 5 fs; 21.675 ueV if Tau = 5.06 fs

# Define the path where you want to save the file
output_file_path = f'D:\\X\\out\\Bayesian_VS\\Bayesian_DSL\\png{fakeTau}_scaled_0.5k\\31S4156{fakeTau}_OmegaGamma.txt'

# Use numpy.savetxt to write the OmegaGamma array to a text file
np.savetxt(output_file_path, OmegaGamma, fmt='%f')  # '%f' specifies the format as floating point

print("OmegaGamma values saved to:", output_file_path)

# Calculate percentiles
percentiles_Tau = np.percentile(np.sort(Tau), [16, 50, 84, 90, 95])
# percentiles_Tau2 = np.percentile(np.sort(Tau2), [16, 50, 84, 90, 95])
percentiles_OmegaGamma = np.percentile(np.sort(OmegaGamma), [16, 50, 84, 10, 5])
# percentiles_OmegaGamma2 = np.percentile(np.sort(OmegaGamma2), [16, 50, 84, 10, 5])

print("16% Tau:", percentiles_Tau[0])
print("50% Tau:", percentiles_Tau[1])
print("84% Tau:", percentiles_Tau[2])
print("90% Tau:", percentiles_Tau[3])
print("95% Tau:", percentiles_Tau[4])
print(percentiles_Tau[1], "+", percentiles_Tau[2]-percentiles_Tau[1], "-", percentiles_Tau[1]-percentiles_Tau[0])

print("16% OmegaGamma:", percentiles_OmegaGamma[0])
print("50% OmegaGamma:", percentiles_OmegaGamma[1])
print("84% OmegaGamma:", percentiles_OmegaGamma[2])
print("10% OmegaGamma:", percentiles_OmegaGamma[3])
print("5% OmegaGamma:", percentiles_OmegaGamma[4])
print(percentiles_OmegaGamma[1], "+", percentiles_OmegaGamma[2]-percentiles_OmegaGamma[1], "-", percentiles_OmegaGamma[1]-percentiles_OmegaGamma[0])

plt.rcParams['axes.linewidth'] = 3.0
plt.rcParams['font.size'] = 60
font_family_options = ['Times New Roman', 'Georgia', 'Cambria', 'Courier New', 'serif']
plt.rcParams['font.family'] = font_family_options
# plt.rcParams['mathtext.default'] = 'regular'
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Times New Roman'
plt.rcParams['mathtext.it'] = 'Times New Roman:italic'

print("[Step 2: Plot histogram of Resonance Strength.]")

plt.figure(figsize=(36, 12))
plt.subplots_adjust(left=0.12, bottom=0.20, right=0.97, top=0.95)
# plt.hist(OmegaGamma, bins=500, range=(0, 100), color='blue', alpha=0.5)
# Increase the number of bins for the histogram

if fakeTau == '_3fs':
    minOmegaGamma = 0
    maxOmegaGamma = 200
    numOmegaGamma = 1000
    minTau = 0
    maxTau = 12

if fakeTau == '_0fs':
    minOmegaGamma = 0
    maxOmegaGamma = 1000
    numOmegaGamma = 3000
    minTau = 0
    maxTau = 6

bins = np.linspace(minOmegaGamma, maxOmegaGamma, numOmegaGamma)

# Plot histogram with transparency and thicker frame line
sns.histplot(OmegaGamma, bins=bins, color='blue', stat='density', alpha=0.4, linewidth=0)

if fakeTau == '_3fs':
    plt.axvline(x=percentiles_OmegaGamma[0], color='#55cc55', linestyle='--', linewidth=3, label='16%')
    plt.axvline(x=percentiles_OmegaGamma[1], color='red', linestyle='--', linewidth=3, label='50%')
    plt.axvline(x=percentiles_OmegaGamma[2], color='#55cc55', linestyle='--', linewidth=3, label='84%')

if fakeTau == '_0fs':
    plt.axvline(x=percentiles_OmegaGamma[3], color='red', linestyle='--', linewidth=3, label='90%')

plt.xlabel('$\omega\gamma$ ($\mathrm{\mu}$eV)', labelpad=25)
plt.ylabel('Probability density', labelpad=35)
plt.xlim(minOmegaGamma, maxOmegaGamma)
# Set ticks to be visible and outside the axes
plt.tick_params(axis='both', which='major', direction='out', length=9, width=2)

plt.savefig(f'Fig_DSL2_Resonance_Strength_Lifetime{fakeTau}_0.5k.png')



print("[Step 3: Plot the 2D joint plot of Tau and OmegaGamma.]")

df = pd.DataFrame({'Tau': Tau, 'OmegaGamma': OmegaGamma})
print(df.head(10))
# df2 = pd.DataFrame({'Tau2': Tau2, 'OmegaGamma2': OmegaGamma2})
# print(df2.head(10))

Tau = np.nan_to_num(Tau)
OmegaGamma = np.nan_to_num(OmegaGamma)

df_filtered = df[(df['OmegaGamma'] <= maxOmegaGamma*30)] # Filter out the outliers, or the plot will be ugly.
# df = df.sort_values(by=['Tau', 'OmegaGamma'])

# g = sns.jointplot(
#     data=df_filtered, x='Tau', y='OmegaGamma', kind='kde',
#     fill=True, color='blue', alpha=0.8,
#     xlim=(minTau, maxTau), ylim=(minOmegaGamma, maxOmegaGamma),
#     marginal_kws=dict(fill=True, color='blue', bw_adjust=1),
#     joint_kws=dict(fill=True, levels=20, color='blue', bw_adjust=1),
#     height=24, space=0
# )

g = sns.jointplot(
    data=df, x='Tau', y='OmegaGamma', kind='hist',
    color='blue', alpha=1.0,
    xlim=(minTau, maxTau), ylim=(minOmegaGamma, maxOmegaGamma),
    joint_kws={'bins': (10000, 60000)},
    marginal_kws={'bins': 60000, 'color': 'blue', 'alpha': 0.4, 'linewidth': 0, 'edgecolor': 'none'},
    height=24, space=0
)

# g.ax_joint.set_yscale('log')
# g.ax_marg_y.set_yscale('log')
# g.ax_joint.set_ylim(bottom=1)  # Set the bottom to a small positive number
# g.ax_marg_y.set_ylim(bottom=1)  # Similarly for the marginal y-axis

# Redraw the joint plot axes after adjustments
# g.fig.canvas.draw()

# Clear the existing marginal plots
# g.ax_marg_x.clear()
# g.ax_marg_y.clear()

# Create new marginal plots with specific bin sizes and remove labels and ticks
# sns.histplot(df['Tau'], ax=g.ax_marg_x, bins=600, color='blue', alpha=0.4, linewidth=0, edgecolor='none')
# g.ax_marg_x.set_xlabel('')
# g.ax_marg_x.set_ylabel('')
# g.ax_marg_x.set_xticks([])
# g.ax_marg_x.set_yticks([])
# g.ax_marg_x.tick_params(axis='both', labelsize=0)

# sns.histplot(df['OmegaGamma'], ax=g.ax_marg_y, bins=1000, color='blue', alpha=0.4, linewidth=0, edgecolor='none', orientation='horizontal')

# g.ax_marg_y.set_xlabel('')
# g.ax_marg_y.set_ylabel('')
# g.ax_marg_y.set_xticks([])
# g.ax_marg_y.set_yticks([])
# g.ax_marg_y.tick_params(axis='both', labelsize=0)


g.fig.subplots_adjust(top=0.96, bottom=0.16, left=0.18, right=0.96)
g.set_axis_labels(xlabel='$\\tau$ (fs)', ylabel='$\omega\gamma$ ($\mathrm{\mu}$eV)', fontsize=80, labelpad=40)
# Set tick labels font size
g.ax_joint.set_xticks(g.ax_joint.get_xticks())
g.ax_joint.set_yticks(g.ax_joint.get_yticks())
g.ax_joint.tick_params(axis='both', which='major', length=9, width=2, labelsize=70)

if fakeTau == '_3fs':
    plt.axvline(x=percentiles_Tau[0], color='#55cc55', linestyle='--', linewidth=3, label='16%')
    plt.axvline(x=percentiles_Tau[1], color='red', linestyle='--', linewidth=3, label='50%')
    plt.axvline(x=percentiles_Tau[2], color='#55cc55', linestyle='--', linewidth=3, label='84%')
    plt.axhline(y=percentiles_OmegaGamma[0], color='#55cc55', linestyle='--', linewidth=3, label='16%')
    plt.axhline(y=percentiles_OmegaGamma[1], color='red', linestyle='--', linewidth=3, label='50%')
    plt.axhline(y=percentiles_OmegaGamma[2], color='#55cc55', linestyle='--', linewidth=3, label='84%')
    
if fakeTau == '_0fs':
    plt.axvline(x=percentiles_Tau[3], color='red', linestyle='--', linewidth=3, label='90%')
    plt.axhline(y=percentiles_OmegaGamma[3], color='red', linestyle='--', linewidth=3, label='90%')

plt.savefig(f'Fig_DSL2_Strength_JointPlot_Lifetime{fakeTau}_0.5k.png')


print("[Step 4: Plot Reaction Rate as a function of T9.]")

Ex_mean = 6390.46e-3  # MeV
Ex_sigma = 0.16e-3  # MeV
Sp_mean = 6130.65e-3  # MeV
Sp_sigma = 0.24e-3  # MeV

Er_mean = Ex_mean - Sp_mean
print("Er_mean: ", Er_mean*1e3, "keV")
Er_sigma = np.sqrt(Ex_sigma**2 + Sp_sigma**2)
print("Er_sigma: ", Er_sigma*1e3, "keV")

Ap = 1
AT = 30
mu = Ap * AT / (Ap + AT)


# T9 = np.random.uniform(low=0.1, high=0.4, size=num_samples)
T9_values = np.linspace(0.1, 0.4, 50)

Er = np.random.normal(loc=Er_mean, scale=Er_sigma, size=num_samples)
Er = np.array(Er)
print("Er0: ", Er[0], "MeV")

# Initialize lists to store percentile values for each T9
# percentiles_2 = []
# percentiles_16 = []
# percentiles_50 = []
# percentiles_84 = []
# percentiles_98 = []

# for T9 in T9_values:
#     ReactionRate = 1.5394e11 * (mu * T9) ** (-3 / 2) * OmegaGamma * 1e-9 * np.exp(-11.605 * Er / T9)
#     percentiles_2.append(np.percentile(ReactionRate, 2.3))
#     percentiles_16.append(np.percentile(ReactionRate, 16))
#     percentiles_50.append(np.percentile(ReactionRate, 50))
#     percentiles_84.append(np.percentile(ReactionRate, 84))
#     percentiles_98.append(np.percentile(ReactionRate, 97.7))


# df = pd.DataFrame({'T9': T9, 'ReactionRate': ReactionRate})
# print(df.head())

# df_filtered = df[(df['ReactionRate'] >= 1e-9)]


# sns.kdeplot(data=df_filtered, x='T9', y='ReactionRate', fill=True, color='blue', alpha=0.8, levels=40, bw_adjust=0.2, thresh=0.05)

# sns.scatterplot(data=df_filtered, x='T9', y='ReactionRate', color='blue', alpha=0.3, s=3, linewidth=0, edgecolor='none')

# Initialize a dictionary to store percentile values for each T9
percentile_ranges = {}
# percentile_ranges2 = {}

# Calculate percentiles from 2% to 99%, every 2%
for i in range(1, 100, 1):
    percentile_ranges[i] = []
    # percentile_ranges2[i] = []

# Calculate ReactionRate percentiles for each T9 value
for T9 in T9_values:
    ReactionRate = 1.5394e11 * (mu * T9) ** (-3 / 2) * OmegaGamma * 1e-9 * np.exp(-11.605 * Er / T9)
    for i in range(1, 100, 1):
        percentile_ranges[i].append(np.percentile(ReactionRate, i))

# for T9 in T9_values:
#     ReactionRate2 = 1.5394e11 * (mu * T9) ** (-3 / 2) * OmegaGamma2 * 1e-9 * np.exp(-11.605 * Er / T9)
#     for i in range(1, 100, 1):
#         percentile_ranges2[i].append(np.percentile(ReactionRate2, i))

fig, ax = plt.subplots(figsize=(25, 24))
fig.subplots_adjust(top=0.95, bottom=0.16, left=0.20, right=0.95)

# Define the color gradient and alpha levels
num_bands = 99  # From 1 to 99 with steps of 1

colors = sns.color_palette("Blues", num_bands)  # Get a list of blues shades

# Create filled bands with varying shades of blue
for i in range(1, num_bands + 1):
    low_percentile = i
    high_percentile = 100 - i
    if high_percentile > low_percentile:
        plt.fill_between(
            T9_values,
            percentile_ranges[low_percentile],
            percentile_ranges[high_percentile],
            color=colors[i-1],
            alpha=1.0  # You can adjust the alpha for overall transparency if you like
        )
        

# base_blue = "#1f77b4"
base_blue = "#0000ff"

# Create filled bands with varying transparency
# for i in range(1, num_bands + 1):
#     low_percentile = i
#     high_percentile = 100 - i
#     if high_percentile > low_percentile:
#         alpha_value = 0.2 * i / num_bands  # Increase transparency with a higher factor
#         plt.fill_between(
#             T9_values,
#             percentile_ranges[low_percentile],
#             percentile_ranges[high_percentile],
#             color=base_blue,
#             alpha=alpha_value
#         )

# Assuming percentile_ranges contains all the percentiles from 2 to 98

if fakeTau == '_3fs':
    plt.plot(T9_values, percentile_ranges[50], label='$^{30}$P$(p,\\gamma)^{31}$S  Median', color='blue', linewidth=3, linestyle='--')
    # plt.plot(T9_values, percentile_ranges[16], label='$^{30}$P$(p,\\gamma)^{31}$S  68% Uncertainty', color='#55cc55', linewidth=3, linestyle='--')
    # plt.plot(T9_values, percentile_ranges[84], color='#55cc55', linewidth=3, linestyle='--')
    # plt.plot(T9_values, percentile_ranges2[10], label='$^{30}$P$(p,\\gamma)^{31}$S  90% Lower Limit', color='#55cc55', linewidth=3, linestyle='--')

if fakeTau == '_0fs':
    plt.plot(T9_values, percentile_ranges[10], label='$^{30}$P$(p,\gamma)^{31}$S  90% Lower Limit', color='blue', linewidth=3, linestyle='--')

plt.xlim(0.1, 0.4)  # Set x-axis limits
plt.ylim(1e-9, 1e3)  # Set y-axis limits
plt.xticks(fontsize=80, fontfamily="Times New Roman")
plt.yticks(fontsize=80, fontfamily="Times New Roman")
plt.tick_params(axis='both', which='major', direction='out', length=16, width=2)
plt.xlabel("Temperature (GK)", fontsize=90, labelpad=45, fontfamily="Times New Roman")
plt.ylabel("Reaction Rate (cm$^3$ s$^{-1}$ mol$^{-1}$)", fontsize=90, labelpad=40, fontfamily="Times New Roman")
plt.yscale('log')
plt.axhline(y=0.006672004, color='red', linestyle='--', linewidth=3, label='$^{30}$P$(\\beta^+)^{30}$Si')

ax.legend(loc='lower right', fontsize=70)
plt.savefig(f'Fig_DSL2_Reaction_Rate_Lifetime{fakeTau}_0.5k.png')


print("[The End]")
