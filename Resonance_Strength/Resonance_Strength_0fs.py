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
Tau = np.loadtxt(r'D:\X\out\Bayesian_VS\Bayesian_DSL\png_0fs_scaled_1k\31S4156_0fs_samples.dat')

print("Tau0: ", Tau[0], "fs")

# Number of samples
num_samples = len(Tau)
print("Number of samples: ", num_samples)

Bp = np.random.normal(loc=Bp_mean, scale=Bp_positive_sigma, size=num_samples)

# Convert lists to arrays
Bp = np.array(Bp)
Tau = np.array(Tau)

# Calculate resonance strength for each sample
OmegaGamma = (2 * Jr + 1) / (2 * Jp + 1) / (2 * JT + 1) * Bp * (1 - Bp) * hbar / Tau *1e6 # Central value: 21.935 ueV if Tau = 5 fs; 21.675 ueV if Tau = 5.06 fs

# Calculate percentiles
percentiles_Tau= np.percentile(np.sort(Tau), [16, 50, 84, 95])
percentiles_OmegaGamma = np.percentile(np.sort(OmegaGamma), [16, 50, 84, 5])

print("16% Tau:", percentiles_Tau[0])
print("50% Tau:", percentiles_Tau[1])
print("84% Tau:", percentiles_Tau[2])
print("95% Tau:", percentiles_Tau[3])
print(percentiles_Tau[1], "+", percentiles_Tau[2]-percentiles_Tau[1], "-", percentiles_Tau[1]-percentiles_Tau[0])

print("16% OmegaGamma:", percentiles_OmegaGamma[0])
print("50% OmegaGamma:", percentiles_OmegaGamma[1])
print("84% OmegaGamma:", percentiles_OmegaGamma[2])
print("5% OmegaGamma:", percentiles_OmegaGamma[3])
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
bins = np.linspace(0, 1000, 5000)
# Plot histogram with transparency and thicker frame line
sns.histplot(OmegaGamma, bins=bins, color='blue', stat='density', alpha=0.3, linewidth=0)

# Plot vertical lines at 16th and 84th percentiles
# plt.axvline(x=percentiles_OmegaGamma[0], color='#55cc55', linestyle='--', linewidth=3, label='16%')
# plt.axvline(x=percentiles_OmegaGamma[1], color='red', linestyle='--', linewidth=3, label='50%')
# plt.axvline(x=percentiles_OmegaGamma[2], color='#55cc55', linestyle='--', linewidth=3, label='84%')
plt.axvline(x=percentiles_OmegaGamma[3], color='red', linestyle='--', linewidth=3, label='95%')

plt.xlabel('$\omega\gamma$ ($\mathrm{\mu}$eV)', labelpad=25)
plt.ylabel('Probability density', labelpad=35)
plt.xlim(0, 1000)
# Set ticks to be visible and outside the axes
plt.tick_params(axis='both', which='major', direction='out', length=9, width=2)

plt.savefig('Fig_DSL2_Resonance_Strength.png')



print("[Step 3: Plot the 2D joint plot of Tau and OmegaGamma.]")

df = pd.DataFrame({'Tau': Tau, 'OmegaGamma': OmegaGamma})
print(df.head(10))

df_filtered = df[(df['OmegaGamma'] <= 1200) & (df['Tau'] <= 6)]

g = sns.jointplot(
    data=df_filtered, x='Tau', y='OmegaGamma', kind='kde',
    fill=True, color='blue', alpha=0.8,
    xlim=(0, 6), ylim=(0, 1000),
    marginal_kws=dict(fill=True, color='blue'),
    joint_kws=dict(fill=True, levels=40, color='blue'),
    height=24, space=0
)

# Plot lines at 16th and 84th percentiles
# plt.axvline(x=percentiles_Tau[0], color='#55cc55', linestyle='--', linewidth=3, label='16%')
# plt.axvline(x=percentiles_Tau[1], color='red', linestyle='--', linewidth=3, label='50%')
# plt.axvline(x=percentiles_Tau[2], color='#55cc55', linestyle='--', linewidth=3, label='84%')
plt.axvline(x=percentiles_Tau[3], color='red', linestyle='--', linewidth=3, label='95%')

# plt.axhline(y=percentiles_OmegaGamma[0], color='#55cc55', linestyle='--', linewidth=3, label='16%')
# plt.axhline(y=percentiles_OmegaGamma[1], color='red', linestyle='--', linewidth=3, label='50%')
# plt.axhline(y=percentiles_OmegaGamma[2], color='#55cc55', linestyle='--', linewidth=3, label='84%')
plt.axhline(y=percentiles_OmegaGamma[3], color='red', linestyle='--', linewidth=3, label='95%')
g.fig.subplots_adjust(top=0.96, bottom=0.16, left=0.18, right=0.96)
g.set_axis_labels(xlabel='$\\tau$ (fs)', ylabel='$\omega\gamma$ ($\mathrm{\mu}$eV)', fontsize=80, labelpad=40)
# Set tick labels font size
tick_fontsize = 70
g.ax_joint.set_xticks(g.ax_joint.get_xticks())
g.ax_joint.set_yticks(g.ax_joint.get_yticks())
g.ax_joint.tick_params(axis='both', which='major', length=9, width=2, labelsize=tick_fontsize)

plt.savefig('Fig_DSL2_Tau_Strength_JointPlot.png')


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

# Calculate percentiles from 2% to 99%, every 2%
for i in range(2, 99, 2):
    percentile_ranges[i] = []

# Calculate ReactionRate percentiles for each T9 value
for T9 in T9_values:
    ReactionRate = 1.5394e11 * (mu * T9) ** (-3 / 2) * OmegaGamma * 1e-9 * np.exp(-11.605 * Er / T9)
    for i in range(2, 99, 2):
        percentile_ranges[i].append(np.percentile(ReactionRate, i))


fig, ax = plt.subplots(figsize=(25, 24))
fig.subplots_adjust(top=0.95, bottom=0.16, left=0.20, right=0.95)

# Define the color gradient and alpha levels
num_bands = 49  # From 2 to 98 with steps of 2 gives us 49 bands

# colors = sns.color_palette("Blues", num_bands)  # Get a list of blues shades

# Create filled bands with varying shades of blue
# for i in range(1, num_bands + 1):
#     low_percentile = 2 * i
#     high_percentile = 100 - low_percentile
#     if high_percentile > low_percentile:
#         plt.fill_between(
#             T9_values,
#             percentile_ranges[low_percentile],
#             percentile_ranges[high_percentile],
#             color=colors[i-1],
#             alpha=1.0  # You can adjust the alpha for overall transparency if you like
#         )



base_blue = "#1f77b4"

# Create filled bands with varying transparency
for i in range(1, num_bands + 1):
    low_percentile = 2 * i
    high_percentile = 100 - low_percentile
    if high_percentile > low_percentile:
        alpha_value = 0.3 * i / num_bands  # Increase transparency with a higher factor
        plt.fill_between(
            T9_values,
            percentile_ranges[low_percentile],
            percentile_ranges[high_percentile],
            color=base_blue,
            alpha=alpha_value
        )

# Assuming percentile_ranges contains all the percentiles from 2 to 98
# plt.plot(T9_values, percentile_ranges[50], label='$^{30}$P$(p,\\gamma)^{31}$S Median', color='blue', linewidth=1)
# plt.plot(T9_values, percentile_ranges[16], label='68% Uncertainty', color='#55cc55', linewidth=3, linestyle='--')
# plt.plot(T9_values, percentile_ranges[84], color='#55cc55', linewidth=3, linestyle='--')
plt.plot(T9_values, percentile_ranges[10], label='$^{30}$P$(p,\gamma)^{31}$S 95% Lower Limit', color='blue', linewidth=3, linestyle='--')

plt.xlim(0.1, 0.4)  # Set x-axis limits
plt.ylim(1e-9, 1e3)  # Set y-axis limits
plt.xticks(fontsize=80, fontfamily="Times New Roman")
plt.yticks(fontsize=80, fontfamily="Times New Roman")
plt.tick_params(axis='both', which='major', direction='out', length=16, width=2)
plt.xlabel("Temperature (GK)", fontsize=90, labelpad=45, fontfamily="Times New Roman")
plt.ylabel("Reaction Rate (cm$^3$ s$^{-1}$ mol$^{-1}$)", fontsize=90, labelpad=40, fontfamily="Times New Roman")
plt.yscale('log')
plt.axhline(y=0.006672004, color='#55cc55', linestyle='--', linewidth=3, label='$^{30}$P$(\\beta^+)^{30}$Si')

ax.legend(loc='lower right', fontsize=70)
plt.savefig('Fig_DSL2_Reaction_Rate.png')


print("[The End]")
