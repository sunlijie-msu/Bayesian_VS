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


# Constants
Jr = 1.5  # 31S 6390 spin 3/2+
Jp = 0.5  # proton spin 1/2
JT = 1  # 30P gs spin 1+
hbar = 6.582119514e-1  # eV*fs
Bp_mean = 2.5e-4
Bp_positive_sigma = 0.4e-4
Bp_negative_sigma = 0.3e-4

print("[Step 1: Read Tau sample file.]")
Tau = np.loadtxt(r'D:\X\out\Bayesian_VS\Bayesian_DSL\DSL_31S4156_5fs\31S4156_samples_160k.dat')

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
percentiles_Tau= np.percentile(np.sort(Tau), [16, 50, 84, 90])
percentiles_OmegaGamma = np.percentile(np.sort(OmegaGamma), [16, 50, 84, 90])

print("16% Tau:", percentiles_Tau[0])
print("50% Tau:", percentiles_Tau[1])
print("84% Tau:", percentiles_Tau[2])
print("90% Tau:", percentiles_Tau[3])
      
print("16% OmegaGamma:", percentiles_OmegaGamma[0])
print("50% OmegaGamma:", percentiles_OmegaGamma[1])
print("84% OmegaGamma:", percentiles_OmegaGamma[2])
print("90% OmegaGamma:", percentiles_OmegaGamma[3])

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
plt.subplots_adjust(left=0.10, bottom=0.20, right=0.97, top=0.95)
# plt.hist(OmegaGamma, bins=500, range=(0, 100), color='blue', alpha=0.5)
# Increase the number of bins for the histogram
bins = np.linspace(0, 70, 280)
# Plot histogram with transparency and thicker frame line
sns.histplot(OmegaGamma, bins=bins, color='blue', stat='density', alpha=0.3, linewidth=0)

# Plot vertical lines at 16th and 84th percentiles
plt.axvline(x=percentiles_OmegaGamma[0], color='#55cc55', linestyle='--', linewidth=3, label='16%')
plt.axvline(x=percentiles_OmegaGamma[1], color='red', linestyle='--', linewidth=3, label='50%')
plt.axvline(x=percentiles_OmegaGamma[2], color='#55cc55', linestyle='--', linewidth=3, label='84%')

plt.xlabel('$\omega\gamma$ ($\mathrm{\mu}$eV)', labelpad=25)
plt.ylabel('Probability density', labelpad=35)
plt.xlim(0, 70)
# Set ticks to be visible and outside the axes
plt.tick_params(axis='both', which='major', direction='out', length=9, width=2)

plt.savefig('Fig_DSL2_Resonance_Strength.png')



print("[Step 3: Plot the 2D joint plot of Tau and OmegaGamma.]")

df = pd.DataFrame({'Tau': Tau, 'OmegaGamma': OmegaGamma})
print(df.head())

df_filtered = df[(df['OmegaGamma'] <= 70) & (df['Tau'] <= 20)]

g = sns.jointplot(
    data=df_filtered, x='Tau', y='OmegaGamma', kind='kde',
    fill=True, color='blue', alpha=0.8,
    xlim=(0, 20), ylim=(0, 70),
    marginal_kws=dict(fill=True, color='blue'),
    joint_kws=dict(fill=True, levels=40, color='blue'),
    height=24, space=0
)

# Plot lines at 16th and 84th percentiles
plt.axvline(x=percentiles_Tau[0], color='#55cc55', linestyle='--', linewidth=2, label='16%')
plt.axvline(x=percentiles_Tau[1], color='red', linestyle='--', linewidth=2, label='50%')
plt.axvline(x=percentiles_Tau[2], color='#55cc55', linestyle='--', linewidth=2, label='84%')
plt.axhline(y=percentiles_OmegaGamma[0], color='#55cc55', linestyle='--', linewidth=2, label='16%')
plt.axhline(y=percentiles_OmegaGamma[1], color='red', linestyle='--', linewidth=2, label='50%')
plt.axhline(y=percentiles_OmegaGamma[2], color='#55cc55', linestyle='--', linewidth=2, label='84%')

g.set_axis_labels(xlabel='$\\tau$ (fs)', ylabel='$\omega\gamma$ ($\mathrm{\mu}$eV)', fontsize=80, labelpad=45)
# Set tick labels font size
tick_fontsize = 70
g.ax_joint.set_xticks(g.ax_joint.get_xticks())
g.ax_joint.set_yticks(g.ax_joint.get_yticks())
g.ax_joint.tick_params(axis='both', which='major', length=9, width=2, labelsize=tick_fontsize)
g.fig.subplots_adjust(top=0.96, bottom=0.14, left=0.14, right=0.96)

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
T9_100 = np.linspace(0.1, 0.4, 4)
T9 = np.repeat(T9_100, num_samples // 4)


Er = np.random.normal(loc=Er_mean, scale=Er_sigma, size=num_samples)
Er = np.array(Er)
print("Er0: ", Er[0], "MeV")

ReactionRate = 1.5394e11 * (mu * T9) ** (-3 / 2) * OmegaGamma * 1e-9 * np.exp(-11.605 * Er / T9)

# Print some random sample values of T9 and ReactionRate
sample_indices = np.random.choice(len(ReactionRate), 5, replace=False)  # Get 5 random indices
print("Random samples of T9 and ReactionRate:")
for index in sample_indices:
    print(f"T9: {T9[index]}, ReactionRate: {ReactionRate[index]}")

t9_range = [0.1, 0.4000000001]
reaction_rate_range = [1e-7, 1.1e2]

# Create a 2D histogram of T9 and ReactionRate with the specified ranges
heatmap, xedges, yedges = np.histogram2d(T9, ReactionRate, bins=[4, 100], range=[t9_range, reaction_rate_range])


# df = pd.DataFrame({'T9': T9, 'ReactionRate': ReactionRate})
# print(df.head())

# df_filtered = df[(df['ReactionRate'] >= 1e-9)]


# sns.set_style("white", {'axes.linewidth': 3.0})
fig = plt.figure(figsize=(26, 24))
fig.subplots_adjust(top=0.96, bottom=0.14, left=0.18, right=0.95)

# Plot the heatmap using contourf
plt.contourf(xedges[:-1], yedges[:-1], heatmap.T, cmap='hot', levels=50)

# sns.kdeplot(data=df_filtered, x='T9', y='ReactionRate', fill=True, color='blue', alpha=0.8, levels=40, bw_adjust=0.2, thresh=0.05)

# sns.scatterplot(data=df_filtered, x='T9', y='ReactionRate', color='blue', alpha=0.3, s=3, linewidth=0, edgecolor='none')

plt.xlim(0.1, 0.4)  # Set x-axis limits
plt.ylim(1e-9, 1.1e2)  # Set y-axis limits
plt.xticks(fontsize=70, fontfamily="Times New Roman")
plt.yticks(fontsize=70, fontfamily="Times New Roman")
plt.tick_params(axis='both', which='major', direction='out', length=16, width=2)
plt.xlabel("Temperature (GK)", fontsize=80, labelpad=45, fontfamily="Times New Roman")
plt.ylabel("Reaction Rate (cm$^3$ s$^{-1}$ mol$^{-1}$)", fontsize=80, labelpad=40, fontfamily="Times New Roman")
plt.yscale('log')

plt.axhline(y=0.006672004, color='#55cc55', linestyle='--', linewidth=3, label='r$^{30}$P Decay')
plt.savefig('Fig_DSL2_Reaction_Rate.png')


# g = sns.jointplot(
#     data=df_filtered, x='Tau', y='OmegaGamma', kind='kde',
#     fill=True, color='blue', alpha=0.8,
#     xlim=(0, 20), ylim=(0, 70),
#     marginal_kws=dict(fill=True, color='blue'),
#     joint_kws=dict(fill=True, levels=40, color='blue'),
#     height=24, space=0
# )

print("[The End]")
