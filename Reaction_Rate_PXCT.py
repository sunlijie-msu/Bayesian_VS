import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.lines as mlines
import matplotlib.ticker as ticker
import matplotlib.cm as cm
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


# plots a list of profiles in the same figure. Each profile corresponds to a simulation replica for the given instance.
plt.rcParams['axes.linewidth'] = 2.0
plt.rcParams['font.size'] = 32
font_family_options = ['Times New Roman', 'Georgia', 'Cambria', 'Courier New', 'serif']
plt.rcParams['font.family'] = font_family_options
# plt.rcParams['mathtext.default'] = 'regular'
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Times New Roman'
plt.rcParams['mathtext.it'] = 'Times New Roman:italic'

# Inputs for 128 60Zn resonances from Brown or Rauscher
Jr = [1, 5, 2, 3, 4, 1, 4, 4, 0, 3, 2, 2, 5, 3, 3, 1, 4, 1, 2, 2, 4, 5, 0, 2, 4, 3, 4, 3, 2, 0, 1, 4, 5, 4, 2, 5, 3, 4, 2, 3, 2, 5, 5, 3, 1, 2, 0, 4, 4, 2, 1, 5, 3, 0, 4, 3, 2, 3, 1, 2, 3, 5, 4, 2, 5, 2, 1, 0, 5, 4, 1, 4, 2, 2, 3, 2, 3, 5, 1, 2, 3, 5, 4, 2, 1, 5, 2, 0, 2, 3, 3, 4, 5, 4, 3, 1, 4, 2, 2, 3, 4, 5, 1, 4, 2, 1, 3, 4, 3, 0, 0, 3, 4, 2, 2, 3, 4, 3, 5, 1, 4, 2, 3, 2, 4, 1, 3, 3]  # 60Zn resonance spins from Brown

Er = [0.012, 0.083, 0.121, 0.147, 0.155, 0.168, 0.223, 0.241, 0.245, 0.27, 0.392, 0.396, 0.405, 0.437, 0.452, 0.461, 0.488, 0.538, 0.54, 0.563, 0.603, 0.653, 0.689, 0.741, 0.811, 0.825, 0.845, 0.872, 0.884, 0.933, 0.943, 0.944, 0.95, 0.959, 0.967, 0.989, 1.001, 1.005, 1.006, 1.007, 1.023, 1.075, 1.081, 1.091, 1.092, 1.14, 1.148, 1.149, 1.164, 1.196, 1.2, 1.22, 1.246, 1.28, 1.285, 1.288, 1.289, 1.3, 1.315, 1.316, 1.324, 1.325, 1.353, 1.372, 1.388, 1.39, 1.405, 1.41, 1.45, 1.456, 1.481, 1.502, 1.504, 1.515, 1.521, 1.533, 1.537, 1.592, 1.599, 1.638, 1.639, 1.647, 1.657, 1.661, 1.668, 1.673, 1.674, 1.71, 1.717, 1.744, 1.753, 1.768, 1.794, 1.808, 1.81, 1.816, 1.822, 1.841, 1.862, 1.876, 1.885, 1.904, 1.906, 1.908, 1.931, 1.94, 1.955, 1.977, 1.989, 1.99, 2.01, 2.028, 2.03, 2.059, 2.074, 2.075, 2.092, 2.112, 2.114, 2.131, 2.133, 2.138, 2.142, 2.148, 2.156, 2.158, 2.163, 2.185] # 60Zn resonance energies in MeV from Brown

Gg = [5.20E-09, 6.20E-10, 6.50E-09, 9.70E-08, 5.30E-09, 8.50E-07, 1.10E-08, 1.60E-08, 1.20E-08, 2.80E-07, 2.70E-08, 3.80E-08, 2.40E-09, 3.90E-09, 2.70E-09, 6.40E-08, 4.60E-09, 1.50E-08, 1.90E-07, 1.40E-08, 4.00E-09, 2.60E-08, 4.20E-08, 2.20E-08, 1.60E-08, 1.90E-08, 5.50E-09, 3.90E-09, 3.30E-08, 8.10E-09, 2.00E-08, 9.60E-08, 1.70E-09, 7.20E-09, 2.50E-07, 2.30E-08, 1.80E-07, 1.00E-08, 2.80E-08, 1.20E-07, 1.50E-07, 2.80E-08, 7.80E-08, 5.20E-09, 5.00E-08, 3.50E-08, 2.70E-07, 3.60E-08, 2.50E-08, 1.80E-08, 2.00E-07, 5.30E-09, 2.00E-07, 4.90E-08, 2.10E-07, 1.10E-08, 1.90E-08, 3.10E-08, 2.30E-08, 2.10E-08, 1.30E-07, 2.40E-09, 1.10E-07, 9.40E-08, 4.30E-08, 2.50E-08, 3.60E-08, 7.40E-09, 2.60E-09, 8.60E-09, 1.70E-08, 9.80E-09, 1.60E-08, 5.50E-08, 2.60E-08, 3.50E-08, 1.50E-07, 1.10E-07, 1.90E-07, 1.60E-08, 1.00E-08, 6.70E-09, 5.00E-09, 1.30E-07, 2.70E-08, 5.70E-09, 2.40E-08, 3.40E-08, 3.90E-08, 4.80E-08, 2.10E-07, 1.10E-08, 1.20E-07, 3.00E-08, 2.50E-07, 2.20E-08, 1.10E-07, 1.90E-07, 2.80E-08, 3.20E-08, 1.50E-08, 2.40E-09, 1.50E-07, 1.50E-07, 2.60E-08, 1.80E-08, 2.70E-08, 1.80E-08, 2.00E-08, 2.90E-07, 5.20E-08, 2.70E-08, 1.90E-08, 1.40E-07, 6.20E-08, 9.60E-08, 1.40E-07, 2.50E-08, 7.60E-09, 8.20E-08, 1.70E-08, 2.30E-08, 1.60E-07, 1.90E-07, 1.40E-07, 1.80E-07, 4.50E-08, 1.90E-07] # 60Zn resonance gamma widths in MeV from Brown

Gp = [0.00E+00, 0.00E+00, 2.80E-31, 5.40E-28, 4.70E-30, 1.80E-25, 1.20E-24, 8.80E-25, 6.80E-22, 1.20E-20, 2.40E-17, 7.40E-16, 3.80E-20, 2.60E-15, 2.30E-15, 1.50E-13, 1.20E-17, 5.00E-13, 2.10E-12, 7.40E-13, 3.60E-14, 4.00E-16, 1.60E-12, 2.60E-11, 8.20E-13, 2.00E-11, 2.50E-12, 8.80E-12, 4.70E-09, 9.90E-10, 1.10E-09, 3.10E-11, 1.20E-12, 1.30E-12, 5.70E-08, 3.40E-13, 1.70E-09, 1.10E-12, 1.10E-08, 4.50E-11, 1.00E-08, 6.80E-14, 4.40E-12, 2.00E-09, 7.30E-08, 2.20E-08, 7.90E-10, 1.90E-10, 8.50E-13, 6.80E-08, 2.10E-07, 2.60E-13, 7.20E-09, 2.80E-08, 3.60E-10, 3.80E-08, 1.00E-07, 4.60E-08, 8.10E-08, 9.20E-08, 5.20E-08, 2.90E-11, 3.00E-10, 1.50E-07, 7.20E-11, 1.50E-07, 1.00E-07, 8.40E-08, 1.10E-11, 3.00E-11, 1.00E-07, 8.40E-10, 4.20E-08, 3.00E-08, 5.10E-08, 1.30E-07, 2.20E-07, 1.30E-10, 1.00E-06, 1.40E-06, 1.00E-08, 1.70E-10, 3.20E-09, 6.10E-07, 3.20E-06, 2.70E-10, 4.30E-07, 1.70E-07, 3.50E-07, 7.60E-08, 6.10E-07, 1.10E-08, 2.90E-09, 6.90E-09, 1.40E-07, 1.50E-06, 4.10E-08, 4.40E-06, 1.60E-06, 6.10E-07, 1.00E-08, 3.50E-10, 1.40E-05, 1.00E-07, 2.40E-07, 1.10E-05, 4.20E-06, 4.90E-09, 7.80E-07, 1.10E-07, 2.70E-06, 3.20E-07, 1.30E-08, 5.10E-06, 1.70E-05, 5.70E-06, 1.20E-07, 2.10E-07, 4.40E-09, 2.70E-05, 2.80E-09, 6.30E-06, 6.10E-07, 4.60E-05, 1.10E-07, 1.30E-04, 4.10E-06, 1.90E-06] # 60Zn resonance proton widths in MeV from Brown

Ga = [0, 0, 1.72457E-14, 0, 8.7643E-15, 0, 2.33934E-13, 4.52878E-13, 5.68255E-13, 0, 2.84047E-13, 2.93612E-13, 0, 0, 0, 0, 4.63546E-12, 0, 1.09104E-12, 1.32576E-12, 1.58264E-11, 2.95177E-63, 3.30387E-11, 5.65361E-12, 1.65694E-11, 2.48746E-47, 9.36026E-11, 8.76057E-45, 1.63906E-11, 2.03829E-10, 4.48509E-42, 1.85426E-10, 1.10013E-42, 3.44158E-11, 2.91729E-11, 9.26842E-41, 5.08285E-40, 1.19864E-11, 3.76376E-11, 2.43032E-39, 4.25846E-11, 1.71776E-37, 2.19179E-37, 1.21342E-36, 1.46155E-36, 9.00461E-11, 8.31747E-10, 7.69766E-10, 2.38437E-10, 1.26613E-10, 1.29171E-33, 1.86691E-33, 1.81258E-32, 1.84677E-09, 1.30086E-09, 2.13484E-31, 2.19952E-10, 2.82035E-31, 1.17421E-30, 2.56544E-10, 1.59165E-30, 4.11153E-31, 2.71743E-09, 3.51844E-10, 6.13991E-30, 3.88053E-10, 4.46534E-29, 3.8101E-09, 6.71084E-29, 2.93032E-10, 1.48024E-27, 5.97144E-09, 7.08598E-10, 7.53904E-10, 5.90373E-27, 8.28611E-10, 9.55272E-27, 1.52938E-26, 8.21575E-26, 1.39767E-09, 2.97249E-25, 8.69505E-26, 1.1563E-09, 1.56722E-09, 9.46576E-25, 2.39478E-25, 1.69035E-09, 1.79682E-08, 2.10992E-09, 6.89423E-24, 8.02665E-24, 7.8604E-09, 6.84255E-24, 3.01502E-09, 4.2354E-23, 6.21834E-23, 1.26177E-08, 3.73041E-09, 4.09412E-09, 2.48011E-22, 2.69159E-08, 1.01938E-22, 5.26903E-22, 4.68648E-09, 5.56097E-09, 1.28305E-21, 1.38711E-21, 3.089E-08, 3.2973E-21, 6.33524E-08, 6.88156E-08, 7.93166E-21, 4.29549E-08, 9.51812E-09, 1.01505E-08, 2.10521E-20, 7.66687E-08, 4.22644E-20, 1.07059E-20, 7.60787E-20, 7.17287E-08, 1.31322E-08, 7.58344E-20, 1.36382E-08, 7.10946E-09, 1.1954E-19, 1.13575E-19, 1.74625E-19] # 60Zn resonance alpha widths in MeV from Rauscher

# Verify that all lists have the same length
if not (len(Jr) == len(Er) == len(Gg) == len(Gp) == len(Ga)):
    raise ValueError("Lists Jr, Er, Gg, Gp, and Ga must have the same length")

Jp = 0.5  # proton spin 1/2
JT = 1.5  # 59Cu gs spin 1+
Ap = 1  # proton number
AT = 59  # 59Cu mass number
Reduced_Mass = Ap*AT/(Ap+AT)  # reduced mass

# T9 = np.linspace(0.001, 2.0, 500)  # temperature in GK
T9 = np.arange(0.01, 2.01, 0.01)  # temperature in GK
# print(T9)

# Reaction rates calculation function
def Calculate_Rate(T9, Er, OmegaGamma):
    Rate = np.zeros((len(T9), len(Er)))
    for i in range(len(T9)):
        for j in range(len(Er)):
            Rate[i, j] = 1.5394E11 * (Reduced_Mass * T9[i]) ** (-1.5) * OmegaGamma[j] * np.exp(-11.605 * Er[j] / T9[i])
    return Rate

# OmegaGamma_pa = [(2 * Jr[i] + 1) / (2 * JT + 1) / (2 * Jp + 1) * Gp[i] * Ga[i] / (Gp[i] + Ga[i] + Gg[i]) for i in range(len(Jr))]
# OmegaGamma_pg = [(2 * Jr[i] + 1) / (2 * JT + 1) / (2 * Jp + 1) * Gp[i] * Gg[i] / (Gp[i] + Ga[i] + Gg[i]) for i in range(len(Jr))]

# Initialize a dictionary to count occurrences of high contribution resonances
resonance_count = {energy: 0 for energy in Er}
high_contribution_resonances = []
high_contribution_counts = []

# Loop to run the calculation 10000 times
for run in range(10000):
    # Gaussian random factor with mean 0.3662 and standard deviation 0.8948
    Log_Random = np.random.normal(0.3662, 0.8948, len(Jr))
    Random = 10 ** Log_Random  # Convert to linear scale
    print(f"Run {run + 1}: Random = {Random[0]}")

    # Calculate OmegaGamma element-wise
    OmegaGamma_pa = [
        (2 * Jr[i] + 1) / (2 * JT + 1) / (2 * Jp + 1) * Gp[i] * Ga[i] * Random[i] / (Gp[i] + Ga[i] * Random[i] + Gg[i])
        for i in range(len(Jr))]
    OmegaGamma_pg = [
        (2 * Jr[i] + 1) / (2 * JT + 1) / (2 * Jp + 1) * Gp[i] * Gg[i] / (Gp[i] + Ga[i] * Random[i] + Gg[i])
        for i in range(len(Jr))]

    Rate_pa = Calculate_Rate(T9, Er, OmegaGamma_pa)
    Rate_pg = Calculate_Rate(T9, Er, OmegaGamma_pg)

    # Calculate total reaction rate
    epsilon = 1e-80  # Small value to avoid division by zero
    Total_Rate_pa = np.sum(Rate_pa, axis=1) + epsilon
    Total_Rate_pg = np.sum(Rate_pg, axis=1) + epsilon

    # Calculate contribution of each resonance to the total rate
    Contribution_pa = Rate_pa / Total_Rate_pa[:, None]
    Contribution_pg = Rate_pg / Total_Rate_pg[:, None]

    # Count the number of resonances with contribution > 0.1 for specific temperatures
    temperatures = [1.0]  # Adjust as needed
    for temp in temperatures:
        index = int(temp * 100 - 1)  # Convert temperature to corresponding index
        high_contributions = [(Er[i], Contribution_pa[index][i]) for i in range(len(Er)) if Contribution_pa[index][i] > 0.1]
        
        # Count the high contribution resonances
        num_high_contributions = len(high_contributions)
        high_contribution_counts.append((run + 1, temp, num_high_contributions))
        
        for energy, contribution in high_contributions:
            high_contribution_resonances.append((run + 1, temp, energy, contribution * 100))
            resonance_count[energy] += 1  # Increment the count for this resonance energy

# Save the high contribution resonances to a text file with the specified path
output_path = "D:\\X\\out\\PXCT_59Cu_pa_Reaction_Rate_high_contribution_resonances.txt"
with open(output_path, "w") as file:
    file.write("Run\tTemperature (GK)\tResonance Energy (MeV)\tContribution (%)\n")
    for resonance in high_contribution_resonances:
        file.write(f"{resonance[0]}\t{resonance[1]}\t{resonance[2]:.3f}\t{resonance[3]:.2f}\n")

# Save the resonance occurrence counts to another text file
occurrence_path = "D:\\X\\out\\PXCT_59Cu_pa_Reaction_Rate_high_contribution_resonances_Occurrences.txt"
with open(occurrence_path, "w") as file:
    file.write("Resonance Energy (MeV)\tOccurrences\n")
    for energy, count in resonance_count.items():
        file.write(f"{energy:.4f}\t{count}\n")

# Save the high contribution counts to another text file
count_path = "D:\\X\\out\\PXCT_59Cu_pa_Reaction_Rate_high_contribution_counts.txt"
with open(count_path, "w") as file:
    file.write("Run\tTemperature (GK)\tHigh Contribution Resonances\n")
    for count in high_contribution_counts:
        file.write(f"{count[0]}\t{count[1]}\t{count[2]}\n")

print(f"High contribution resonances have been saved to {output_path}")
print(f"Resonance occurrence counts have been saved to {occurrence_path}")
print(f"High contribution counts have been saved to {count_path}")



'''
# Plot for (p,a) reaction rate
# Define specific colors for high contribution resonances
specific_colors = {
    0.396: 'darkviolet',
    0.540: 'green',
    0.563: 'darkorange',
    0.603: 'maroon',
    0.884: 'red',
    0.933: 'goldenrod',
    0.967: 'royalblue',
    1.006: 'magenta',
    1.196: 'lime',
    # Add more specific colors as needed
}

fig, ax = plt.subplots(figsize=(11, 7))
# Plot all black curves first
for i in range(len(Er)):
    energy = Er[i]
    if energy not in specific_colors:
        ax.plot(T9, Contribution_pa[:, i]*100, color='black', linewidth=1.5)

# Plot colored curves on top
for i in range(len(Er)):
    energy = Er[i]
    if energy in specific_colors:
        color = specific_colors[energy]
        ax.plot(T9, Contribution_pa[:, i]*100, color=color, linewidth=3)

# Add annotations with arrows for specific energies with adjusted positions
annotations = {
    0.396: {'xytext': (0.11, 92)},
    0.540: {'xytext': (0.19, 62)},
    0.563: {'xytext': (0.34, 54)},
    0.603: {'xytext': (0.38, 43)},
    0.884: {'xytext': (0.58, 48)},
    0.933: {'xytext': (0.61, 38)},
    0.967: {'xytext': (0.71, 29)},
    1.006: {'xytext': (0.81, 21)},
    1.196: {'xytext': (1.10, 16)},
    # Add more annotations as needed with their positions
}

for energy, color in specific_colors.items():
    i = Er.index(energy)
    max_contribution = np.max(Contribution_pa[:, i]*100)
    max_index = np.argmax(Contribution_pa[:, i])
    annotation_info = annotations[energy]
    ax.annotate(f'{1000*energy:.0f}', xy=(T9[max_index], max_contribution),
                xytext=annotation_info['xytext'],
                arrowprops=dict(facecolor=color, edgecolor=color, shrink=0.05, width=1, headwidth=6),
                fontsize=30, color=color)

ax.set_xlabel('Temperature (GK)', fontsize=34, labelpad=14)
ax.set_ylabel('Contribution (%)', fontsize=34, labelpad=24)
ax.set_xlim(0, 2)
ax.set_ylim(0, 100)
ax.tick_params(axis='both', which='major', direction='in', length=9, width=2, labelsize=34, pad=10)
ax.set_xticks(np.arange(0, 2.1, 0.5))  # Set ticks at intervals of 0.5
plt.text(1.9, 95, r'$^{59}$Cu$(p,\alpha)^{56}$Ni', fontsize=38, color='black', horizontalalignment='right', verticalalignment='top')

fig.subplots_adjust(left=0.16, bottom=0.18, right=0.96, top=0.96)
# Save figure with the corresponding random number index in D:\X\out
# plt.show()
plt.savefig(f'D:\\X\\out\\Fig_PXCT_59Cu_pa_Reaction_Rate_Contribution_{Random[0]}_{Random[1]}.png', dpi=300)

# Plot for (p,g) reaction rate
# Define specific colors for high contribution resonances
specific_colors = {
    0.396: 'darkviolet',
    0.461: 'darkorange',
    0.540: 'green',
    0.884: 'red',
    0.967: 'royalblue',
    1.200: 'lime',
    # Add more specific colors as needed
}

fig, ax = plt.subplots(figsize=(11, 7))
# Plot all black curves first
for i in range(len(Er)):
    energy = Er[i]
    if energy not in specific_colors:
        ax.plot(T9, Contribution_pg[:, i]*100, color='black', linewidth=1.5)

# Plot colored curves on top
for i in range(len(Er)):
    energy = Er[i]
    if energy in specific_colors:
        color = specific_colors[energy]
        ax.plot(T9, Contribution_pg[:, i]*100, color=color, linewidth=3)

# Add annotations with arrows for specific energies with adjusted positions
annotations = {
    0.396: {'xytext': (0.09, 92)},
    0.461: {'xytext': (0.165, 62)},
    0.540: {'xytext': (0.34, 65)},
    0.884: {'xytext': (0.55, 33)},
    0.967: {'xytext': (0.75, 70)},
    1.200: {'xytext': (1.70, 20)},
    # Add more annotations as needed with their positions
}

for energy, color in specific_colors.items():
    i = Er.index(energy)
    max_contribution = np.max(Contribution_pg[:, i]*100)
    max_index = np.argmax(Contribution_pg[:, i])
    annotation_info = annotations[energy]
    ax.annotate(f'{1000*energy:.0f}', xy=(T9[max_index], max_contribution),
                xytext=annotation_info['xytext'],
                arrowprops=dict(facecolor=color, edgecolor=color, shrink=0.05, width=1, headwidth=6),
                fontsize=30, color=color)

ax.set_xlabel('Temperature (GK)', fontsize=34, labelpad=14)
ax.set_ylabel('Contribution (%)', fontsize=34, labelpad=24)
ax.set_xlim(0, 2)
ax.set_ylim(0, 100)
ax.tick_params(axis='both', which='major', direction='in', length=9, width=2, labelsize=34, pad=10)
ax.set_xticks(np.arange(0, 2.1, 0.5))  # Set ticks at intervals of 0.5
plt.text(1.9, 95, r'$^{59}$Cu$(p,\gamma)^{60}$Zn', fontsize=38, color='black', horizontalalignment='right', verticalalignment='top')

fig.subplots_adjust(left=0.16, bottom=0.18, right=0.96, top=0.96)
# Save figure with the corresponding random number index in D:\X\out
# plt.show()
plt.savefig(f'D:\\X\\out\\Fig_PXCT_59Cu_pg_Reaction_Rate_Contribution_{Random[0]}_{Random[1]}.png', dpi=300)
    
'''