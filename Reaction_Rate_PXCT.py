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

# gaussian random factor with mean 0.3662 and standard deviation 0.8948
# This is a random factor to account for the uncertainty in the statistical model calculated alpha width
# np.random.seed(0) # Set seed0 for reproducibility
Log_Random = np.random.normal(0.3662, 0.8948, len(Jr))
Random = 10**Log_Random # Convert to linear scale
print(Random)

# Calculate OmegaGamma element-wise
# OmegaGamma = [(2 * Jr[i] + 1) / (2 * JT + 1) / (2 * Jp + 1) * Gp[i] * Ga[i] / (Gp[i] + Ga[i] + Gg[i]) for i in range(len(Jr))]
OmegaGamma = [(2 * Jr[i] + 1) / (2 * JT + 1) / (2 * Jp + 1) * Gp[i] * Ga[i] * Random[i] / (Gp[i] + Ga[i] * Random[i] + Gg[i]) for i in range(len(Jr))]

T9 = np.linspace(0.001, 2.0, 300)  # temperature in GK

# Reaction rates calculation function
def Calculate_Rate(T9, Er, OmegaGamma):
    Rate = np.zeros((len(T9), len(Er)))
    for i in range(len(T9)):
        for j in range(len(Er)):
            Rate[i, j] = 1.5394E11 * (Reduced_Mass * T9[i]) ** (-1.5) * OmegaGamma[j] * np.exp(-11.605 * Er[j] / T9[i])
    return Rate

Rate = Calculate_Rate(T9, Er, OmegaGamma)

# Calculate total reaction rate, avoiding division by zero
epsilon = 1e-60  # Small value to avoid division by zero
Total_Rate = np.sum(Rate, axis=1) + epsilon

# Calculate contribution of each resonance to the total rate
Contribution = Rate / Total_Rate[:, None]

# Plot
fig, ax = plt.subplots(figsize=(10, 6))
for i in range(len(Er)):
    if np.max(Contribution[:, i]) > 0.1:  # Only add to legend if contribution exceeds 20%
        ax.plot(T9, Contribution[:, i], label=f'E$_r$ = {Er[i]:.3f} MeV')
    else:
        ax.plot(T9, Contribution[:, i])

ax.set_xlabel('Temperature (GK)', fontsize=20)
ax.set_ylabel('Contribution to total reaction rate', fontsize=20)
ax.set_xlim(0, 2)
ax.set_ylim(0, 1)
ax.tick_params(axis='both', which='major', labelsize=18)
ax.legend(fontsize=13, loc='upper right')

plt.tight_layout()
plt.show()
