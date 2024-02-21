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

# def rnd(n):
#     values = sps.uniform.rvs(0, 30, size=10000000)
#     indices = np.random.randint(0, 10000000, n)  
#     return values[indices]

# values = rnd(4000000)  
# plt.hist(values, density=True, bins=200)
# # plt.xlim(0,30)
# # plt.ylim(0, 0.00032) 
# plt.show()

# Correcting the error in the previous code attempt and recalculating E and its error using proper error propagation.

# Given values
T = 13444.6
A = 3007
error_A = 90.21
N = 129436
error_N = 359.772
I = 85.1 / 100  # Convert percentage to fraction
error_I = 0.2 / 100  # Convert percentage error to fraction

# Calculate E
E = N / (A * T * I)

# Calculate relative errors for A, N, and I
relative_error_A = error_A / A
relative_error_N = error_N / N
relative_error_I = error_I / I

# Calculate the total relative error for E using the square root of the sum of squares of the relative errors
total_relative_error_E = (relative_error_A ** 2 + relative_error_N ** 2 + relative_error_I ** 2) ** 0.5

# Calculate the absolute error of E
error_E = E * total_relative_error_E

print(f"E = {E} and {error_E}")
