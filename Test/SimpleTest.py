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

def rnd(n):
    values = sps.uniform.rvs(0, 30, size=10000000)
    indices = np.random.randint(0, 10000000, n)  
    return values[indices]

values = rnd(4000000)  
plt.hist(values, density=True, bins=200)
# plt.xlim(0,30)
# plt.ylim(0, 0.00032) 
plt.show()

