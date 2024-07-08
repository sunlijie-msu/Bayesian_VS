import numpy as np
import matplotlib.pyplot as plt

# Energy levels (in keV)
E = np.array([0, 491.5, 914.2])
g = np.array([4, 2, 6])

# Temperatures in GK and corresponding partition functions
T9_values = [0.01, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,
             0.8, 0.9, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5,
             4.0, 4.5, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]

Z_values = [1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000006, 1.000038, 1.000146,
            1.000405, 1.000900, 1.001712, 1.012506, 1.037057, 1.076405, 1.130801, 1.201070,
            1.289277, 1.399034, 1.535686, 1.921222, 2.535596, 3.528145, 5.146226, 7.801148]

# Boltzmann constant in keV/GK
kT_per_GK = 86.17

# Calculate thermal populations for each temperature
ground_state_populations = []
first_excited_state_populations = []
second_excited_state_populations = []

for T9, Z in zip(T9_values, Z_values):
    kT = T9 * kT_per_GK
    Boltzmann_factors = np.exp(-E / kT)
    populations = (g * Boltzmann_factors) / Z
    ground_state_populations.append(populations[0])
    first_excited_state_populations.append(populations[1])
    second_excited_state_populations.append(populations[2])

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(T9_values, ground_state_populations, label='Ground state')
plt.plot(T9_values, first_excited_state_populations, label='First excited state')
plt.plot(T9_values, second_excited_state_populations, label='Second excited state')
plt.xlabel('Temperature (GK)')
plt.ylabel('Population Fraction')
plt.title('Thermal Population of Excited States in $^{59}$Cu')
plt.legend()
plt.grid(True)
plt.yscale('log')
plt.ylim(1e-4, 10)
plt.savefig('Thermal_Population_59Cu.png')
plt.show()
