import pandas as pd
from datetime import timedelta
import matplotlib.pyplot as plt
import os

# Global settings for plots
plt.rcParams['axes.linewidth'] = 4.0
plt.rcParams['font.size'] = 60
plt.rcParams['font.family'] = ['Times New Roman', 'Georgia', 'Cambria', 'Courier New', 'serif']
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Times New Roman'
plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
plt.rcParams['legend.frameon'] = False

# Function to parse timestamps in 'd:hh:mm:ss' format
def parse_time(time_string):
    if time_string == '':
        return None
    parts = list(map(int, time_string.split(':')))
    return timedelta(days=parts[0], hours=parts[1], minutes=parts[2], seconds=parts[3])


# Function to adjust timestamps for resets
# def adjust_timestamps(df):
#     adjusted_times = []
#     cumulative_days = 0
    
#     for i in range(len(df)):
#         if i > 0 and df.iloc[i]["Time Stamp"] < df.iloc[i - 1]["Time Stamp"]:
#             cumulative_days += 1
#         adjusted_time = df.iloc[i]["Time Stamp"] + timedelta(days=cumulative_days * df.iloc[i]["Time Stamp"].days)
#         adjusted_times.append(adjusted_time)

#     df["Time Stamp"] = adjusted_times
#     return df


# Read the CSV files
df_north = pd.read_csv(r'D:\X\out\Bayesian_VS\North5593_iPA_Log_For_Plot.csv', converters={'Time Stamp': parse_time})
df_south = pd.read_csv(r'D:\X\out\Bayesian_VS\South5596_iPA_Log_For_Plot.csv', converters={'Time Stamp': parse_time})
df_lege = pd.read_csv(r'F:\e21010\iPA_Log\LEGe_13725_iPAlog.csv', converters={'Time Stamp': parse_time})

# Parameters to filter
parameters = ["11:PRTD 1", "12:PRTD 2", "13:Ambient Temperature", "73:DC Detector Leakage Current", "16:Charge Loop DC Level"]

# Custom y-axis labels for each plot
custom_y_labels = [
    "PRTD1 (°C)",
    "PRTD2 (°C)",
    "Ambient T (°C)",
    r"$\mathit{I}_{\mathrm{leak}}$ (pA)",  # LaTeX syntax
    "DC Level (V)"
]

# Filter the data
df_north_filtered = [df_north[df_north["Parameter Code"] == param] for param in parameters]
df_south_filtered = [df_south[df_south["Parameter Code"] == param] for param in parameters]
df_lege_filtered = [df_lege[df_lege["Parameter Code"] == param] for param in parameters]

# Create a 5x1 grid of plots
fig, axs = plt.subplots(5, 1, figsize=(32, 25), sharex=True)
plt.tight_layout()
plt.subplots_adjust(top=0.96, bottom=0.10, hspace=0.23, left=0.11, right=0.97)

# Plot each parameter
for i, param in enumerate(parameters):
    axs[i].scatter(df_north_filtered[i]["Time Stamp"].dt.total_seconds() / (24 * 60 * 60), df_north_filtered[i]["Parameter Value"], color='red', alpha=0.9, label='North', s=40, linewidth=0)
    axs[i].scatter(df_south_filtered[i]["Time Stamp"].dt.total_seconds() / (24 * 60 * 60), df_south_filtered[i]["Parameter Value"], color='#0033ff', alpha=0.9, label='South', s=40, linewidth=0)
    axs[i].scatter(df_lege_filtered[i]["Time Stamp"].dt.total_seconds() / (24 * 60 * 60), df_lege_filtered[i]["Parameter Value"], color='green', alpha=0.9, label='LEGe', s=40, linewidth=0)
    
    # Set the custom y-label instead of the dataset label
    axs[i].set_ylabel(custom_y_labels[i], fontsize=60, labelpad=30)
    axs[i].tick_params(axis='x', labelsize=60, length=9, width=3)
    axs[i].tick_params(axis='y', labelsize=60, length=9, width=3)
    
    # Adjust legend position to avoid blocking data
    if i == 1:
        legend = axs[i].legend(fontsize=60, loc='best', markerscale=3, frameon=False, labelspacing=0.2, handletextpad=-0.3)
        for text, color in zip(legend.get_texts(), ['red', '#0033ff', 'green']):
            text.set_color(color)
    
    
    # Set consistent x-axis limits
    axs[i].set_xlim([0, 160])
    
    # Set y-axis limits for specific plots if necessary
    if param == "11:PRTD 1":
        axs[i].set_ylim([-200, 40])
    elif param == "12:PRTD 2":
        axs[i].set_ylim([-220, 40])
    elif param == "13:Ambient Temperature":
        axs[i].set_ylim([19, 48])
    elif param == "73:DC Detector Leakage Current":
        axs[i].set_ylim([-200, 1100])
    elif param == "16:Charge Loop DC Level":
        axs[i].set_ylim([-2, 0.1])

# Only add the x-axis label to the bottom plot
axs[-1].set_xlabel("Time (days)", fontsize=60, labelpad=20)


# Save the figure
plt.savefig(r'D:\X\out\Bayesian_VS\Fig_PXCT_iPA_log.png')
