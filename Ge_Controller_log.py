import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import os


# plots a list of profiles in the same figure. Each profile corresponds to a simulation replica for the given instance.
plt.rcParams['axes.linewidth'] = 3.0
plt.rcParams['font.size'] = 60
font_family_options = ['Times New Roman', 'Georgia', 'Cambria', 'Courier New', 'serif']
plt.rcParams['font.family'] = font_family_options
# plt.rcParams['mathtext.default'] = 'regular'
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Times New Roman'
plt.rcParams['mathtext.it'] = 'Times New Roman:italic'


# Define the file paths directly
north_file_path = r'D:\X\out\Bayesian_VS\North_5593_controllerlog.txt'
south_file_path = r'D:\X\out\Bayesian_VS\South_5596_controllerlog.txt'
lege_file_path = r'D:\X\out\Bayesian_VS\LEGe_13725_controllerlog.txt'

# Directory where you want to save the figures
save_directory = r'D:\X\out\Bayesian_VS'

# Read the data from the text files
north_data = pd.read_csv(north_file_path, sep='\t')
south_data = pd.read_csv(south_file_path, sep='\t')
lege_data = pd.read_csv(lege_file_path, sep='\t')

# Combine 'Date' and 'Time' columns and convert to datetime
def convert_to_datetime(df):
    df['Time'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%m/%d/%Y %H:%M:%S')
    df.drop('Date', axis=1, inplace=True)
    return df

north_data = convert_to_datetime(north_data)
south_data = convert_to_datetime(south_data)
lege_data = convert_to_datetime(lege_data)

# Function to plot and save figures
def plot_data(title, ylabel, data_field, file_name):
    fig, ax = plt.subplots(figsize=(36, 13))
    fig.subplots_adjust(left=0.10, bottom=0.20, right=0.98, top=0.91)
    
    ax.scatter(north_data['Time'], north_data[data_field], color='red', s=6, label='North', alpha=0.9, linewidth=0)
    ax.scatter(south_data['Time'], south_data[data_field], color='blue', s=6, label='South', alpha=0.9, linewidth=0)
    ax.scatter(lege_data['Time'], lege_data[data_field], color='green', s=6, label='LEGe', alpha=0.9, linewidth=0)
    
    ax.set_title(title, fontsize=60)
    ax.set_xlabel('Time', fontsize=60, labelpad=40)
    ax.set_ylabel(ylabel, fontsize=60, labelpad=40)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%Y'))
    ax.yaxis.set_major_formatter(mticker.StrMethodFormatter('{x:.0f}'))
    ax.legend(fontsize=60, loc='best', markerscale=3)
    ax.tick_params(axis='both', labelsize=60, length=9, width=2)
    plt.xticks(rotation=45)
    
    ax.grid(True)
    
    for spine in ax.spines.values():
        spine.set_linewidth(2)
    
    # Save the figure to the specified directory
    full_path = os.path.join(save_directory, file_name)
    plt.savefig(full_path)

# Plot and save figures
plot_data('Cooler Power Over Time', 'Cooler power', 'Cooler power', 'cooler_power.png')
plot_data('Coldtip Temperature Over Time', 'Coldtip temp', 'Coldtip temp', 'coldtip_temp.png')
plot_data('Coldhead Temperature Over Time', 'Coldhead temp', 'Coldhead temp', 'coldhead_temp.png')
