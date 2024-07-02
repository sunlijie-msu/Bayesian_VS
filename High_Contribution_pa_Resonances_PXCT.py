import pandas as pd

# Load the data from the output file
file_path = "D:\\X\\out\\PXCT_59Cu_pa_Reaction_Rate_high_contribution_counts.txt"
data = pd.read_csv(file_path, delimiter='\t')

# Count the frequency of each number of high contribution resonances
frequency_counts = data['High Contribution Resonances'].value_counts().sort_index()

# Save the frequency counts to a new text file
frequency_output_path = "D:\\X\\out\\PXCT_59Cu_pa_Reaction_Rate_high_contribution_counts_frequency.txt"
with open(frequency_output_path, "w") as file:
    file.write("High Contribution Resonances\tFrequency\n")
    for count, frequency in frequency_counts.items():
        file.write(f"{count}\t{frequency}\n")

print(f"Frequency of high contribution resonances has been saved to {frequency_output_path}")
