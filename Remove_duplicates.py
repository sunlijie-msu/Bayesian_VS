import pandas as pd
import os

# Define the input file path
# input_file_path = r'C:\Users\sun\Downloads\North5593_Log_021.csv'
input_file_path = r'C:\Users\sun\Downloads\South5596_Log_022.csv'

# Load the CSV file into a DataFrame
df = pd.read_csv(input_file_path)

# Remove duplicates based on all columns
df_cleaned = df.drop_duplicates()

# Create the output file name by adding "_cleaned" before the file extension
base_name, ext = os.path.splitext(input_file_path)
output_file_path = f"{base_name}_cleaned{ext}"

# Save the cleaned DataFrame back to a new CSV file
df_cleaned.to_csv(output_file_path, index=False)

print(f"Original data had {len(df)} rows.")
print(f"Data after removing duplicates has {len(df_cleaned)} rows.")
print(f"Cleaned data saved to: {output_file_path}")
