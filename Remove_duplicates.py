import pandas as pd
import os

# Define the directory containing the CSV files
directory_path = r'F:\e21010\iPA_Log'  # Update this path as necessary

# Initialize an empty list to store each cleaned DataFrame
cleaned_dataframes = []

# Iterate over all files in the directory
for filename in os.listdir(directory_path):
    # Process only the relevant original CSV files
    if filename.endswith(".csv") and filename.startswith("South5596_Log_") and "_Cleaned" not in filename:
        input_file_path = os.path.join(directory_path, filename)
        
        # Load the CSV file into a DataFrame
        df = pd.read_csv(input_file_path)

        # Remove duplicates based on all columns
        df_cleaned = df.drop_duplicates()

        # Create the output file name by adding "Cleaned" before the file extension
        base_name, ext = os.path.splitext(input_file_path)
        output_file_path = f"{base_name}_Cleaned{ext}"

        # Ensure we don't re-clean an already cleaned file
        if not os.path.exists(output_file_path):
            df_cleaned.to_csv(output_file_path, index=False)
            cleaned_dataframes.append(df_cleaned)

        print(f"Processed {filename}: Original data had {len(df)} rows, cleaned data has {len(df_cleaned)} rows.")

# Combine all cleaned DataFrames into one DataFrame
if cleaned_dataframes:
    combined_df = pd.concat(cleaned_dataframes, ignore_index=True)
    
    # Save the combined DataFrame to a new CSV file
    combined_output_path = os.path.join(directory_path, "South5596_Log_Combined_Cleaned.csv")
    combined_df.to_csv(combined_output_path, index=False)
    
    print(f"Combined cleaned data saved to: {combined_output_path}")
else:
    print("No relevant files found or processed.")
