import pandas as pd
import subprocess
import os

# Paths
script_dir = os.path.dirname(os.path.abspath(__file__))
print(script_dir)
pdf_path = os.path.join(script_dir, "zn60t.pdf")
excel_path = os.path.join(script_dir, "zn60t_improved.xlsx")
tabula_jar_path = os.path.join(script_dir, "tabula.jar")

# Run tabula-java as a subprocess
try:
    result = subprocess.run(
        ['java', '-jar', tabula_jar_path, '--pages', 'all', '--multiple-tables', pdf_path],
        capture_output=True, text=True, check=True
    )
    tables = result.stdout.split('\n\n')
    
    # Convert tables to DataFrames
    df_list = [pd.read_csv(pd.compat.StringIO(table)) for table in tables if table.strip()]

    # Ensure df_list is not empty
    if df_list:
        # Export the tables to an Excel file
        with pd.ExcelWriter(excel_path) as writer:
            for i, df in enumerate(df_list):
                # Check if the DataFrame is not empty before saving
                if not df.empty:
                    df.to_excel(writer, sheet_name=f"Table_{i+1}", index=False)
        
        print(f"Tables extracted and saved to {excel_path}")
    else:
        print("No tables found in the PDF.")
except subprocess.CalledProcessError as e:
    print(f"An error occurred: {e}")
