import os
import pandas as pd
from Consts import CV1_SUBJECT
'''
Combine several validation Excel files into one
'''
def extract_seq_length(file_name):
    """Extract sequence length from file name."""
    seq_length = int(file_name.split('_')[1].replace('seq', ''))
    return seq_length

def merge_excel_files(directory):
    """Merge multiple Excel files into one DataFrame."""
    merged_df = pd.DataFrame()

    for file_name in os.listdir(directory):
        if file_name.endswith('.xlsx'):
            file_path = os.path.join(directory, file_name)
            df = pd.read_excel(file_path)
            seq_length = extract_seq_length(file_name)
            df['Seq_Length'] = seq_length
            merged_df = pd.concat([merged_df, df], ignore_index=True)

    return merged_df

subject_root='cv1_with_temp'
subject=CV1_SUBJECT
# Directory containing the Excel files
directory = f'{subject_root}/{subject}'

# Merge files and save to a new Excel file
merged_df = merge_excel_files(directory)
output_file = f'{subject_root}/{subject}/{subject}_merged.xlsx'
merged_df.to_excel(output_file, index=False)

print(f"Merged file saved to: {output_file}")
