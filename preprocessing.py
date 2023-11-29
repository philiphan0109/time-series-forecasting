import pandas as pd 
import os

def process_file(path):
    df = pd.read_csv(path, delim_whitespace=True, header=None)
    loc = path.split('.')[3] + path.split('.')[4]

    df.columns = [
        'year', 'month', f'temp_anomaly_{loc}', f'total_error_var_{loc}', f'high_freq_error_var_{loc}',
        f'low_freq_error_var_{loc}', f'bias_error_var_{loc}', f'diag_var1_{loc}', f'diag_var2_{loc}', f'diag_var3_{loc}'
    ]

    df = df.drop(columns=[f'diag_var1_{loc}', f'diag_var2_{loc}', f'diag_var3_{loc}', f'total_error_var_{loc}', f'high_freq_error_var_{loc}',
        f'low_freq_error_var_{loc}', f'bias_error_var_{loc}'])

    return df

data_path = "data/"

all_data = pd.DataFrame()

for file in os.listdir(data_path):
    if file.endswith(".txt"):
        file_path = os.path.join(data_path, file)
        loc_df = process_file(file_path)


    # print(file, len(loc_df))
    if len(all_data) == 0:
        all_data = loc_df
    else:
        all_data = pd.merge(all_data, loc_df, on = ["year", "month"])

# all_data.to_csv('data/processed_data.csv', index=False)

segments = []
start_year = all_data['year'].min()
end_year = all_data['year'].max()

for year in range(start_year, end_year, 11):
    # Get the subset of data for the current 11-year segment
    segment_end = min(year + 11, end_year + 1)  # Ensure we don't go beyond the last year
    segment_df = all_data[(all_data['year'] >= year) & (all_data['year'] < segment_end)]
    
    if not segment_df.empty:
        segments.append(segment_df)

print(segments[0].shape)