import pandas as pd
import os

from model import Transformer
import torch
from torch import nn
import numpy as np

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

def create_segments(data, input_years_length, target_years_length, overlapping = True):
    input_data = []
    target_data = []
    
    total_years = input_years_length + target_years_length

    start_year = data['year'].min()
    end_year = data['year'].max()

    if overlapping:
        step = 1
    else:
        step = total_years

    for year in range(start_year, end_year - total_years + 2, step):
        segment_end_year = year + total_years
        segment_df = data[(data['year'] >= year) & (data['year'] < segment_end_year)]

        if not segment_df.empty:
            input_segment = segment_df.head(input_years_length)
            target_segment = segment_df.tail(target_years_length)
            input_data.append(input_segment)
            target_data.append(target_segment)

    return input_data, target_data
