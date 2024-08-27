

# First use EDA to check the data, put the results into txt, analyze the results, and generate the corresponding filter program
# First filter the program, then generate snr and rmse, and then adjust the filter program to ensure that SNR is a positive number
import os

import pandas as pd
import numpy as np
from scipy.stats import zscore
from scipy.signal import butter, filtfilt

from tqdm import tqdm


# Define the Butterworth low-pass filter function
def butter_lowpass_filter(data, cutoff, fs, order):

    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

# Apply the mean filter function
def apply_mean_filter(x, filt_length):

    filter_kernel = np.ones(filt_length) / filt_length
    return np.convolve(x, filter_kernel, mode='same')

# Repeatedly apply the mean filter function
def repeat_filter(x, n, filt_length):

    for _ in range(n):
        x = apply_mean_filter(x, filt_length)
    return x

# Function: Count the number of 0-1 cycles
def count_cycles(data):

    return sum(data['Header'] == 0)

# Calculate the signal-to-noise ratio of a signal
def calculate_snr(signal, noise):

    power_signal = np.mean(signal ** 2)
    power_noise = np.mean(noise ** 2)
    snr = 10 * np.log10(power_signal / power_noise)
    return snr


# Define the function to calculate the root mean square error
def calculate_rmse(signal, reference):

    mse = np.mean((signal - reference) ** 2)
    rmse = np.sqrt(mse)
    return rmse

# Replace to the corresponding folder
folder_path =r'D:\date\shuju'

activity_patterns = ['walk', 'rampascent', 'rampdescent', 'stairascent', 'stairdescent', 'treadmill']
# activity_patterns = ['walk', 'rampascent', 'rampdescent', 'stairascent', 'stairdescent']
# Loop through the data for each activity type
for activity in tqdm(activity_patterns, desc='Processing cycles imu Filtering'):

    input_path = os.path.join(folder_path, f'1_normali_zation_AB0_{activity}.csv')
    df = pd.read_csv(input_path, header=0)
    original_df = df.copy()  # Copy the original data as a baseline for subsequent comparisons
    # Calculate the number of original 0-1 cycles
    original_cycle_count = count_cycles(df)


    print(f'Processing：{input_path}')

    # 设置滤波参数
    cutoff = 3.5  # Cutoff frequency
    fs = 100  # Sampling frequency
    order = 6  # Filter order
    columns_to_check = ['foot_Accel_X', 'foot_Accel_Y', 'foot_Accel_Z', 'foot_Gyro_X', 'foot_Gyro_Y', 'foot_Gyro_Z',
                        'shank_Accel_X', 'shank_Accel_Y', 'shank_Accel_Z', 'shank_Gyro_X', 'shank_Gyro_Y', 'shank_Gyro_Z',
                        'thigh_Accel_X', 'thigh_Accel_Y', 'thigh_Accel_Z', 'thigh_Gyro_X', 'thigh_Gyro_Y', 'thigh_Gyro_Z',
                        'trunk_Accel_X', 'trunk_Accel_Y', 'trunk_Accel_Z', 'trunk_Gyro_X', 'trunk_Gyro_Y', 'trunk_Gyro_Z']
    # Filter the signal
    print("Apply Filter...")

    # Apply Filter
    for col in tqdm(columns_to_check, desc="Processing signal sequence",
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'):
        if "Gyro" in col:
            df[col] = butter_lowpass_filter(df[col].values, cutoff, fs, order)
        else:
            df[col] = repeat_filter(df[col].values, n=3, filt_length=5)
        # Create a Cycle_Index column before processing outliers
    df['Cycle_Index'] = df.index // 101  # This should be consistent with the period definition in your data

    print(f"Number of original 0-1 cycles: {original_cycle_count}")

    processed_df=df
    # Save the processed data to a CSV file
    filtered_df_path =os.path.join(folder_path, f'2_filtering_AB0_e_{activity}_imu.csv')

    df.to_csv(filtered_df_path, index=False)


    print(f"The processed data has been saved to {filtered_df_path}")


 # Calculate the RMSE and SNR of each signal column and save them in a TXT file
    results = []
    for col in tqdm(columns_to_check, desc="Calculating RMSE and SNR",
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'):
        original_signal = original_df[col].dropna().values
        filtered_signal = processed_df[col].values
        if len(original_signal) > len(filtered_signal):
            original_signal = original_signal[:len(filtered_signal)]
        noise = original_signal - filtered_signal
        snr = calculate_snr(filtered_signal, noise)
        rmse = calculate_rmse(filtered_signal, original_signal)
        results.append(f"{col}List: RMSE = {rmse:.4f}, SNR = {snr:.4f} dB")

    # Save the results to a TXT file
    results_filepath = os.path.join(folder_path, f'2_filtering_snr_AB0_e_{activity}_imu.txt')
    with open(results_filepath, "w", encoding="utf-8") as f:
        for line in results:
            f.write(line + "\n")

    print(f"Results saved to {results_filepath}")
