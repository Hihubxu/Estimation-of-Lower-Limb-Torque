import os
import pandas as pd
import numpy as np
from tqdm import tqdm

from tqdm import tqdm

def load_data(file_path):
    """Load data from CSV file, add exception handling to ensure file exists"""
    try:
        data = pd.read_csv(file_path)
        return data
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None

# Insert the features at the end of each sensor data block in the original dataset
def insert_features_after_sensor_columns(df, features_dfs, sensor_prefixes):
    for sensor_prefix, features_df in zip(sensor_prefixes, features_dfs):
        sensor_columns = [col for col in df.columns if col.startswith(sensor_prefix)]
        if sensor_columns:
            last_sensor_col_index = df.columns.get_loc(sensor_columns[-1]) + 1
            for feature_name, feature_data in features_df.items():
                df.insert(last_sensor_col_index, feature_name, feature_data)
                last_sensor_col_index += 1
    return df


folder_path =r'D:\date\shuju'

activity_patterns = ['walk', 'rampascent', 'rampdescent', 'stairascent', 'stairdescent', 'treadmill']

# Handle each activity mode
for activity in tqdm(activity_patterns, desc='Processing cycles imu Processing'):
    input_path = os.path.join(folder_path, f'2_filtering_AB0_e_{activity}_imu.csv')
    df = load_data(input_path)

    print(f'Processing：{input_path}')
    if df is not None:
        features_dfs = []
        sensor_prefixes = []
        # Calculate features for foot, shank, thigh, and trunk respectively
        sensor_locations = ['foot', 'shank', 'thigh', 'trunk']
        for location in sensor_locations:
            prefix = f'{location}_'

            # Extracting acceleration and angular velocity data
            acc_x = df[prefix + 'Accel_X']
            acc_y = df[prefix + 'Accel_Y']
            acc_z = df[prefix + 'Accel_Z']
            gyro_x = df[prefix + 'Gyro_X']
            gyro_y = df[prefix + 'Gyro_Y']
            gyro_z = df[prefix + 'Gyro_Z']


            l2_norm_acc = np.sqrt(acc_x ** 2 + acc_y ** 2 + acc_z ** 2)
            l2_norm_gyro = np.sqrt(gyro_x ** 2 + gyro_y ** 2 + gyro_z ** 2)
            # # Calculate the mean
            mean_acc = (acc_x + acc_y + acc_z) / 3
            mean_gyro = (gyro_x + gyro_y + gyro_z) / 3
            # # Calculate the ratio of acceleration to angular velocity
            acc_gyro_ratio = l2_norm_acc / l2_norm_gyro
            g = 9.81
            # Calculating dynamic acceleration
            dynamic_acc = np.sqrt(acc_x ** 2 + acc_y ** 2 + (acc_z - g) ** 2)

            features_df = pd.DataFrame({
                prefix + 'L2_Norm_Acc': l2_norm_acc,
                prefix + 'L2_Norm_Gyro': l2_norm_gyro,
                prefix + 'Mean_Acc': mean_acc,
                prefix + 'Mean_Gyro': mean_gyro,
                prefix + 'Acc_Gyro_Ratio': acc_gyro_ratio,
                prefix + 'Dynamic_Acc': dynamic_acc
            })

            features_dfs.append(features_df)
            sensor_prefixes.append(prefix)

        df = insert_features_after_sensor_columns(df, features_dfs, sensor_prefixes)
        filtered_df_path = os.path.join(folder_path, f'3_imu_data_expansion_{activity}_imu.csv')
        df.to_csv(filtered_df_path, index=False)
        print(f"The processed data has been saved to {filtered_df_path}")
        # Display the header of the processed data
        print(df.head())