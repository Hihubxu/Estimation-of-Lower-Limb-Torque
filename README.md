# Estimation-of-Lower-Limb-Torque
Real-time analysis of lower limb joint torque using IMU data and deep learning. Combines time-frequency analysis, multi-head self-attention, and Bi-LSTM to estimate hip, knee, and ankle torques with high accuracy 
# Estimation of Lower Limb Torque: A Novel Hybrid Method Based on Continuous Wavelet Transform and Deep Learning Approach

## Description
This project analyzes IMU data using a combination of Continuous Wavelet Transform (CWT) and deep learning methods to estimate lower limb torque during human activities.

## Environment Requirements
- **Hardware Requirements**:
  - 12th Gen Intel® Core™ i9-12900K Processor, clock speed 3.20 GHz
  - 64 GB system RAM
  - Two NVIDIA GeForce RTX 3090 GPUs, each with 24 GB GDDR6X RAM
- **Operating System**: Ubuntu 20.04
- **Python Version**: 3.10
- **Python Libraries**: `numpy`, `pandas`, `tensorflow`, `keras`

## File List and Descriptions

### Related Data
- `1_normalization_AB0_rampascent.csv`: Normalized data referenced in the document.
- `2_filtering_AB0_e_rampascent_imu.csv`: Filtered IMU data.
- `3_imu_data_expansion_AB0_rampascent_imu.csv`: Expanded IMU data, serving as the complete dataset.

### Related Programs
- `1_imu_filtering.py`: Filters IMU data from `1_normalization_AB0_rampascent.csv`.
- `2_imu_data_expansion.py`: Expands IMU data from `2_filtering_AB0_e_rampascent_imu.csv`.
- `3_cs-104507-peerj_code_cwt_1dcnn_03.py`: Splits the dataset into training, validation, and test sets; processes data with CWT; implements the deep learning model; and evaluates results.
