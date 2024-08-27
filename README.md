# Estimation of Lower Limb Torque: A Novel Hybrid Method Based on Continuous Wavelet Transform and Deep Learning Approach

## Description
This project analyzes IMU data using a combination of Continuous Wavelet Transform (CWT) and deep learning methods to estimate lower limb torque during human activities.

## Dataset

[A comprehensive, open-source dataset of lower limb biomechanics in multiple conditions of stairs, ramps, and level-ground ambulation and transitions](https://www.epic.gatech.edu/opensource-biomechanics-camargo-et-al/),[1_normali_zation_AB0_rampascent.csv](/Processed%20data/1_normali_zation_AB0_rampascent.csv),[2_filtering_AB0_e_rampascent_imu.csv](/Processed%20data/2_filtering_AB0_e_rampascent_imu.csv)

## Usage
  
1. **Setup Environment**:
   - Ensure that you meet the hardware and software requirements specified above.
   - Install the necessary Python libraries by running the following command:
     ```bash
     pip install numpy pandas tensorflow keras
     ```

2. **Data Preprocessing**:
   - **Step 1: Filter IMU Data**  
     Run the `1_imu_filtering.py` script to filter the raw IMU data contained in `1_normalization_AB0_rampascent.csv`. This step removes noise and unwanted components from the data.
     ```bash
     python 1_imu_filtering.py
     ```
     After running this script, you will get a new filtered data file.

   - **Step 2: Expand IMU Data**  
     Use the `2_imu_data_expansion.py` script to expand the filtered IMU data from `2_filtering_AB0_e_rampascent_imu.csv`. This script prepares the data for further processing by adding additional features or data points as required by the model.
     ```bash
     python 2_imu_data_expansion.py
     ```
     The output will be a file containing the expanded dataset, which is ready for input into the model.

3. **Model Training and Evaluation**:
   - **Step 3: Train the Model**  
     Execute the `3_cs-104507-peerj_code_cwt_1dcnn_03.py` script. This script performs several critical tasks:
     - Splits the dataset into training, validation, and test sets.
     - Applies Continuous Wavelet Transform (CWT) to the data to extract time-frequency features.
     - Trains the deep learning model, which includes a convolutional neural network (CNN) for feature extraction, and a Bi-directional Long Short-Term Memory (Bi-LSTM) network with a multi-head self-attention mechanism for sequence modeling.
     - Evaluates the model's performance based on metrics like RMSE, R², and Pearson correlation coefficient.
     ```bash
     python 3_cs-104507-peerj_code_cwt_1dcnn_03.py
     ```

4. **Results**:
   - Once the model has been trained and evaluated, the script will output the estimated lower limb torques.
   - You can compare the predicted torques against the actual values in your dataset to assess the model's accuracy.
   - Results will be saved in specified output files or displayed directly in the console/graphical plots, depending on how the script is configured.

5. **Optional: Customization**:
   - If you need to adjust parameters (e.g., model architecture, CWT settings, dataset split ratios), you can modify the corresponding sections in the scripts before running them.
   - Detailed comments within the code will guide you on how to make these adjustments.
  
`Noted that different versions of tensorflow and CUDA may cause the results to be slightly different from the results in the article.`
## Environment Requirements
- **Hardware Requirements**:
  - 12th Gen Intel® Core™ i9-12900K Processor, clock speed 3.20 GHz
  - 64 GB system RAM
  - Two NVIDIA GeForce RTX 3090 GPUs, each with 24 GB GDDR6X RAM
- **Operating System**: Ubuntu 20.04
- **Python Version**: 3.10
- **Python Libraries**: `numpy`, `pandas`, `tensorflow`, `keras`

## File List and Descriptions

### Processed Data
- `1_normalization_AB0_rampascent.csv`: Normalized data referenced in the document.
- `2_filtering_AB0_e_rampascent_imu.csv`: Filtered IMU data.
- `3_imu_data_expansion_AB0_rampascent_imu.csv`: Expanded IMU data, serving as the complete dataset.

### Related Programs
- `1_imu_filtering.py`: Filters IMU data from `1_normalization_AB0_rampascent.csv`.
- `2_imu_data_expansion.py`: Expands IMU data from `2_filtering_AB0_e_rampascent_imu.csv`.
- `3_cs-104507-peerj_code_cwt_1dcnn_03.py`: Splits the dataset into training, validation, and test sets; processes data with CWT; implements the deep learning model; and evaluates results.

  
## Email:

If you have any questions, please email to: [shuxu@mail.ustc.edu.cn](mailto:shuxu@mail.ustc.edu.cn)

