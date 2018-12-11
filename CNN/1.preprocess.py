
#%%
import pathlib
import pickle
import random
import re
import sys

import nibabel as nib
import numpy as np
import pandas as pd


# Path to the data folder. This may be different between users
location = '../data/secure/'

data_path = pathlib.Path(location)

# Rename folders
for feature_dir in data_path.glob('*/*/*/feature/'):
    parent_dir = feature_dir.parent
    parent_dir.rename(parent_dir.parent / 'feature_masks')

# Train/test split
# Get list of patients with feature maps
patients_list = [subdir.name for subdir in data_path.glob('*/') if subdir.is_dir()]

# Read in case/control information
case_control_df = pd.read_excel('../controlcase.xlsx')

# Create a dictionary mapping patient_id to case/control status
patient_id_to_case = case_control_df[['DummyID', 'Class']].set_index('DummyID')['Class'].to_dict()

# Set random seed so that split can be done reproducibly
np.random.seed(0)

# Pick patients whose images will be in train/test sets
training_patients = np.random.choice(patients_list, replace=False, size=455)
testing_patients = [patient for patient in patients_list if patient not in training_patients]

# Verify the train/test split sizes
print(f'Training patients: {len(training_patients)}\n'
      f'Testing patients: {len(testing_patients)}\n')

# Verify the relative numbers of cases and controls between training and testing
num_training_cases = sum([patient_id_to_case[int(patient_id)] for patient_id in training_patients])
num_testing_cases = sum([patient_id_to_case[int(patient_id)] for patient_id in testing_patients])

print(f'Percent cases in training data: {num_training_cases / len(training_patients)}\n'
      f'Percent cases in testing data: {num_testing_cases / len(testing_patients)}')

# Replicate the number of training cases
is_train_case = [patient_id_to_case[int(patient_id)] for patient_id in training_patients]
training_patients = np.concatenate((
    3 * [case for i, case in enumerate(training_patients) if is_train_case[i]],
    training_patients
))

# Verify the train/test split sizes
print(f'Training patients: {len(training_patients)}\n'
      f'Testing patients: {len(testing_patients)}\n')

# Verify the relative numbers of cases and controls between training and testing
num_training_cases = sum([patient_id_to_case[int(patient_id)] for patient_id in training_patients])
num_testing_cases = sum([patient_id_to_case[int(patient_id)] for patient_id in testing_patients])

print(f'Percent cases in training data: {num_training_cases / len(training_patients)}\n'
      f'Percent cases in testing data: {num_testing_cases / len(testing_patients)}')

# Write the ids of patients in training/testing to a file
# so that our methods can be replicated exactly
with open('data/training_patients.txt', 'w') as train_file:
    train_file.write('patient_id,case_status\n')
    for patient_id in training_patients:
        case_status = patient_id_to_case[int(patient_id)]
        train_file.write(f'{patient_id},{case_status}\n')

with open('data/testing_patients.txt', 'w') as test_file:
    test_file.write('patient_id,case_status\n')
    for patient_id in testing_patients:
        case_status = patient_id_to_case[int(patient_id)]
        test_file.write(f'{patient_id},{case_status}\n')#%% [markdown]


def patient_id_list_to_features(patient_list):
    features = list()
    classes = list()

    for patient_id in patient_list:
        # Get patient's case/control status
        patient_class = patient_id_to_case[int(patient_id)]

        # Iterate over potentially two samples
        sample_paths = data_path.glob(f'{patient_id}/*')
        for sample in sample_paths:
            mask_path = next(sample.glob('feature_masks/mask/*_mean.nii.gz')).as_posix()
            mask = nib.load(mask_path).get_data().T

            patient_features = dict()
            features_paths = sample.glob('feature_masks/feature/*.nii.gz')
            for feature_path in features_paths:

                # Load feature map and apply mask
                feature_map = np.nan_to_num(nib.load(feature_path.as_posix()).get_data().T)
                masked_feature_map = np.multiply(feature_map, mask)

                # Extract the feature name from its filename. Eg: norm_win_97_sliding_97_box_counting from
                # DPm.1.2.840.113681.2863050709.1375427076.3328_norm_win_97_sliding_97_box_counting.nii.gz
                feature_name = re.search('(?<=_).+(?=\.nii\.gz)', feature_path.name).group()  # noqa: W605
                patient_features[feature_name] = masked_feature_map

            features.append(patient_features)
            classes.append(patient_class)
    return (features, classes)


random.shuffle(training_patients)
train_features, train_classes = patient_id_list_to_features(training_patients)
test_features, test_classes = patient_id_list_to_features(testing_patients)


# Create an ordered list of feature names to ensure they are in the same
# order for every sample in the training and testing data
ordered_feature_names = sorted(train_features[0].keys())

# Save the data in 4D arrays
train_data = np.zeros((len(train_features), 34, 26, 29))
test_data = np.zeros((len(test_features), 34, 26, 29))

for sample_number, sample_dict in enumerate(train_features):
    for feature_number, feature_name in enumerate(ordered_feature_names):
        # Crop images to all be 34 x 26. Some are originally larger at 42 x 37
        train_data[sample_number, :, :, feature_number] = sample_dict[feature_name][0:34, 0:26]

for sample_number, sample_dict in enumerate(test_features):
    for feature_number, feature_name in enumerate(ordered_feature_names):
        # Crop images to all be 34 x 26. Some are originally larger at 42 x 37
        test_data[sample_number, :, :, feature_number] = sample_dict[feature_name][0:34, 0:26]

# Convert label lists to numpy arrays
train_classes = np.asarray(train_classes)
test_classes = np.asarray(test_classes)

epsilon = 1e-8

# Normalize the data across samples
# Combine the data and find the largest magnitude values for each feature
full_data = np.concatenate((train_data, test_data))
max_image = np.abs(full_data).max(axis=0)

train_data = np.divide(train_data, max_image + epsilon)
test_data = np.divide(test_data, max_image + epsilon)

# Normalize feature maps within samples so that the maximum value in each is 1.
# # This is the within-sample normalization that was performed
# # in the preprocessing code we received from the 2016 paper
for data_source in (train_data, test_data):
    for sample_number, sample in enumerate(data_source):
        for feature_number in range(29):
            feature_map = sample[:, :, feature_number]
            max_val = np.abs(feature_map).max()
            data_source[sample_number, :, :, feature_number] = np.divide(feature_map, max_val + epsilon)

# Save the data as pickled tuples of data, labels
training_set = (train_data, train_classes)
testing_set = (test_data, test_classes)

train_data_path = data_path.parent.joinpath('train_data.pkl')
test_data_path = data_path.parent.joinpath('test_data.pkl')

with open(train_data_path, 'wb') as f:
    pickle.dump(training_set, f)

with open(test_data_path, 'wb') as f:
    pickle.dump(testing_set, f)
