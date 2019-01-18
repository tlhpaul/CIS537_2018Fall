import pathlib
import pickle
import re
import sys

import nibabel as nib
import numpy as np
import pandas as pd

# location = sys.argv[1]
location = 'data/secure/'

# Rename folders
data_path = pathlib.Path(location)

for feature_dir in data_path.glob('*/*/*/feature/'):
    parent_dir = feature_dir.parent
    parent_dir.rename(parent_dir.parent / 'feature_masks')

# Train/test split
# Read in case/control information
case_control_df = pd.read_excel('controlcase.xlsx')
patient_id_to_case = case_control_df[['DummyID', 'Class']].set_index('DummyID')['Class'].to_dict()

patients_list = list(patient_id_to_case.keys())

np.random.seed(0)

# Pick patients whose images will be in train/test sets
training_patients = np.random.choice(patients_list, replace=False, size=460)
testing_patients = [patient for patient in patients_list if patient not in training_patients]

# Prepare data for CNN
feature_masks = data_path.glob('*/*/feature_masks/')

train_features = list()
train_classes = list()
test_features = list()
test_classes = list()

for feature_mask_path in feature_masks:
    # Get patient's dummy id
    patient_id = int(feature_mask_path.parent.parent.name)

    # Get patient's case/control status
    patient_class = patient_id_to_case[patient_id]

    # Load the sample's mask
    mask_path = list((feature_mask_path / 'mask').glob('*_mean.nii.gz'))[0].as_posix()
    mask = nib.load(mask_path).get_data().T

    # Iterate through all feature maps. Load and apply mask to each.
    patient_features = dict()
    features_paths = (feature_mask_path / 'feature').glob('*.nii.gz')
    for feature_path in features_paths:

        # Load feature map and apply mask
        feature_map = np.nan_to_num(nib.load(feature_path.as_posix()).get_data().T)
        masked_feature_map = np.multiply(feature_map, mask)

        # Normalize feature maps so that the maximum value in each is 1
        epsilon = 1e-8
        normalized_masked_feature_map = masked_feature_map \
            / (np.abs(masked_feature_map).max() + epsilon)

        # Extract the feature name from its filename. Eg: norm_win_97_sliding_97_box_counting from
        # DPm.1.2.840.113681.2863050709.1375427076.3328_norm_win_97_sliding_97_box_counting.nii.gz
        feature_name = re.search('(?<=_).+(?=\.nii\.gz)', feature_path.name).group()  # noqa: W605
        patient_features[feature_name] = normalized_masked_feature_map

    # Get patient's train/test category and add the data in the corresponding list
    is_test = patient_id in testing_patients
    if is_test:
        test_features.append(patient_features)
        test_classes.append(patient_class)
    else:
        train_features.append(patient_features)
        train_classes.append(patient_class)

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

training_set = (train_data, train_classes)
testing_set = (test_data, test_classes)

train_data_path = data_path.parent.joinpath('train_data.pkl')
test_data_path = data_path.parent.joinpath('test_data.pkl')

with open(train_data_path, 'wb') as f:
    pickle.dump(training_set, f)

with open(test_data_path, 'wb') as f:
    pickle.dump(testing_set, f)
