
# pizza-pasta-classifier
# Author: Georgia Gunson

# Function: A machine learning pipeline for the classification of pizza and pasta containing dishes.

# Load libraries
import os
from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import spkit as sp

from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score
from sklearn.pipeline import make_pipeline

from skimage import color, filters, feature

from scipy.stats import skew
import cv2

from joblib import Parallel, delayed


# Set working directory
os.chdir('/content/drive/MyDrive/Data/MLEnd/yummy/')

# Read in the csv file
MLENDYD_df = pd.read_csv('/content/drive/MyDrive/Data/MLEnd/yummy/MLEndYD_image_attributes_benchmark.csv').set_index('filename')

# Get pizza data
pizza_df = MLENDYD_df.loc[(MLENDYD_df['Dish_name'].str.contains('pizza', case=False, na=False)) | (MLENDYD_df['Ingredients'].str.contains('pizza', case=False, na=False))].copy()
# Add pizza label
pizza_df['Pizza_Pasta'] = 'pizza'
print(len(pizza_df), "pizza rows")

# Get pasta data
pasta_df = MLENDYD_df.loc[(MLENDYD_df['Dish_name'].str.contains('pasta', case=False, na=False)) | (MLENDYD_df['Ingredients'].str.contains('pasta', case=False, na=False))].copy()
# Add pasta label
pasta_df['Pizza_Pasta'] = 'pasta'
print(len(pasta_df), "pasta rows")

# Join data together
MLEND_pp_df = pd.concat([pizza_df, pasta_df], ignore_index=False)

# Encode the pizza pasta label
# Encode the Rice =1  and Chips=0 instead of string labels
encoder = LabelEncoder()
MLEND_pp_df['Pizza_Pasta_encoded'] = encoder.fit_transform(MLEND_pp_df['Pizza_Pasta'])

# Get value counts, pasta = 0, pizza = 1
MLEND_pp_df['Pizza_Pasta_encoded'].value_counts()

# Split the data into training and 10% test sets, ensure balance of classes
# the 90% train data will go on to k-cross validation
train_data, test_data  = train_test_split(MLEND_pp_df['Pizza_Pasta_encoded'], test_size=0.1, stratify=MLEND_pp_df['Pizza_Pasta_encoded'].values, random_state=42)

# Get the image paths and labels of the training and test dataset
X_train_paths = train_data.reset_index()['filename']
X_test_paths  = test_data.reset_index()['filename']

Y_train = train_data.reset_index()['Pizza_Pasta_encoded']
Y_test  = test_data.reset_index()['Pizza_Pasta_encoded']


# Functions to make image proportions uniform (shape and size)
filepath_images = '/content/drive/MyDrive/Data/MLEnd/yummy/MLEndYD_images/'

# Custom function to make image shape square by including black pizels
def make_it_square(I, pad=0):
    # Get the dimensions of the input image
    # C represents the number of colour channels
    N, M, C = I.shape

    # Check if the number of rows (N) is greater than the number of columns (M)
    if N > M:
        # If N > M, pad the image to make it square by adding black pixels to the right
        Is = [np.pad(I[:,:,i], [(0,0), (0, N-M)], 'constant', constant_values=pad) for i in range(C)]
    else:
        # If M >= N, pad the image to make it square by adding black pixels to the bottom
        Is = [np.pad(I[:,:,i], [(0, M-N), (0,0)], 'constant', constant_values=pad) for i in range(C)]

    # Combine the padded channels and transpose the result
    return np.array(Is).transpose([1, 2, 0])

# Cutom function to resize the image to 200 by 200 pixel
def resize_img(I, size=[200, 200]):
    # Get the dimensions of the input image
    N, M, C = I.shape

    # Resize each channel of the image independently
    Ir = [sp.core.processing.resize(I[:,:,i], size) for i in range(C)]

    # Combine the resized channels and transpose the result
    return np.array(Ir).transpose([1, 2, 0])


# Custom function to calculate sobel magnitude
def calculate_sobel_magnitude(I):

  # Convert to grey image
  grey_image = cv2.cvtColor((I * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)

  # Calculate sobel gradients
  sobel_x = cv2.Sobel(grey_image, cv2.CV_64F, 1, 0, ksize=5)
  sobel_y = cv2.Sobel(grey_image, cv2.CV_64F, 0, 1, ksize=5)
  sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

  # Return the magnitude alone
  return sobel_magnitude

# Custom function to process each image
# Input is file path
# Outputs is numpy array of the histograms for a* and b* channels, sobel magnitude and entropy
def process_image(file):
    I = plt.imread(filepath_images + file)
    I = make_it_square(I, pad=0)
    I = resize_img(I, size=[200, 200])

    # Get sobel mag
    sobel_magnitude = calculate_sobel_magnitude(I)

    # Convert to CIE L*a*b* colourspace
    I_lab = color.rgb2lab(I / 255.0)
    I_ab = I_lab[:, :, 1:]

    # Flatten the a* and b* channels
    ab_flat = I_ab.reshape(-1, 2)

    # Concatenate a* and b* channels with sobel magnitude
    features = np.concatenate([ab_flat.flatten(), np.array([sobel_magnitude]).flatten()])

    return features


# Get image examples of both classes
example_pizza = MLEND_pp_df.loc[MLEND_pp_df['Pizza_Pasta_encoded']==1].index[0]
example_pizza = filepath_images + example_pizza
example_pasta = MLEND_pp_df.loc[MLEND_pp_df['Pizza_Pasta_encoded']==0].index[0]
example_pasta = filepath_images + example_pasta
example_images = [example_pizza, example_pasta]

# Custom function to compare original,resized, sobel magnitude, a* and b* channels images
def show_features(image_path):
    # Load original image
    original_image = cv2.imread(image_path)
    original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    # Make the image square
    square_image = make_it_square(original_image_rgb, pad=0)

    # Resize the image
    resized_image = resize_img(square_image, size=[200, 200])

    # Convert to greyscale
    grey_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

    # Calculate Sobel gradients
    sobel_x = cv2.Sobel(grey_image, cv2.CV_64F, 1, 0, ksize=5)
    sobel_y = cv2.Sobel(grey_image, cv2.CV_64F, 0, 1, ksize=5)
    sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

    # Convert to CIE Lab* colorspace
    lab_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2Lab)

    # Split into channels
    _, lab_a_channel, lab_b_channel = cv2.split(lab_image)

    # Show the images
    plt.figure(figsize=(20, 5))

    # Original
    plt.subplot(1, 5, 1)
    plt.imshow(original_image_rgb)
    plt.title('Original Image')
    plt.axis('off')

    # Resized
    plt.subplot(1, 5, 2)
    plt.imshow(resized_image)
    plt.title('Resized Image')
    plt.axis('off')

    # Sobel
    plt.subplot(1, 5, 3)
    plt.imshow(sobel_magnitude, cmap='gray')
    plt.title('Sobel Magnitude')
    plt.axis('off')

    # a*
    plt.subplot(1, 5, 4)
    plt.imshow(lab_a_channel, cmap='gray')
    plt.title('CIE Lab*: a* Channel')
    plt.axis('off')

    # b*
    plt.subplot(1, 5, 5)
    plt.imshow(lab_b_channel, cmap='gray')
    plt.title('CIE Lab*: b* Channel')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

# Show images
for image_path in example_images:
    show_features(image_path)
    
# Process images in parallel
num_cores = -1
# This takes ~13 minutes to run, not ideal (apologies) but fewer features (colour histograms, mean, std, skew) has worse performance
X_train_features = Parallel(n_jobs=num_cores)(delayed(process_image)(file) for file in X_train_paths)

# Convert the list of features to a numpy array
X_train_features = np.array(X_train_features)

# Display the shape of the training set
print("Training set shape:", X_train_features.shape)


# Extract features of test dataset
# Should take ~2 mins
X_test_features = Parallel(n_jobs=num_cores)(delayed(process_image)(file) for file in X_test_paths)

# Convert to numpy array
X_test_features = np.array(X_test_features)

# Display shape of test set
print("Test set shape:", X_test_features.shape)

# Standardize features
scaler = StandardScaler()
X_train_features_standardized = scaler.fit_transform(X_train_features)
X_test_features_standardized = scaler.transform(X_test_features)

# PCA
# Define components
num_components = 15
pca = PCA(n_components=num_components)
X_train_pca = pca.fit_transform(X_train_features_standardized)
X_test_pca = pca.transform(X_test_features_standardized)

# Get variance ratio per component for training set
var_ratio_train = pca.explained_variance_ratio_
print("Explained Variance Ratio (Training data):", var_ratio_train)

# Confirm new shape for training set
print("PCA-transformed Training set shape:", X_train_pca.shape)

# Get total amount of variance explained for training set
print("Sum of Explained Variance Ratio (Training data):", np.sum(var_ratio_train))

# Get variance ratio per component for test set
var_ratio_test = pca.explained_variance_ratio_
print("Explained Variance Ratio (Test data):", var_ratio_test)

# Confirm new shape for test set
print("PCA-transformed Test set shape:", X_test_pca.shape)

# Get total amount of variance explained for test set
print("Sum of Explained Variance Ratio (Test data):", np.sum(var_ratio_test))


# Create SVM models with different kernels
svm_linear = make_pipeline(SVC(kernel='linear'))
svm_poly2 = make_pipeline(SVC(kernel='poly', degree=2))
svm_poly3 = make_pipeline(SVC(kernel='poly', degree=3))
svm_poly4 = make_pipeline(SVC(kernel='poly', degree=4))
svm_poly5 = make_pipeline(SVC(kernel='poly', degree=5))

# List all models
models = [svm_linear, svm_poly2, svm_poly3, svm_poly4, svm_poly5]

# Create KFold cross-validation object
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Empty list to store results
results = []

# For each model
for model in models:
    # Create a pipeline with current SVM model
    pipeline = make_pipeline(model)

    # Create empty lists to store predicted vs actual vals
    y_true, y_pred = [], []

    # For each fold
    for train_index, test_index in kf.split(X_train_pca):
        # Get that fold's data
        X_train_fold, X_test_fold = X_train_pca[train_index], X_train_pca[test_index]
        y_train_fold, y_test_fold = Y_train[train_index], Y_train[test_index]

        # Fit model
        pipeline.fit(X_train_fold, y_train_fold)

        # Make predictions
        y_true.extend(y_test_fold)
        y_pred.extend(pipeline.predict(X_test_fold))

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')

    # Get model information
    model_name = model.steps[-1][1].kernel.capitalize()
    kernel = model.steps[-1][1].kernel
    degree = model.steps[-1][1].degree if kernel == 'poly' else ''

    # Collect all results
    results.append([model_name, kernel, degree, accuracy, precision, recall])

# Put all results in table to compare
columns = ['Model', 'Kernel', 'Degree', 'Accuracy', 'Precision', 'Recall']
results_df = pd.DataFrame(results, columns=columns)

# Display the DataFrame
display(results_df)

# Fit the linear SVM model on entire training set
svm_linear.fit(X_train_pca, Y_train)

# Make predictions on the PCA-transformed test set
Y_pred_test = svm_linear.predict(X_test_pca)

# Get metrics
accuracy_test = accuracy_score(Y_test, Y_pred_test)
precision_test = precision_score(Y_test, Y_pred_test, average='weighted')
recall_test = recall_score(Y_test, Y_pred_test, average='weighted')

print("Test set metrics:")
print(f"Accuracy: {accuracy_test:.4f}")
print(f"Precision: {precision_test:.4f}")
print(f"Recall: {recall_test:.4f}")

# Display confusion matrix and classification report
conf_matrix_test = confusion_matrix(Y_test, Y_pred_test)
class_report_test = classification_report(Y_test, Y_pred_test)

print("\nConfusion Matrix (Test set):")
print(conf_matrix_test)

print("\nClassification Report (Test set):")
print(class_report_test)





    
