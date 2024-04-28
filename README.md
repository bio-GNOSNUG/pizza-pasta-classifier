# pizza-pasta-classifier
A mini-project exploring a machine learning pipeline for the classification of pizza and pasta containing dishes.

# Introduction
Machine learning techniques capable of accurately identifying food items, groups, or contaminants have the potential to revolutionise the food industry at scale through improvements to food quality and safety during production. However, the development of a robust and generalisable machine learning pipeline poses challenges, including the need for extensive datasets and addressing variations in food presentation, such as differences in shape, color, and form.

Here, on a comparatively small dataset, we explore a machine learning pipeline to classify pizza and pasta containing dishes, two commonly consumed meals in the UK. Through the development of this pipeline, we aim to gain insights into the challenges posed by high-dimensional feature space, inconsistencies in food presentation, and the effectiveness of different functions in capturing complex relationships between features. Furthermore we will provide recommendations for further analysis to enhance our understanding of food image classifiers.

# Machine learning pipeline

Stage 1: Data collection & pre-processing
The data sourced from GitHub ('https://github.com/MLEndDatasets/Yummy') contains 3250 JPEG files and 1 CSV file of 11 features per image, indexed by image filename. This data was collected by crowd sourcing with participants reporting dish name and ingredients as free text fields. Image data from CSV file containing the string 'pizza' or 'pasta' in the 'Ingredients' or 'Dish Name' is isolated and labelled. String labels are then encoded to numeric labels. Dataset is split into training and test sets with a 90:10 ratio, stratified by encoded label to ensure balanced representation of classes. The training dataset will undergo additional partitioning through k-fold cross-validation downstream, enabling the evaluation of Support Vector Machine (SVM) models. The outputs are 2 arrays of image filenames and encoded labels for both training and test datasets.

Stage 2: Data transformation
File paths of images used as input; images are made square and resized to 200x200 pixels. Images are converted from an RGB to CIE Lab* colour space, where only a* and b* channels are extracted as features. Sobel magnitude extracted as a shape feature. Principle component analysis (PCA) is applied to reduce the quantified 120000 feature space. All features will be standardised in scale prior to PCA application. The number of components is chosen as a trade-off between parameter reduction to minimise model overfitting risk and loss of image information for class discriminiation. This trade-off will be evaluated using the sum of the explained variance ratios from the PCA components.

Stage 3: Model training
Training data, composed of standardised PCA components and encoded labels, is split into 5 folds for cross validation of SVM models. SVM models with different kernel functions (linear, polynomial of degrees 2, 3, 4, and 5) are fitted to data and cross validated. Performance metrics of accuracy, precision and recall are evaluated to inform final model choice.

Stage 4: Model Evaluation Final SVM model is trained on full training dataset and evaluated on test dataset. The test dataset contains 10% of orginal image samples, with 15 standardised PCA component features and respective encoded labels. Performance metrics of accuracy, precision and recall will are analysed to evaluate quality of final SVM model.

# Data transformation 

All images are standardised to a square shape and resized to 200x200 pixels, ensuring uniformity in dimensions and fixed size inputs. Any variation between images we want to be due to inter-class variation within the feature space rather than differences in image size and shape. The choice of a 200x200 pixels balances retaining sufficient detail for feature extraction while avoiding excessive computational load. The input for this step comprises paths to .jpg image files from both the training and test datasets.

Images are transformed from the RGB colourspace to CIE Lab* colourspace. Notably, the CIE Lab* color space has demonstrated superior performance in classification models compared to models trained on RGB features (Du and Sun, 2005). The L* channel represents 'lightness' in an image, this feature is omitted to reduce dimensionality and avoid interpretation of variation in lighting across images. The a* channel represents values on a green to red axis while the b* channels represents a blue to yellow axis. Pizza and pasta are expected to present different colours. However, both food types are expected to exhibit diversity within each class due to variation in accompanying components i.e. pizza toppings, pasta sauces. This will pose a challenge to the classification model.

The sobel operator is a technique used in edge detection of images (Vincent and Folorunso, 2009; Roslan et al, 2017). Sobel magnitude is implemented as a shape feature in this model under the hypothesis that pizza and pasta generally exhibit distinct shapes with less intra-class than inter-class shape variation. Sobel gradients are computed from greyscale-converted image files (.jpg), and the Sobel magnitude is selected as a singular feature to reduce dimensionality.The resulting output is two-dimensional array to the length of the size of the input image (200x200 pixels). It is important to acknowlege differing presentation of food items within both classes may add to the complexity of this classification problem.

The feature space following extraction is large with 120000 dimensions. To minimise the risk of the model overfitting, Principle component analysis (PCA) is applied to reduce the number of features. All features are standardised in scale prior to PCA application due to its scale sensitivity; features with larger magnitudes can dominate the principal components, leading to biased results. The number of PCA components is chosen to balance parameter reduction with the loss of image information.

# Modelling

In this pipeline, 5 SVM models will be built, each with different kernel functions: linear, polynomial of degrees 2, 3, 4, and 5. The use of different kernel functions enables exploration of linear and non-linear decision boundaries to attempt to capture the complexity of the relationship between selected features. The performance of these models, evaluated through classification accuracy, will be compared in order to gain understanding of which function is most effective in distinguishing between pizza and pasta images using the selected features. SVMs are already widely implemented in image classification, including that of other food image classifiers (Zhang et al., 2023) and are effective in high dimensional feature space thus making them a reasonable candidate for this classification task.

# Methodology 

A 5-fold cross-validation method is used to train and evaluate the SVM models with different kernels (linear, polynomial of degrees 2, 3, 4, and 5). Each split is used to evaluate a different model. The training data inputted into the 5-fold cross-validation comprises 217 samples, each labeled with 1 for 'pizza' or 0 for 'pasta,' and is characterized by 15 features composed of the largest PCA components. The performance of each model is assessed through the evaluation of overall accuracy, precision, and recall. Accuracy represents the proportion of correctly predicted instances among the total instances. Precision quantifies the ratio of true positives (accurate predictions of pizza) to the sum of true positives and false positives (erroneous predictions of pizza). Recall is defined as the ratio of true positives to the sum of true positives and false negatives (incorrect predictions of pasta).

The final model selection is based on a balance between accuracy, precision, and recall. The chosen model is then trained on the complete 217-sample training set and subsequently tested on the remaining 25-sample test data. Both the training and test datasets are stratified to ensure a balanced representation of both pizza and pasta classes. This final model will be assessed using the mentioned statistics, in combination with a confusion matrix that presents class-specific accuracy rates.


# References
Du, C-J, and Sun, D-W 2005, 'Comparison of three methods for classification of pizza topping using different colour space transformations', Journal of Food Engineering, vol. 68, no. 3, pp. 277–287.

Vincent, O & Folorunso, O 2009, 'A descriptive algorithm for Sobel image edge detection', in Proceedings of the 2009 InSITE Conference, 2009, Informing Science Institute. Available at: link (accessed 4 December 2023).

Roslan, R, Nazery, NA, Jamil, N & Hamzah, R 2017, 'Color-based bird image classification using Support Vector Machine,' in 2017 IEEE 6th Global Conference on Consumer Electronics (GCCE), Nagoya, Japan, pp. 1-5. doi: 10.1109/GCCE.2017.8229492.

Zhang, Y., Deng, L., Zhu, H., Wang, W., Ren, Z., Zhou, Q., Lu, S., Sun, S., Zhu, Z., Gorriz, J. M., & Wang, S. (2023). 'Deep learning in food category recognition', Information Fusion, 98, 101859. https://doi.org/10.1016/j.inffus.2023.101859.

