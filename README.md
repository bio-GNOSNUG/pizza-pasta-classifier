# pizza-pasta-classifier
A mini-project exploring a machine learning pipeline for the classification of pizza and pasta containing dishes.

# Introduction
Machine learning techniques capable of accurately identifying food items, groups, or contaminants have the potential to revolutionise the food industry at scale through improvements to food quality and safety during production. However, the development of a robust and generalisable machine learning pipeline poses challenges, including the need for extensive datasets and addressing variations in food presentation, such as differences in shape, color, and form.

Here, on a comparatively small dataset, we explore a machine learning pipeline to classify pizza and pasta containing dishes, two commonly consumed meals in the UK. Through the development of this pipeline, we aim to gain insights into the challenges posed by high-dimensional feature space, inconsistencies in food presentation, and the effectiveness of different functions in capturing complex relationships between features. Furthermore we will provide recommendations for further analysis to enhance our understanding of food image classifiers.

# Machine Learning pipeline

Stage 1: Data collection & pre-processing
The data sourced from GitHub ('https://github.com/MLEndDatasets/Yummy') contains 3250 JPEG files and 1 CSV file of 11 features per image, indexed by image filename. This data was collected by crowd sourcing with participants reporting dish name and ingredients as free text fields. Image data from CSV file containing the string 'pizza' or 'pasta' in the 'Ingredients' or 'Dish Name' is isolated and labelled. String labels are then encoded to numeric labels. Dataset is split into training and test sets with a 90:10 ratio, stratified by encoded label to ensure balanced representation of classes. The training dataset will undergo additional partitioning through k-fold cross-validation downstream, enabling the evaluation of Support Vector Machine (SVM) models. The outputs are 2 arrays of image filenames and encoded labels for both training and test datasets.

Stage 2: Data transformation
File paths of images used as input; images are made square and resized to 200x200 pixels. Images are converted from an RGB to CIE Lab* colour space, where only a* and b* channels are extracted as features. Sobel magnitude extracted as a shape feature. Principle component analysis (PCA) is applied to reduce the quantified 120000 feature space. All features will be standardised in scale prior to PCA application. The number of components is chosen as a trade-off between parameter reduction to minimise model overfitting risk and loss of image information for class discriminiation. This trade-off will be evaluated using the sum of the explained variance ratios from the PCA components.

Stage 3: Model training
Training data, composed of standardised PCA components and encoded labels, is split into 5 folds for cross validation of SVM models. SVM models with different kernel functions (linear, polynomial of degrees 2, 3, 4, and 5) are fitted to data and cross validated. Performance metrics of accuracy, precision and recall are evaluated to inform final model choice.

Stage 4: Model Evaluation Final SVM model is trained on full training dataset and evaluated on test dataset. The test dataset contains 10% of orginal image samples, with 15 standardised PCA component features and respective encoded labels. Performance metrics of accuracy, precision and recall will are analysed to evaluate quality of final SVM model.





