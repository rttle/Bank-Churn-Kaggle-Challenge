<img width="200" alt="image" src="https://github.com/rttle/Bank-Churn-Kaggle-Challenge/assets/143844181/dbbeb760-7ac3-4d53-84ce-a08071725da1">

# Bank Churn Kaggle Challenge
This repository holds an attempt to apply machine learning on customer data to predict whether or not they will continue being a bank customer, the data used is provided by Kaggle: https://www.kaggle.com/competitions/playground-series-s4e1/data.

## Overview
The Kaggle challenge tasks competitors with preparing a provided dataset to train a model to predict a customer’s Exited status, whether they continue being a customer or not. The dataset is made up of a mix of categorical and numerical features, 13 features in total. This repository shows how this challenge was addressed as a binary classification problem, with data cleaning and feature engineering applied to the provided dataset in preparation for training multiple models to predict Exited status. The best model was then used to predict Exited status on the test dataset provided which has the same features minus the target variable. XGBoost was the best model based on metrics, particularly looking at F1 and AUC-ROC score due to the dataset being imbalanced. 

## Summary of Workdone
### Data
- Data: Type: Tabular
  - Input: CSV file of features, output: Exited status in last column.
- Size: 165034 Customers, 13 features
- Instances (Train, Test, Validation Split): 
  - Train: 115523 customers (70%)
  - Test: 49,511 customers (30%)
  - Validation: 0
 
### Preprocessing / Clean up
**Missing Values.** There were no missing values to deal with in the provided dataset.

**Outliers.** Outliers in numerical columns were found using the IQR method, 1.5*IQR +/- Q3/Q1, respectively. The outliers were then replaced with either the 5th or 95th percentile.

**Feature Engineering.** Three additional features were created, the Age/Tenure Ratio, the Balance/Tenure Ratio, and the Balance/Age Ratio. Due to the denominator feature having 0s, NA and infinity values were introduced into the dataset. Those values were replaced with 0s.

**Encoding.** Some features were already encoded, while others were not. Thus, features such as Geography and Gender needed to be encoded. Additionally, One Hot Encoding had to be done in preparation for training the machine learning algorithms.

**Normalization.** The scale of the numerical columns were widely ranged, so normalization was applied.


### Data Visualization
The figure below is a summary table of the dataset. Note that the categorical/numerical classification is initial determinations and changed as a better understanding of the dataset was reached.

<img width="468" alt="image" src="https://github.com/rttle/Bank-Churn-Kaggle-Challenge/assets/143844181/c29be876-9dd4-4c3a-92f8-cfb365ba6f30">

The figure below is a portion of a pairplot, which was meant to show relationships of the features through the scatterplots and the distribution of the target variable classes, Exited. The pairplot also easily showed what should be categorical features when bars of data were shown in the scatterplots, like seen below for Tenure.

<img width="468" alt="image" src="https://github.com/rttle/Bank-Churn-Kaggle-Challenge/assets/143844181/a05bb1f4-c531-4c09-91dd-44f2680b56b4">

The figure below is histograms of the original numerical features in the Kaggle dataset. From the histograms, Age has separation in the distributions and looks like the best predictor feature. Balance also looks like it has potential with the center of its Exited distribution having higher frequencies than the Retained distribution.

<img width="468" alt="image" src="https://github.com/rttle/Bank-Churn-Kaggle-Challenge/assets/143844181/0157a124-b445-438a-ad73-1b0938eb670f">

The heatmap below shows that there was little concern for multicollinearity.

<img width="468" alt="image" src="https://github.com/rttle/Bank-Churn-Kaggle-Challenge/assets/143844181/0d0cca16-db4c-4f4f-9f03-f66c79cf8d71">

Below is a bar graph that shows the Kaggle dataset is imbalanced and must be considered when looking at model metrics.

<img width="468" alt="image" src="https://github.com/rttle/Bank-Churn-Kaggle-Challenge/assets/143844181/f75349ca-a36c-44a3-9831-9fb120f05dbb">


### Problem Formulation
- Input / Output
  - Input: 7 numerical features, 6 categorical features
  - Output: Exited Status (1 = Exited, 0 = Retained)
- Models 
  - K-Nearest Neighbors
  - Linear SVM
  - Decision Tree
  - Random Forest
  - XGBoost
- Hyperparameters
  - C
  - N Estimators
  - Max Depth
  - Max Features
  - Learning Rate

### Training
For initial models, the dataset was split 70% for training and 30% for testing. The models were basic implementation of instances, with some hyperparameter choices. Hyperparameter tuning was stunted due to limitations of the computer used. Cross validation was also done on the XGBoost model. Grid Search was used to perform hyperparameter tuning.
The most significant issue when training the models was lack of familiarity with machine learning causing implementation to be less streamline. For example, a function was implemented to train multiple models and give metric scores for each; however, it was only at the end that thoughts of cross validation and more extensive hyperparameter tuning were had.


### Performance Comparison
Multiple metrics were computed for the models, including: accuracy, precision, recall, F1, and AUC-ROC score. The score computed from the function within the classifier instance was also included. All were included in a comparison table; however, of importance is F1 and AUC-ROC score due to the imbalanced nature of the dataset.
Below is table of metrics for the models trained, along with one ROC curve for the best performing model. From the table, XGBoost performed the best in the most informing metrics, F1 and AUC-ROC.

<img width="468" alt="image" src="https://github.com/rttle/Bank-Churn-Kaggle-Challenge/assets/143844181/85abbbb4-b38c-4829-9c46-37918f9f2812">

<img width="468" alt="image" src="https://github.com/rttle/Bank-Churn-Kaggle-Challenge/assets/143844181/f22507b6-97ba-4593-bfa1-5d17d0921905">

### Conclusions
Of the models trained, XGBoost did the best at predicting Exited status of bank customers. For the Kaggle challenge, the XGBoost model’s prediction received a Private Score of 0.74835 and a Public Score of 0.74169. In the Private Score Leaderboard when the challenge was live, the Private Score of 0.74835 lands between 3215th and 3216th place.

### Future Work
To achieve better results, more data preprocessing should be done. For example, trying to clean up the EstimatedSalary column where there seemed to be a mix of annual salary and hourly wage. More thought and research can be done to see if more informative feature engineering is possible. For the models themselves, doing more extensive hyperparameter tuning could also aid in getting a better model. Also doing further reading into all the available models to see if other less known ones would fit this dataset better.

## How to reproduce results
To reproduce results, download the Churn dataset from Kaggle. Then ensure that the Churn_Preprocess.py file is downloaded from this repository and run the ML_Models.ipynb notebook also found in this repository.

## Overview of files in repository
- **Preprocessing_Visualization.ipynb:** Notebook that takes the provided Kaggle dataset and prepares it as a dataframe to be used to train models. Also creates tables and visualizations for data understanding.
- **ML_Models.ipynb:** Notebook that takes a dataset, trains multiple models, compares the models through a metrics table, and prepares a CSV file for submission for the Kaggle challenge.
- **Churn_Preprocess.py:** Module created to wrap all preprocessing done to the dataset in the preprocessing notebook 

## Data
Data is from the Kaggle Challenge, Binary Classification with a Bank Churn Dataset. https://www.kaggle.com/competitions/playground-series-s4e1

## Citations
- Walter Reade & Ashley Chow, Binary Classification with a Bank Churn Dataset, WWW.KAGGLE.COM, Jan. 1, 2024, https://kaggle.com/competitions/playground-series-s4e1.
- Check if dataframe contains infinity in Python – Pandas, WWW.GEEKSFORGEEKS.ORG, Dec. 26, 2020, https://www.geeksforgeeks.org/check-if-dataframe-contains-infinity-in-python-pandas/.
- David Fagbuyiro, Understanding the support vector machine (SVM) model, WWW.MEDIUM.COM, Jun. 26, 2023, https://medium.com/@davidfagb/understanding-the-support-vector-machine-svm-model-c8eb9bd54a97.


