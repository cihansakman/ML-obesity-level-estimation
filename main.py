# -*- coding: utf-8 -*-
"""
Created on Sun Jun 13 00:15:05 2021

@author: sakma
"""

import matplotlib.pyplot as plt
import seaborn as sns

# prep
from sklearn.model_selection import train_test_split

# models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

# Library imports
import pandas as pd
import numpy as np
import warnings
import time

start_time = time.time()
warnings.filterwarnings("ignore")

# LOADING DATASET
data = pd.read_csv("./Data/ObesityDataSet_raw_and_data_sinthetic.csv")

# Show the percentage of null values
total_num_of_values = data.shape[0]
print(((data.isnull().sum()) / total_num_of_values) * 100)
# As we can see there are no null values in dataset


# We can get the count of unique values for each column.
for col in data.columns:
    print(col + ' ' + str(len(data[col].unique())) + ' ', (data[col]).dtype)
    if (col == "CAEC" or col == "CALC" or col == "MTRANS" or col == "NObeyesdad"):
        print(data[col].unique())
    print()

'''
There was a balancing problem before the author producing new variables with SMOTE and we can see the data
after applied SMOTE. There are still some unbalancing between Genders. For example for Obesity Type II and 
Obesity Type III it's extremely unbalanced.
'''

# This data obtained after producing new values from SMOTE
# Plot the obesity levels by gender
ax = sns.countplot(y="NObeyesdad", hue="Gender", data=data, palette="icefire")
plt.title("Obesity Category by Their Gender")
plt.show()

## PREPROCESSING

# ENCODING#
# In this secture we'll preprocess our string values into float with appropirate methods.

# Our binary columns are...
binary_cols = ['FAVC', 'SMOKE', 'SCC', 'family_history_with_overweight', 'Gender']

# We have only one nominal columns called MTRANS
# MTRANS = ['Public_Transportation' 'Walking' 'Automobile' 'Motorbike' 'Bike']
nominal_cols = ['MTRANS']

# We'll apply mapping to the ordinal columns manually.
# CAEC and CALC have the same orders
CAEC_CALC_ord_map = {'no': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3}
data['CAEC'] = data['CAEC'].map(CAEC_CALC_ord_map)
data['CALC'] = data['CALC'].map(CAEC_CALC_ord_map)

'''
NObeyesdad is our target column and has 7  object with order
   'Insufficient_Weight', 'Normal_Weight', 
   'Overweight_Level_I', 'Overweight_Level_II',
   'Obesity_Type_I', 'Obesity_Type_II', 'Obesity_Type_III  
But these are belongs to updated version of the paper. We'll convert it the first version which don't have Overweight_Level_II
'''
obesity_level_map = {'Insufficient_Weight': 0, 'Normal_Weight': 1, 'Overweight_Level_I': 2, 'Overweight_Level_II': 2,
                     'Obesity_Type_I': 3, 'Obesity_Type_II': 4, 'Obesity_Type_III': 5}

data['NObeyesdad'] = data['NObeyesdad'].map(obesity_level_map)

# correlation matrix for data before splitting the categories into new columns.
corrmat = data.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True, annot=True);
plt.show()

# We'll apply OneHotEncoding for nominal_cols with get_dummies.
for col in nominal_cols:
    data = pd.concat([data, pd.get_dummies(data[col], prefix=col + '_')], axis=1)
    data.drop([col], axis=1, inplace=True)

# We should convert String values into integer values
from sklearn import preprocessing

le = preprocessing.LabelEncoder()

# Use LabelEncoder for the binary columns
# Our binary columns are: ('FAVC', 'SMOKE', 'SCC', 'family_history_with_overweight')
for col in binary_cols:
    column = data.iloc[:, data.columns.get_loc(col):data.columns.get_loc(col) + 1].values
    data.iloc[:, data.columns.get_loc(col):data.columns.get_loc(col) + 1] = le.fit_transform(column[:, 0])

# Now we don't have any string values.


# There are some float points in categorical columns which computed by SMOTE. We'll round them.
SMOTE_columns = ['FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
data[SMOTE_columns] = data[SMOTE_columns].round()

# We can get the count of unique values for each column.
for col in data.columns:
    print(col + ' ' + str(len(data[col].unique())) + ' ', (data[col]).dtype)
    if (col == "CAEC" or col == "CALC" or col == "MTRANS" or col == "NObeyesdad"):
        print(data[col].unique())
    print()

data['Age'].plot.hist(grid=True, bins=10, rwidth=0.9,
                      color='#607c8e')
plt.title('Commute Times for 1,000 Commuters')
plt.xlabel('Counts')
plt.ylabel('Commute Time')
plt.grid(axis='y', alpha=0.75)
plt.show()

######################

# we will use the implementations provided by the imbalanced-learn Python library, which can be installed via pip as follows:
# sudo pip install imbalanced-learn
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score, precision_score, \
    recall_score
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import mutual_info_classif

y = data.NObeyesdad
X = data.drop(["NObeyesdad"], axis=1, inplace=False)

# We split the data into train(%80) and test(%20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=0)
##FEATURE SELECTION

# We'll compute some feature selection approaches to have better scores

# NUMERIC FEATURE SELECTION
numeric_columns = ['Age', 'Height', 'Weight']

X_train_numeric = X_train[numeric_columns]
X_test_numeric = X_test[numeric_columns]

X_train_categorical = X_train.drop(numeric_columns, axis=1, inplace=False)
X_test_categorical = X_test.drop(numeric_columns, axis=1, inplace=False)


# Select 5 columns with chi-squared test and Mutual Information Features from categorical values.
# After trying two different selection methods Mutual Information improved the score better.
def select_features_categorical(X_train, y_train, X_test):
    fs = SelectKBest(score_func=mutual_info_classif, k=5)
    fs.fit(X_train, y_train)
    X_train_fs = fs.transform(X_train)
    X_test_fs = fs.transform(X_test)
    return X_train_fs, X_test_fs, fs


# We'll use the ANOVA feature selection for the numerical values
def select_features_numerical(X_train, y_train, X_test):
    fs = SelectKBest(score_func=f_classif, k=3)
    fs.fit(X_train, y_train)
    X_train_fs = fs.transform(X_train)
    X_test_fs = fs.transform(X_test)
    return X_train_fs, X_test_fs


# feature selection
X_train_categorical_fs, X_test_categorical_fs, fs = select_features_categorical(X_train_categorical, y_train,
                                                                                X_test_categorical)
X_train_numerical_fs, X_test_numerical_fs = select_features_numerical(X_train_numeric, y_train, X_test_numeric)

# what are scores for the features
for i in range(len(fs.scores_)):
    print('Feature %d: %f' % (i, fs.scores_[i]))
# plot the scores
plt.bar([i for i in range(len(fs.scores_))], fs.scores_)
plt.show()

X_train = np.concatenate((X_train_categorical_fs, X_train_numerical_fs), axis=1)
X_test = np.concatenate((X_test_categorical_fs, X_test_numerical_fs), axis=1)


# Hyperparameter Tuning
# Let's split the training data and try to find best parameters.
# from sklearn.model_selection import GridSearchCV

# #Parameter Tuning for Logistic Regression
# #Solvers: ‘newton-cg’, ‘lbfgs’, ‘liblinear’, ‘sag’, ‘saga’

# parameters={"splitter":["best","random"],
#             "max_depth" : [1,3,5,7,9,11,12],
#             "min_samples_leaf":[1,2,3,4,5,6,7,8,9,10],
#             "min_weight_fraction_leaf":[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],
#             "max_features":["auto","log2","sqrt",None],
#             "max_leaf_nodes":[None,10,20,30,40,50,60,70,80,90] }


# classification_method = DecisionTreeClassifier()
# classification_method= GridSearchCV(classification_method , parameters, cv=10)
# classification_method.fit(X_train, y_train)

# means = classification_method.cv_results_['mean_test_score']
# stds = classification_method.cv_results_['std_test_score']
# for mean, std, params in zip(means, stds, classification_method.cv_results_['params']):
#     print("%0.3f (+/-%0.03f) for %r"
#           % (mean, std * 2, params))
# print("Best parameters:",classification_method.best_params_)


#####################################################################################################

# Function for obtaining the True Positives and False Positives
def counts_from_confusion(confusion):
    """
    Obtain TP and FP for each class in the confusion matrix
    """
    fp_total = 0
    # Iterate through classes and store the counts
    for i in range(confusion.shape[0]):
        tp = confusion[i, i]

        fp_mask = np.zeros(confusion.shape)
        fp_mask[:, i] = 1
        fp_mask[i, i] = 0
        fp = np.sum(np.multiply(confusion, fp_mask))

        fp_total += fp

    return fp_total


machine_learning_algorithms = (
    LogisticRegression(C=10, solver='newton-cg', multi_class='auto'),

    DecisionTreeClassifier(),
    GaussianNB(var_smoothing=0.0003511191734215131)

)

ml_names = ("Logistic Regression", "DecisionTree", "Naive Bayes")

# We'll keep AUC scores as dictionary
auc_scores = []

for ml, ml_name in zip(machine_learning_algorithms, ml_names):
    clf = ml
    clf.fit(X_train, y_train)

    y_preds = clf.predict_proba(X_test)
    preds = y_preds[:, 1]  # pred probability
    predict = clf.predict(X_test)
    conf_matrix = confusion_matrix(y_test, predict)

    FP = counts_from_confusion(conf_matrix) / y_test.shape[0]
    precision = precision_score(y_test, predict, average="weighted")
    recall = recall_score(y_test, predict, average="weighted")
    auc_scores.append({'class': ml_name,
                       'Precision': precision,
                       'Recall': recall,
                       'TP': recall,
                       'FP': FP})

    print(conf_matrix)
    print("Classification Report : for:", ml_name, "\n", classification_report(y_test, predict))
    print("*********************")

df = pd.DataFrame(data=auc_scores)
df.plot(x="class", y=["Precision", "Recall", "TP", "FP"], kind="bar")
from IPython.display import display

print()
print()
display(df)
print()
print()






