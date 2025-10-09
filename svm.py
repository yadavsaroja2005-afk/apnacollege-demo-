from warnings import filterwarnings
filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf

from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score, cross_val_predict
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import scale
from sklearn import model_selection, preprocessing
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    classification_report,
    confusion_matrix, accuracy_score
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, BaseEnsemble, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import SVC, LinearSVC

import time
from matplotlib.colors import ListedColormap

from xgboost import XGBRegressor
from skompiler import skompile
from lightgbm import LGBMRegressor

# Display settings
pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)

# Load dataset
df = pd.read_csv(r"diabetes.csv")
print(df.head())
print(df.shape)
print(df.describe())

# Feature and target separation
x = df.drop("Outcome", axis=1)
y = df["Outcome"]  # We will predict Outcome (diabetes)

# Train-test split
x_train = x.iloc[:600]
x_test = x.iloc[600:]
y_train = y[:600]
y_test = y[600:]

print("x_train shape: ", x_train.shape)
print("x_test shape:", x_test.shape)
print("y_train shape: ", y_train.shape)
print("y_test shape:", y_test.shape)

# SVC Model (Linear Kernel)
support_vector_classifier = SVC(kernel="linear").fit(x_train, y_train)
support_vector_classifier

# Default C
support_vector_classifier.C
support_vector_classifier

# Predictions
y_pred = support_vector_classifier.predict(x_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)
print("Our Accuracy is: ", (cm[0][0] + cm[1][1]) / (cm[0][0] + cm[1][1] + cm[0][1] + cm[1][0]))
accuracy_score(y_test, y_pred)
print(classification_report(y_test, y_pred))
support_vector_classifier

# Cross-validation
accuracies = cross_val_score(estimator=support_vector_classifier, X=x_train, y=y_train, cv=10)
print("Average Accuracy: {:.2f} %".format(accuracies.mean() * 100))
print("Standard Deviation of Accuracies: {:.2f}%".format(accuracies.std() * 100))

# Predictions preview
support_vector_classifier.predict(x_test)[:10]
print(np.array([0, 0, 0, 1, 1, 0, 1, 0, 1, 0]))

# Hyperparameter tuning
svm_params = {"C": np.arange(1, 20)}
svm = SVC(kernel='linear')
svm_cv = GridSearchCV(svm, svm_params, cv=8)

start_time = time.time()
svm_cv.fit(x_train, y_train)
elapsed_time = time.time() - start_time
print(f"Elapsed time for support vector regression cross-validation: {elapsed_time:.3f} seconds")

# Best score and parameters
svm_cv.best_score_
svm_cv.best_params_  # {'C': 2}

# Final tuned model
svm_tuned = SVC(kernel='linear', C=2).fit(x_train, y_train)
svm_tuned

# Predictions with tuned model
y_pred = svm_tuned.predict(x_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)
print("Our Accuracy is:", (cm[0][0] + cm[1][1]) / (cm[0][0] + cm[1][1] + cm[0][1] + cm[1][0]))
accuracy_score(y_test, y_pred)
print(classification_report(y_test, y_pred))
