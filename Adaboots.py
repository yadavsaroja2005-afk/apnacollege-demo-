import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_val_score

# Updated dataset URL (working GitHub link)
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"

# Column names
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']

# Load the dataset
dataframe = pd.read_csv(url, names=names)

# Extract features and target
array = dataframe.values
X = array[:, 0:8]
Y = array[:, 8]

# Random seed for reproducibility
seed = 7

# Number of trees in AdaBoost
num_trees = 30

# Initialize AdaBoost classifier
model = AdaBoostClassifier(n_estimators=num_trees, random_state=seed)

# Perform cross-validation and calculate mean accuracy
results = cross_val_score(model, X, Y)

print("Mean Accuracy:", results.mean())
