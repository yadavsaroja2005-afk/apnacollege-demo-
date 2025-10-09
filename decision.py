# numpy and pandas initialization
import numpy as np
import pandas as pd
import graphviz
import matplotlib

# Loading the PlayTennis data
PlayTennis = pd.read_csv('bar.csv')

# Import LabelEncoder
from sklearn.preprocessing import LabelEncoder

# Encoding categorical features
Le = LabelEncoder()
PlayTennis['outlook'] = Le.fit_transform(PlayTennis['outlook'])
PlayTennis['temp'] = Le.fit_transform(PlayTennis['temp'])
PlayTennis['humidity'] = Le.fit_transform(PlayTennis['humidity'])
PlayTennis['windy'] = Le.fit_transform(PlayTennis['windy'])
PlayTennis['play'] = Le.fit_transform(PlayTennis['play'])

# Display the dataset
print(PlayTennis)

# Splitting features and label
y = PlayTennis['play']
x = PlayTennis.drop(['play'], axis=1)

# Fitting the Decision Tree model
from sklearn import tree
clf = tree.DecisionTreeClassifier(criterion='entropy')
clf = clf.fit(x, y)

# Visualizing the tree using tree.plot_tree
tree.plot_tree(clf)
print(tree.plot_tree)

# Exporting and viewing the decision tree using Graphviz
dot_data = tree.export_graphviz(clf, out_file=None)
graph = graphviz.Source(dot_data)
graph.render("decision_tree")
graph.view()

# Predictions
x_pred = clf.predict(x)
x_pred == y
print(x_pred == y)
