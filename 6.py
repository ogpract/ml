#Aim: To implement Random Forest Algorithm

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn import datasets
from sklearn import metrics

# Load the Iris dataset
iris = datasets.load_iris()

# Print target names, feature names, and sample data
print(iris.target_names)
print(iris.feature_names)
print(iris.data[0:5])
print(iris.target)

# Create a DataFrame with the dataset
data = pd.DataFrame({
    'sepal length': iris.data[:, 0],
    'sepal width': iris.data[:, 1],
    'petal length': iris.data[:, 2],
    'petal width': iris.data[:, 3],
    'species': iris.target
})

# Display first few rows of the dataset
data.head()

# Define features (X) and labels (y)
x = data[['sepal length', 'sepal width', 'petal length', 'petal width']]  # Features
y = data['species']  # Labels

# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

# Create and train the RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100)  # Corrected parenthesis
clf.fit(x_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(x_test)

# Print accuracy of the model
print("Accuracy: ", metrics.accuracy_score(y_test, y_pred))

# Predict species for a new sample
ans = clf.predict([[3, 5, 4, 2]])

# Output the predicted species
if ans[0] == 0:
    print('setosa')
elif ans[0] == 1:
    print('versicolor')
else:
    print('virginica')