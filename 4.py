import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

# Create a sample dataset
data = {
    "Outlook": ["sunny", "sunny", "overcast", "rainy", "rainy", "rainy", "overcast", "sunny", "sunny", "rainy"],
    "Temperature": ["hot", "hot", "hot", "mild", "cool", "cool", "cool", "mild", "cool", "mild"],
    "Humidity": ["high", "high", "high", "high", "normal", "normal", "normal", "high", "normal", "high"],
    "Windy": [False, True, False, False, False, True, True, False, False, True],
    "PlayTennis": ["no", "no", "yes", "yes", "yes", "no", "yes", "no", "yes", "yes"]
}

# Convert the dataset to a pandas DataFrame
df = pd.DataFrame(data)

# Encode categorical variables (convert text to numbers)
df_encoded = pd.get_dummies(df[["Outlook", "Temperature", "Humidity", "Windy"]])

# Target variable (PlayTennis) -> convert yes/no to 1/0
df['PlayTennis'] = df['PlayTennis'].map({"yes": 1, "no": 0})

# Prepare X (features) and y (target)
X = df_encoded
y = df['PlayTennis']

# Train a decision tree classifier
clf = DecisionTreeClassifier(criterion='entropy', max_depth=3)
clf = clf.fit(X, y)

# Plot the decision tree
plt.figure(figsize=(12,8))
tree.plot_tree(clf, feature_names=X.columns, class_names=["No", "Yes"], filled=True, rounded=True)
plt.show()