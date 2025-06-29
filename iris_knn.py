# simple script to illustrate git concepts
# Scenario: A team is trying out different parameters for KNN
#      Some may push for a very different value of neighbours
#      DS Office has defined a minimum accuracy for the model
#      The unit test would enforce this
#      Unit test in pytest enviroment is executed as a git workflow


import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

import pickle

# Load the Iris dataset
iris = load_iris()

data = 'sample_data.csv'
column_headers = ['sepal-length', 'sepal-width',
                  'petal-length', 'petal-width', 'class']
df = pd.read_csv(data, names=column_headers)

data = df.values
X = data[:, 0:4]
y = data[:, 4]

# parameter search for neigbours - manual
neighb = 4

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create a K-Nearest Neighbors (KNN) classifier
knn = KNeighborsClassifier(n_neighbors=neighb)

# Train the model on the training data
knn.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = knn.predict(X_test)

# Evaluate model accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\n\nModel Accuracy: {accuracy:.2f}")

#now save the model
knnPickle = open('knnpickle_file', 'wb') 
pickle.dump(knn, knnPickle)  
knnPickle.close()
print("\n\n\n           Done\n")
