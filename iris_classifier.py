import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the Iris dataset from sklearn
from sklearn.datasets import load_iris

# Load dataset
iris = load_iris()
data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
data['species'] = iris.target

# Explore the data
print("Data Overview:")
print(data.head())

# Data Preprocessing
# Encode the 'species' column (already encoded as 0, 1, 2 in the dataset, but we can label it)
label_encoder = LabelEncoder()
data['species'] = label_encoder.fit_transform(data['species'])

# Split the data into features and target
X = data.drop('species', axis=1)
y = data['species']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model 1: Logistic Regression
logreg_model = LogisticRegression(max_iter=200)
logreg_model.fit(X_train, y_train)

# Model 2: Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predictions
logreg_pred = logreg_model.predict(X_test)
rf_pred = rf_model.predict(X_test)

# Model Evaluation
logreg_accuracy = accuracy_score(y_test, logreg_pred)
rf_accuracy = accuracy_score(y_test, rf_pred)

# Display results
print("\nLogistic Regression Accuracy:", logreg_accuracy)
print("\nRandom Forest Accuracy:", rf_accuracy)

# Confusion Matrix and Classification Report for Logistic Regression
print("\nLogistic Regression Classification Report:")
print(classification_report(y_test, logreg_pred))
print("\nLogistic Regression Confusion Matrix:")
print(confusion_matrix(y_test, logreg_pred))

# Confusion Matrix and Classification Report for Random Forest
print("\nRandom Forest Classification Report:")
print(classification_report(y_test, rf_pred))
print("\nRandom Forest Confusion Matrix:")
print(confusion_matrix(y_test, rf_pred))

# Plot Confusion Matrix for Random Forest (Optional)
plt.figure(figsize=(6,6))
sns.heatmap(confusion_matrix(y_test, rf_pred), annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.savefig("images/rf_confusion_matrix.png")
plt.show()

# Plot Confusion Matrix for Logistic Regression (Optional)
plt.figure(figsize=(6,6))
sns.heatmap(confusion_matrix(y_test, logreg_pred), annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title("Logistic Regression Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.savefig("images/logreg_confusion_matrix.png")
plt.show()


