import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

# Load the dataset
data = load_breast_cancer()

# Select only 5 features
selected_features = ['mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness']
X = pd.DataFrame(data.data, columns=data.feature_names)
X = X[selected_features]  # Filter to include only selected features
y = data.target  # Target (0: Malignant, 1: Benign)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Logistic Regression Model
lr_model = LogisticRegression(max_iter=5000)
lr_model.fit(X_train, y_train)

# Train Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Test accuracy
lr_acc = accuracy_score(y_test, lr_model.predict(X_test))
rf_acc = accuracy_score(y_test, rf_model.predict(X_test))

print(f"Logistic Regression Accuracy (5 features): {lr_acc * 100:.2f}%")
print(f"Random Forest Accuracy (5 features): {rf_acc * 100:.2f}%")

# Save the better model
best_model = rf_model if rf_acc > lr_acc else lr_model
with open('model.pkl', 'wb') as f:
    pickle.dump(best_model, f)
print("Model training complete. Model saved as 'model.pkl'")
