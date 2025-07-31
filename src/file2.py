import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

import dagshub

dagshub.init(repo_owner='itzayush21', repo_name='MLFlow', mlflow=True)

mlflow.set_tracking_uri("https://dagshub.com/itzayush21/MLFlow.mlflow/")

# Load Wine dataset
wine = load_wine()
X = wine.data
y = wine.target

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)

# Model parameters
max_depth = 10
n_estimators = 3

# Create experiment
mlflow.set_experiment('YT-MLOPS-Exp1')

with mlflow.start_run():
    rf = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators, random_state=42)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Log metrics and parameters
    mlflow.log_metric('accuracy', accuracy)
    mlflow.log_param('max_depth', max_depth)
    mlflow.log_param('n_estimators', n_estimators)

    # Confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=wine.target_names, yticklabels=wine.target_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")

    # Save plot
    cm_path = "Confusion-matrix.png"
    plt.savefig(cm_path)
    mlflow.log_artifact(cm_path)

    # Log the model as an artifact (not via registry)
    model_path = "Random-Forest-Model"
    mlflow.sklearn.save_model(rf, model_path)
    mlflow.log_artifact(model_path)

    print(f"Logged run with accuracy: {accuracy}")
