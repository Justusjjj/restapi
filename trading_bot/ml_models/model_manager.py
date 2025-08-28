import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import logging

"""
This module handles the machine learning model's lifecycle:
- Training the model on processed data.
- Evaluating its performance.
- Saving the trained model to a file.
- Loading a model from a file.
"""

def train_model(df):
    """
    Trains a RandomForestClassifier model on the provided DataFrame.

    :param df: The feature-engineered DataFrame.
    :return: The trained model, and the test sets (X_test, y_test) for evaluation.
    """

    # --- 1. Define Features (X) and Target (y) ---
    # We want to use our engineered features to predict the target.
    # Explicitly select feature columns to avoid data leakage.
    feature_cols = [col for col in df.columns if col not in ['time', 'open', 'high', 'low', 'close', 'target']]
    features = df[feature_cols]
    target = df['target']

    # Ensure all feature columns are numeric, fill NaNs just in case
    features = features.apply(pd.to_numeric, errors='coerce').fillna(0)

    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42, stratify=target
    )

    logging.info(f"Training data shape: {X_train.shape}")
    logging.info(f"Testing data shape: {X_test.shape}")

    # --- 2. Train the Model ---
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    logging.info("Training RandomForestClassifier...")
    model.fit(X_train, y_train)
    logging.info("Model training complete.")

    return model, X_test, y_test

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the trained model and prints a performance report.

    :param model: The trained scikit-learn model.
    :param X_test: The testing features.
    :param y_test: The testing target.
    """
    logging.info("Evaluating model performance...")
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)

    print("\n--- Model Evaluation Report ---")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(report)
    print("-----------------------------")

def save_model(model, filepath):
    """
    Saves the trained model to a file using joblib.

    :param model: The trained model object.
    :param filepath: The path to save the model file to.
    """
    try:
        joblib.dump(model, filepath)
        logging.info(f"Model successfully saved to {filepath}")
    except Exception as e:
        logging.error(f"Error saving model: {e}")

def load_model(filepath):
    """
    Loads a model from a file.

    :param filepath: The path to the model file.
    :return: The loaded model object.
    """
    try:
        model = joblib.load(filepath)
        logging.info(f"Model successfully loaded from {filepath}")
        return model
    except FileNotFoundError:
        logging.error(f"Model file not found at {filepath}")
        return None
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        return None
