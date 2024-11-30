import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle

def train_model(train_csv, model_path):
    """
    Train a Logistic Regression model and save it.

    Args:
        train_csv (str): Path to the training dataset CSV.
        model_path (str): Path to save the trained model.
    """
    # Load training data
    train_data = pd.read_csv(train_csv)
    X_train = train_data[['co2_emissions', 'energy_consumption', 'tourism_activity']]
    y_train = train_data['impact']
    
    # Train Logistic Regression model
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    # Save the model
    with open(model_path, 'wb') as file:
        pickle.dump(model, file)
    
    print(f"Model training complete. Model saved at {model_path}")

def evaluate_model(test_csv, model_path):
    """
    Evaluate the trained model and display evaluation metrics.

    Args:
        test_csv (str): Path to the testing dataset CSV.
        model_path (str): Path to the saved model.
    """
    # Load test data
    test_data = pd.read_csv(test_csv)
    X_test = test_data[['co2_emissions', 'energy_consumption', 'tourism_activity']]
    y_test = test_data['impact']
    
    # Load the trained model
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print("Model Evaluation Metrics:")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")

if __name__ == "__main__":
    train_csv = "data/train/train_data.csv"
    test_csv = "data/test/test_data.csv"
    model_path = "models/tourism_model.pkl"
    
    train_model(train_csv, model_path)
    evaluate_model(test_csv, model_path)
