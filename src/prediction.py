import pandas as pd
import pickle
import numpy as np

def load_model(model_path):
    """Load the trained model from the specified path."""
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model

def load_scaler(scaler_path):
    """Load the saved scaler for data normalization."""
    with open(scaler_path, 'rb') as file:
        scaler = pickle.load(file)
    return scaler

def predict_single(model, scaler, features):
    """
    Make a prediction for a single data point.

    Args:
        model: Trained model object.
        scaler: Scaler object for normalization.
        features: List of input features.

    Returns:
        prediction (int): Predicted class.
        probability (list): Prediction probabilities.
    """
    feature_names = ['co2_emissions', 'energy_consumption', 'tourism_activity']
    features_df = pd.DataFrame([features], columns=feature_names)
    features_normalized = scaler.transform(features_df)
    prediction = model.predict(features_normalized)
    probability = model.predict_proba(features_normalized)
    return prediction[0], probability[0]

def predict_batch(model, scaler, input_csv, output_csv):
    """
    Make predictions for a batch of data.

    Args:
        model: Trained model object.
        scaler: Scaler object for normalization.
        input_csv (str): Path to input CSV file.
        output_csv (str): Path to save predictions.

    Returns:
        None
    """
    data = pd.read_csv(input_csv)
    features = data[['co2_emissions', 'energy_consumption', 'tourism_activity']]
    features_normalized = scaler.transform(features)
    predictions = model.predict(features_normalized)
    probabilities = model.predict_proba(features_normalized)
    
    # Save predictions
    data['impact_prediction'] = predictions
    data['probability_high'] = probabilities[:, 1]
    data.to_csv(output_csv, index=False)
    print(f"Batch predictions saved to {output_csv}")

if __name__ == "__main__":
    # Load the model and scaler
    model_path = "models/tourism_model.pkl"
    scaler_path = "models/scaler.pkl"
    model = load_model(model_path)
    scaler = load_scaler(scaler_path)
    
    # Example single prediction
    features = [3.5, 2.1, 1.8]  # Example input
    prediction, probability = predict_single(model, scaler, features)
    print("Single Prediction:")
    print(f"Impact: {prediction}, Probability: {probability}")
    
    # Example batch prediction
    input_csv = "data/test/test_data.csv"
    output_csv = "data/test/test_predictions.csv"
    predict_batch(model, scaler, input_csv, output_csv)
