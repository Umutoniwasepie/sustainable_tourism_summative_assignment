import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle

def preprocess_data(input_csv, train_path, test_path, scaler_path):
    """
    Preprocess the data by normalizing features and splitting into training and testing datasets.
    
    Args:
        input_csv (str): Path to the input dataset CSV.
        train_path (str): Path to save the training dataset.
        test_path (str): Path to save the testing dataset.
        scaler_path (str): Path to save the scaler for normalizing features.
    """
    # Load the dataset
    data = pd.read_csv(input_csv)
    
    # Define features and target
    X = data[['co2_emissions', 'energy_consumption', 'tourism_activity']]
    y = data['impact']
    
    # Normalize features
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X)
    
    # Save the scaler
    with open(scaler_path, 'wb') as file:
        pickle.dump(scaler, file)
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.2, random_state=42)
    
    # Save preprocessed data
    train_data = pd.DataFrame(X_train, columns=['co2_emissions', 'energy_consumption', 'tourism_activity'])
    train_data['impact'] = y_train.values
    train_data.to_csv(train_path, index=False)
    
    test_data = pd.DataFrame(X_test, columns=['co2_emissions', 'energy_consumption', 'tourism_activity'])
    test_data['impact'] = y_test.values
    test_data.to_csv(test_path, index=False)
    
    print("Preprocessing completed. Train, test data, and scaler saved.")

if __name__ == "__main__":
    preprocess_data(
        "data/sustainable_tourism_dataset(1).csv",
        "data/train/train_data.csv",
        "data/test/test_data.csv",
        "models/scaler.pkl"
    )
