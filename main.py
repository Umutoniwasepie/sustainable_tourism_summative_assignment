# FastAPI Application for Sustainable Tourism Analysis

from fastapi import FastAPI, HTTPException, UploadFile, File
import pandas as pd
from src.prediction import predict_single, predict_batch, load_model, load_scaler
import pickle
import os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware


# Initialize FastAPI app
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# File paths
MODEL_PATH = "models/tourism_model.pkl"
SCALER_PATH = "models/scaler.pkl"
TRAIN_PATH = "data/train/train_data.csv"

# Dynamic model and scaler loading
model = load_model(MODEL_PATH)
scaler = load_scaler(SCALER_PATH)

# Endpoints

@app.get("/")
async def root():
    """Welcome message for the API."""
    return {"message": "Welcome to the Sustainable Tourism API. Navigate to /frontend/index.html to access the frontend"}

# Serve HTML frontend
app.mount("/frontend", StaticFiles(directory="frontend", html=True), name="frontend")


@app.post("/predict_single/")
async def predict_single_endpoint(co2_emissions: float, energy_consumption: float, tourism_activity: float):
    """Predict the environmental impact for a single data point."""
    features = [co2_emissions, energy_consumption, tourism_activity]
    prediction, probability = predict_single(model, scaler, features)
    return {
        "prediction": int(prediction),
        "probability": probability.tolist()
    }


@app.post("/predict_batch/")
async def predict_batch_endpoint(file: UploadFile = File(...)):
    """Predict the environmental impacts for a batch of data."""
    input_csv = f"data/uploads/{file.filename}"
    output_csv = f"data/uploads/predictions_{file.filename}"

    os.makedirs("data/uploads", exist_ok=True)
    with open(input_csv, "wb") as f:
        f.write(await file.read())

    predict_batch(model, scaler, input_csv, output_csv)

    return {"message": "Batch predictions complete.", "output_file": output_csv}


@app.post("/retrain_model/")
async def retrain_model(file: UploadFile = File(...)):
    """
    Retrain the model with new data.

    Input:
    - file: A CSV file containing the new training data.

    Output:
    - Message: Confirmation of retraining success.
    - Evaluation metrics: Accuracy, Precision, Recall, F1 Score.
    """
    try:
        # Save the uploaded file temporarily
        file_location = f"data/uploads/{file.filename}"
        os.makedirs(os.path.dirname(file_location), exist_ok=True)
        with open(file_location, "wb") as f:
            f.write(await file.read())
        
        # Load the new data
        new_data = pd.read_csv(file_location)

        # Ensure required columns exist
        if not all(col in new_data.columns for col in ["co2_emissions", "energy_consumption", "tourism_activity", "impact"]):
            raise HTTPException(
                status_code=400,
                detail="The uploaded dataset must contain 'co2_emissions', 'energy_consumption', 'tourism_activity', and 'impact' columns."
            )
        
        # Load existing training data
        train_data = pd.read_csv("data/train/train_data.csv")
        combined_data = pd.concat([train_data, new_data], ignore_index=True)

        # Preprocess data
        X = combined_data[['co2_emissions', 'energy_consumption', 'tourism_activity']]
        y = combined_data['impact']
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        # Retrain the model
        model = LogisticRegression()
        model.fit(X_train, y_train)

        # Evaluate the model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # Save updated model and scaler
        with open("models/tourism_model.pkl", "wb") as model_file:
            pickle.dump(model, model_file)
        with open("models/scaler.pkl", "wb") as scaler_file:
            pickle.dump(scaler, scaler_file)

        # Update train data
        combined_data.to_csv("data/train/train_data.csv", index=False)

        return {
            "message": "Model retrained successfully.",
            "evaluation": {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/visualizations/")
async def visualizations():
    """Generate and return feature distribution visualizations."""
    df = pd.read_csv("data/sustainable_tourism_dataset(1).csv")

    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.figure(figsize=(14, 8))

    plt.subplot(1, 3, 1)
    sns.histplot(df['co2_emissions'], kde=True, color='blue')
    plt.title("CO2 Emissions Distribution")

    plt.subplot(1, 3, 2)
    sns.histplot(df['energy_consumption'], kde=True, color='red')
    plt.title("Energy Consumption Distribution")

    plt.subplot(1, 3, 3)
    sns.histplot(df['tourism_activity'], kde=True, color='green')
    plt.title("Tourism Activity Distribution")

    plt.tight_layout()
    plot_path = "data/uploads/feature_distributions.png"
    plt.savefig(plot_path)
    plt.close()

    return FileResponse(plot_path, media_type="image/png", filename="feature_distributions.png")


@app.get("/correlation_heatmap/")
async def correlation_heatmap():
    """Generate and return the correlation heatmap."""
    df = pd.read_csv("data/sustainable_tourism_dataset(1).csv")

    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.figure(figsize=(10, 8))
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)

    plot_path = "data/uploads/correlation_heatmap.png"
    plt.savefig(plot_path)
    plt.close()

    return FileResponse(plot_path, media_type="image/png", filename="correlation_heatmap.png")
