# sustainable_tourism_summative_assignment
This Repository contains Semester 1, Year 3 Summative Assignment

![option3](https://github.com/user-attachments/assets/446ba874-f4b6-4d22-94b7-fa1852f88718)

## Project Description
The **Sustainable Tourism Project** is a machine learning application that predicts the environmental impact of tourism activities based on factors such as CO2 emissions, energy consumption, and tourism activity. The project is built using a **FastAPI**-based backend and leverages a **Logistic Regression** model for making predictions.

![option2](https://github.com/user-attachments/assets/b0b6a20a-2ac8-4177-84f6-56cf4d4a0af1)


### Model Overview
The machine learning model predicts the environmental impact using the following features:
- **CO2 Emissions**
- **Energy Consumption**
- **Tourism Activity**

The app is designed to predict the sustainability of tourism activities based on these factors, helping stakeholders make informed decisions.

### Technologies Used:
- **Backend**: FastAPI (for building the REST API)
- **Machine Learning**: Scikit-learn (Logistic Regression)
- **Containerization**: Docker (for deployment)
- **Load Testing**: Locust (for simulating traffic and load testing)
- **Deployment**: Render (for cloud deployment)

## Setup Instructions

### Prerequisites
Ensure you have the following installed on your local machine:
- **Docker Desktop** (with Docker Compose support)
- **Git** (optional, for cloning the project repository)
- **Python 3.9 or higher** (for running the backend, if you prefer not to use Docker)

### Step-by-step Setup

#### Option 1: Using Docker Compose
1. Clone the Repository:
   ```bash
   git clone https://github.com/Umutoniwasepie/sustainable_tourism_summative_assignment.git
   cd sustainable_tourism_summative_assignment

2. Navigate to the Project Directory: Make sure you are in the root directory of the project, where docker-compose.yml is located.

3. Build and Run the Containers: Use Docker Compose to build and start both the frontend and backend services:
   ```bash
   docker-compose up --build

This will build the frontend and backend images according to the Dockerfile definitions and start the services. The frontend is accessible at **http://localhost:8081**, and the backend at **http://localhost:8000**.

#### Option 2: Running the Backend Locally

1. Install Dependencies: Run the following command to install the necessary Python dependencies:
   ```bash
   pip install -r requirements.txt

2. Run the Backend: Start the FastAPI server with:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000
   

## Deployment Package

### Public URL
The Sustainable Tourism API is deployed and accessible via the following public URL 
- [Sustainable Tourism Analysis](https://sustainable-tourism-summative-analysis.onrender.com/frontend/index.html)

### Docker Image
To run the application locally with the official Docker image, use:
1. Pull the image:
   ```bash
   docker pull umutoniwasepie/sustainable-tourism-app:latest

2. Run the container:
   ```bash
   docker run -p 8000:8000 umutoniwasepie/sustainable-tourism-app:latest

## API Endpoints

### /predict_single/
Method: POST
Description: Predict the environmental impact for a single data point (CO2 emissions, energy consumption, tourism activity).

Example Request:
    
    curl -X POST "http://localhost:8000/predict_single/" -d '{"co2_emissions": 200, "energy_consumption": 50, "tourism_activity": 10}' -H "Content-Type: application/json"

Response:
    
    {
    "prediction": 1,
    "probability": [0.25, 0.75]
    }

### /predict_batch/
Method: POST
Description: Upload a CSV file for batch predictions.

Example Request:
    
    curl -X POST "http://localhost:8000/predict_batch/" -F "file=@data.csv"

### /retrain_model/
Method: POST

Description: Upload a new dataset to retrain the model.

Example Request:
    
    curl -X POST "http://localhost:8000/retrain_model/" -F "file=@new_data.csv"

Response:
    
    {
    "message": "Model retrained successfully.",
    "evaluation": {
        "accuracy": 0.81,
        "precision": 0.79,
        "recall": 0.75,
        "f1_score": 0.76
        }
    }

## Video Demo

A video demo of the application showing how it works and how to interact with the API.

- [Watch the Demo on YouTube](https://youtu.be/MgphsdCNIpY)
  

## Locust Load Testing Results

The application was stress-tested using **Locust** to simulate a flood of requests. Below are the results with varying user loads:

### Results for 3 Scales

| Users | Requests per second (RPS) | Ramp up(users started/sec) |Failures |
|-------|---------------------------|----------------------------|---------|
| 50    | 16.5                      | 2                          |0%       |
| 100   | 32.3                      | 10                         |0%       |
| 200   | 65.6                      | 20                         |0%       |

### Results for 5 Scales

| Users | Requests per second (RPS) | Ramp up(users started/sec) |Failures |
|-------|---------------------------|----------------------------|---------|
| 50    | 17.4                      |  2                         |0%       |
| 100   | 33.9                      |  10                        |0%       |
| 200   | 64.9                      |  20                        |0%       |
| 250   | 65.9                      |  30                        |0%       |

### Example chart on 5 users:

![5users](https://github.com/user-attachments/assets/742e2143-ee44-43e0-9b40-de1962ba1fa6)


### Report Access

To access a detailed report for one of the requests (e.g., the request made with 200 users), you can find the Locust report in the `locust_reports` folder. Here's an example of where you can find the report:

- **Path**: `locust_report/report_250_users.html`, It contains additional information about the individual requests, response times, and success rates for the test scenario with 200 users.


**Wish you all the best, ciao!!**

