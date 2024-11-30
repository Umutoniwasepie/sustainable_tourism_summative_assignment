from locust import HttpUser, task, between
import random

class SustainableTourismUser(HttpUser):
    host = "http://localhost:8081"  # Set the correct base URL for your application

    wait_time = between(1, 5)  # Simulate user delay

    @task(3)
    def single_prediction(self):
        # Generate random inputs for single predictions
        co2_emissions = random.uniform(10.0, 300.0)
        energy_consumption = random.uniform(50.0, 500.0)
        tourism_activity = random.uniform(5.0, 100.0)

        self.client.post(
            "/predict_single/",
            params={
                "co2_emissions": co2_emissions,
                "energy_consumption": energy_consumption,
                "tourism_activity": tourism_activity
            }
        )

    @task(1)
    def batch_prediction(self):
        # Upload a batch test file
        with open("data/test/test_data.csv", "rb") as f:
            self.client.post(
                "/predict_batch/",
                files={"file": ("test_data.csv", f, "text/csv")}
            )
