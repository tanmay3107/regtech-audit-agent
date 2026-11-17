from locust import HttpUser, task, between
import json

class RegTechApiUser(HttpUser):
    # Wait between 1 and 3 seconds between simulated user actions
    wait_time = between(1.0, 3.0)

    @task(3)
    def audit_liquidity_reserves(self):
        """Simulate a standard FCA compliance audit query request."""
        payload = {
            "query": "Audit liquid capital reserves against FCA Handbook threshold conditions."
        }
        headers = {"Content-Type": "application/json"}
        
        with self.client.post("/api/v1/audit", json=payload, headers=headers, catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Failed with status code: {response.status_code}")

    @task(1)
    def health_check(self):
        """Simulate frequent health check polling by Kubernetes or Docker container probes."""
        self.client.get("/health")