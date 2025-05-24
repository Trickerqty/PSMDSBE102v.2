import ray
import time
import requests
from serve import launch_service_from_checkpoint

# === Start Ray Serve with the best checkpoint ===
if ray.is_initialized():
    ray.shutdown()
ray.init()

# Start serving
checkpoint_path = "C:/Users/PC/ray_results/Movie_Sentiment_Tune_20250517_071012/TorchTrainer_2655ae35/checkpoint_000002"
launch_service_from_checkpoint(checkpoint_path)

print("Waiting for API to be ready...")
time.sleep(5)

url = "http://localhost:8000/predict/"
data = {"text": "I absolutely loved this movie. It was brilliant!"}
headers = {"Content-Type": "application/json"}

try:
    response = requests.post(url, json=data, headers=headers)
    response.raise_for_status()
    print("Prediction:", response.json())
except requests.RequestException as e:
    print("Request failed:", e)
    
