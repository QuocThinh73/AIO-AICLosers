import requests
import json

# Test the search API
response = requests.post(
    'http://localhost:5000/api/search',
    json={"query": "a person", "top_k": 2, "model_name": "clip"}
)

print("Status Code:", response.status_code)
print("Response:", json.dumps(response.json(), indent=2))
