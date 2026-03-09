import requests

url = "http://localhost:5000/predict"

data = {
 "description": "Senior ML Engineer",
 "experience_level": "Senior",
 "is_remote": True,
 "skills_count": 5
}

response = requests.post(url, json=data)

print(response.json())