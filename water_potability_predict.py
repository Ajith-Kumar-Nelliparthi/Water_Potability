# Water Quality Prediction API Client
# ======================================

'''This client is designed to interact with the Water Quality Prediction API.
It provides a simple way to send water sample data to the API and retrieve
predictions about the water's potability.'''

import requests
url = 'http://localhost:5000/predict'

water_sample_id = 'water-230'

water_sample = {
    "ph": 5.821262,
    "hardness": 204.048890,
    "solids": 37174.005414,
    "chloramines": 7.867815,
    "sulfate": 329.019554,
    "conductivity": 466.783264,
    "organic_carbon": 13.988707,
    "trihalomethanes": 96.826961,
    "turbidity": 4.371079
}


response = requests.post(url, json=water_sample).json()
print(response)
if response['potability'] == True:
    print(f'Water sample id {water_sample_id} is potable water')
else:
    print(f'Water sample id {water_sample_id} is Non-potable water')

#* Predicting the potability of a water sample based on its chemical properties.
#* Automating the water quality testing process using the API.