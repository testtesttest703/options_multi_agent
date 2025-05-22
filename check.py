import json

# Read and load the JSON file
with open("models/AAPL_specialist_results.json", "r") as file:
    data = json.load(file)

# Now `data` is a Python dict or list depending on the file
print(data)
