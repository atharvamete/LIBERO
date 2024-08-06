import numpy as np

# Load the .npz file
data = np.load('rest_few.npz')

# Access the arrays
for key in data.files:
    print(f"Array for {key}:")
    print(data[key])
    print("--------------------")