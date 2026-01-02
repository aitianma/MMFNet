import numpy as np
import pandas as pd
import torch
import torch_dct as dct
from matplotlib import pyplot as plt

# Read the CSV file
data = pd.read_csv("../dataset/ETTm1.csv")

# Extract the 'OT' column and convert to numpy array
# ot_data = np.array(data["LUFL"][:128])
ot_data = np.array(data["LUFL"][-1440:])

# Create the plot
plt.figure(figsize=(12, 6))

# Plot time domain data
plt.plot(ot_data, label="Time Domain")

# Compute and plot frequency domain data
freq_data = dct.dct(torch.from_numpy(ot_data)[-1440:]).numpy()/100
plt.plot(freq_data[:128], label="Freq Domain")

# Customize the plot
plt.title("Time Domain vs Frequency Domain")
plt.xlabel("Sample/Frequency")
plt.ylabel("Amplitude")
plt.legend()

# Display the plot
plt.show()

# Print the first 1440 values of the 'OT' column
print(ot_data)