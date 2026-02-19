"""
Practice with Standard Scaling
"""

from sklearn.preprocessing import StandardScaler
import numpy as np

# Exercise 2a: Scale different ranges
print("=== Exercise 2a: Scaling Different Ranges ===")

# House data with very different scales
data = np.array(
    [
        [1500, 3, 10],  # sqft, bedrooms, age
        [2000, 4, 5],
        [1000, 2, 15],
        [2500, 5, 2],
    ]
)

print("Original data:")
print("sqft  bedrooms  age")
print(data)

scaler = StandardScaler()
scaled = scaler.fit_transform(data)

print("\nScaled data:")
print("sqft  bedrooms  age")
print(scaled)

print("\nMeans after scaling:")
print(scaled.mean(axis=0))  # Should be ~0

print("\nStd devs after scaling:")
print(scaled.std(axis=0))  # Should be ~1

# YOUR TASK: Why is scaling important?
print("\n💡 Notice: All features now have similar ranges!")

# Exercise 2b: What scaling does to predictions
print("\n=== Exercise 2b: Impact on Distance ===")

# Without scaling
house1 = np.array([2000, 3, 10])
house2 = np.array([2100, 4, 11])

distance_unscaled = np.sqrt(np.sum((house1 - house2) ** 2))
print(f"Distance without scaling: {distance_unscaled:.2f}")
# Dominated by sqft!

# With scaling
house1_scaled = scaler.transform([house1])[0]
house2_scaled = scaler.transform([house2])[0]

distance_scaled = np.sqrt(np.sum((house1_scaled - house2_scaled) ** 2))
print(f"Distance with scaling: {distance_scaled:.2f}")
# All features contribute equally!

print("\n✅ Scaling makes all features equally important!")
