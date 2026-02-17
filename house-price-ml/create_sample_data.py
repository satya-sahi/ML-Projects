"""
Generate sample house price data for training
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)

# Define neighborhoods with their base price multipliers
neighborhoods = {
    "Downtown": 1.5,
    "Suburb_North": 1.2,
    "Suburb_South": 1.0,
    "Suburb_East": 1.1,
    "Rural": 0.8,
}

# Property types
property_types = ["Single_Family", "Condo", "Townhouse"]

# Generate 10 sample houses
data = []

for i in range(10):
    # Basic features
    bedrooms = np.random.randint(2, 6)
    bathrooms = np.random.choice([1.0, 1.5, 2.0, 2.5, 3.0, 3.5])
    sqft = np.random.randint(800, 3500)
    lot_size = np.random.randint(2000, 10000) if np.random.random() > 0.3 else None
    year_built = np.random.randint(1970, 2023)
    property_type = np.random.choice(property_types)
    location = np.random.choice(list(neighborhoods.keys()))

    # Additional features
    garage_spaces = np.random.randint(0, 4)
    has_pool = np.random.choice([True, False], p=[0.2, 0.8])
    has_fireplace = np.random.choice([True, False], p=[0.4, 0.6])
    renovated = np.random.choice([True, False], p=[0.3, 0.7])

    # Distance features (in miles)
    distance_to_school = round(np.random.uniform(0.5, 5.0), 1)
    distance_to_transit = round(np.random.uniform(0.2, 3.0), 1)
    distance_to_shopping = round(np.random.uniform(0.3, 4.0), 1)

    # Calculate base price (price per sqft * sqft)
    base_price_per_sqft = 150

    # Apply multipliers
    price = base_price_per_sqft * sqft
    price *= neighborhoods[location]

    # Adjustments
    if property_type == "Single_Family":
        price *= 1.1
    elif property_type == "Condo":
        price *= 0.9

    # Age adjustment
    age = 2024 - year_built
    if age < 10:
        price *= 1.15
    elif age > 30:
        price *= 0.85

    # Feature adjustments
    price += garage_spaces * 15000
    if has_pool:
        price += 30000
    if has_fireplace:
        price += 10000
    if renovated:
        price *= 1.2

    # Add some random variation (-5% to +5%)
    price *= np.random.uniform(0.95, 1.05)

    # Round to nearest 1000
    price = round(price / 1000) * 1000

    # Listing date (random date in last 6 months)
    days_ago = np.random.randint(1, 180)
    listing_date = datetime.now() - timedelta(days=days_ago)

    # Days on market
    days_on_market = np.random.randint(5, 90)

    # Sold date
    sold_date = listing_date + timedelta(days=days_on_market)

    # Create property ID
    property_id = f"PROP_{i + 1:03d}"

    data.append(
        {
            "property_id": property_id,
            "location": location,
            "property_type": property_type,
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "sqft": sqft,
            "lot_size": lot_size,
            "year_built": year_built,
            "age": age,
            "garage_spaces": garage_spaces,
            "has_pool": has_pool,
            "has_fireplace": has_fireplace,
            "renovated": renovated,
            "distance_to_school_miles": distance_to_school,
            "distance_to_transit_miles": distance_to_transit,
            "distance_to_shopping_miles": distance_to_shopping,
            "listing_date": listing_date.strftime("%Y-%m-%d"),
            "sold_date": sold_date.strftime("%Y-%m-%d"),
            "days_on_market": days_on_market,
            "price": int(price),
        }
    )

# Create DataFrame
df = pd.DataFrame(data)

# Save to Excel
output_file = "data/houses.xlsx"
df.to_excel(output_file, index=False, sheet_name="Properties")

print(f"✅ Sample data created: {output_file}")
print("\nDataset Overview:")
print(f"- Number of properties: {len(df)}")
print(f"- Price range: ${df['price'].min():,} - ${df['price'].max():,}")
print(f"- Average price: ${df['price'].mean():,.0f}")
print(f"- Median price: ${df['price'].median():,.0f}")
print(f"\nLocations: {df['location'].unique().tolist()}")
print(f"Property types: {df['property_type'].unique().tolist()}")
print("\nFirst 3 properties:")
print(df[["property_id", "location", "bedrooms", "bathrooms", "sqft", "price"]].head(3))
