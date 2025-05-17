#Data_Cleaning

# STEP 1:Import libraries
import pandas as pd
import numpy as np
import re
from google.colab import files

# STEP 2: Upload the CSV file
uploaded = files.upload()
# Load the data
df = pd.read_csv('electric_cars_1.csv')

# Helper function to safely convert to float
def safe_float(x):
    try:
        return float(x)
    except:
        return np.nan

# 1. Clean price column
# Remove £ and commas, convert to numeric
df['price'] = df['price'].str.replace('£', '').str.replace(',', '')
df['price'] = df['price'].apply(safe_float)

# Calculate average price and fill missing values
avg_price = round(df['price'].mean())
df['price'] = df['price'].fillna(avg_price).astype(int)

# 2. Clean range column
# Remove 'mi' and convert to numeric
df['range'] = df['range '].str.replace('mi', '', regex=False).apply(safe_float)

# Calculate average range and fill missing values
avg_range = round(df['range'].mean())
df['range'] = df['range'].fillna(avg_range).astype(int)
df = df.drop(columns=['range '])  # Remove the original column with space

# 3. Clean battery_size column
# Remove 'kWh' variations and 'Battery sizes:' prefix
df['battery_size'] = df['battery_size'].str.replace('kWh', '', regex=False)\
                       .str.replace('kwh', '', regex=False)\
                       .str.replace('Battery sizes: ', '', regex=False)\
                       .str.replace('Wh', '', regex=False)\
                       .str.replace(' ', '', regex=False)

# Function to handle ranges and calculate average
def process_battery_size(value):
    if pd.isna(value) or value == 'N/A':
        return np.nan
    if '-' in value:
        parts = value.split('-')
        try:
            return (float(parts[0]) + float(parts[1])) / 2
        except:
            return np.nan
    try:
        return float(value)
    except:
        return np.nan

df['battery_size'] = df['battery_size'].apply(process_battery_size)

# Calculate average battery size and fill missing values
avg_battery = round(df['battery_size'].mean())
df['battery_size'] = df['battery_size'].fillna(avg_battery).astype(int)

# 4. Clean miles_per_kwh column
# Convert to numeric, handling N/A and estimates
df['miles_per_kwh'] = df['miles_per_kwh'].str.replace('(est)', '', regex=False)\
                        .str.replace('(tested)', '', regex=False)\
                        .str.replace('(claimed)', '', regex=False)\
                        .str.replace('N/A', '', regex=False)\
                        .str.replace(' ', '', regex=False)

# Extract first numeric value when there's a range
def process_miles_per_kwh(value):
    if pd.isna(value) or value == '':
        return np.nan
    # Handle cases like "3.8. - 4.1" by taking first value
    if '.' in value and '-' in value:
        parts = value.split('-')
        try:
            return float(parts[0].strip())
        except:
            return np.nan
    try:
        return float(value)
    except:
        return np.nan

df['miles_per_kwh'] = df['miles_per_kwh'].apply(process_miles_per_kwh)

# Calculate average and fill missing values
avg_miles = round(df['miles_per_kwh'].mean(), 1)
df['miles_per_kwh'] = df['miles_per_kwh'].fillna(avg_miles).round(1)
# 5. Clean max_dc_charge column
def clean_max_dc_charge(value):
    if pd.isna(value) or value == 'N/A':
        return np.nan

    # Remove 'kW' and any parentheses content like (est)
    cleaned = str(value).replace('kW', '').replace(' ', '')\
                        .split('(')[0].strip()

    # Handle ranges (value-value)
    if '-' in cleaned:
        parts = cleaned.split('-')
        try:
            return (float(parts[0]) + float(parts[1])) / 2
        except:
            return np.nan

    try:
        return float(cleaned)
    except:
        return np.nan

df['max_dc_charge'] = df['max_dc_charge'].apply(clean_max_dc_charge)

# Calculate average and fill missing values
avg_max_charge = round(df['max_dc_charge'].mean())
df['max_dc_charge'] = df['max_dc_charge'].fillna(avg_max_charge).astype(int)

# Save the cleaned data again
df.to_csv('electric_cars_cleaned.csv', index=False)

print(f"Used average for max_dc_charge: {avg_max_charge}")

# Save cleaned data
df.to_csv('electric_cars_cleaned.csv', index=False)

print("Data cleaning complete. Saved to electric_cars_cleaned.csv")
print(f"Used averages: Price={avg_price}, Range={avg_range}, Battery={avg_battery}, Miles/kWh={avg_miles}")

files.download('electric_cars_cleaned.csv')
