
#Data_Cleaning

from google.colab import files
uploaded = files.upload()

import pandas as pd
import re
from google.colab import files

# STEP 2: Upload the CSV file
uploaded = files.upload()

# STEP 3: Load the data
df = pd.read_csv('electric_cars_all_pages.csv')

# STEP 4: Remove duplicates
df = df.drop_duplicates()

# STEP 5: Rename 'Detailed Car Title' to 'Car name'
df = df.rename(columns={'Detailed Car Title': 'Car name'})

# STEP 6: Remove 'Specification' from the end of each name
df['Car name'] = df['Car name'].str.replace(' Specification', '', regex=False).str.strip()

# STEP 7: Handle missing values in 'Seller Location'
df['Seller Location'] = df['Seller Location'].fillna('Unknown')

# STEP 8: Clean 'Phone Number' column
def clean_phone(phone):
    if pd.isna(phone):
        return 'NONE'
    phone = str(phone)
    phone = phone.replace('Phone:', '').replace('‪', '').replace('‬', '').strip()
    if not phone.startswith('+') and phone != 'NONE':
        phone = '+' + phone if phone[0].isdigit() else 'NONE'
    return phone

df['Phone Number'] = df['Phone Number'].apply(clean_phone)

# STEP 9: Replace 'other' in 'E-car Producer' with the first word from 'Car name'
df['E-car Producer'] = df.apply(lambda row: row['Car name'].split()[0] if row['E-car Producer'] == 'other' else row['E-car Producer'], axis=1)
# STEP 9: Replace empty or missing values in 'E-car Producer' with the first word from 'Car name'
df['E-car Producer'] = df.apply(lambda row: row['Car name'].split()[0] if pd.isna(row['E-car Producer']) or row['E-car Producer'] == '' else row['E-car Producer'], axis=1)


# STEP 10: Clean 'Charge Rate' column by averaging values in the 'value-value' format and removing 'km'
def clean_charge_rate(charge_rate):
    if pd.isna(charge_rate):
        return None
    charge_rate = str(charge_rate)
    charge_rate = charge_rate.replace('km', '').strip()  # Remove 'km'
    if '-' in charge_rate:
        # If the value is in the form of 'value-value' (e.g., 100-120)
        parts = charge_rate.split('-')
        if len(parts) == 2:
            try:
                # Calculate the average of the two values
                return (float(parts[0]) + float(parts[1])) / 2
            except ValueError:
                return None
    return charge_rate  # If there's no '-' return the original value

df['Charge Rate'] = df['Charge Rate'].apply(clean_charge_rate)
# STEP: Clean 'Charge Rate' by replacing empty values with the rounded average
df['Charge Rate'] = df['Charge Rate'].replace('', pd.NA)  # Replace empty strings with NaN
df['Charge Rate'] = pd.to_numeric(df['Charge Rate'], errors='coerce')  # Convert to numeric (coerce errors)
avg_charge_rate = df['Charge Rate'].mean()  # Calculate average (NaN values will be ignored)
df['Charge Rate'] = df['Charge Rate'].fillna(avg_charge_rate).round(0)  # Fill NaN with average and round

# STEP 11: Clean 'Price ($)' column with rounding
df['Price ($)'] = df['Price ($)'].replace('', pd.NA).str.replace(' ', '')
df['Price ($)'] = pd.to_numeric(df['Price ($)'], errors='coerce')
avg_price = df['Price ($)'].mean()
df['Price ($)'] = df['Price ($)'].fillna(avg_price).round(0)  # Rounded to nearest whole number

# STEP 13: Clean 'Max Speed (km/h)' with rounding
df['Max Speed (km/h)'] = df['Max Speed (km/h)'].replace('--', pd.NA)
df['Max Speed (km/h)'] = pd.to_numeric(df['Max Speed (km/h)'], errors='coerce')
avg_speed = df['Max Speed (km/h)'].mean()
df['Max Speed (km/h)'] = df['Max Speed (km/h)'].fillna(avg_speed).round(0)

# STEP 15: Clean 'Engine Capacity (p.h.)' with rounding
df['Engine Capacity (p.h.)'] = df['Engine Capacity (p.h.)'].replace('--', pd.NA)
df['Engine Capacity (p.h.)'] = df['Engine Capacity (p.h.)'].str.extract('(\d+)')[0]
df['Engine Capacity (p.h.)'] = pd.to_numeric(df['Engine Capacity (p.h.)'], errors='coerce')
avg_engine = df['Engine Capacity (p.h.)'].mean()
df['Engine Capacity (p.h.)'] = df['Engine Capacity (p.h.)'].fillna(avg_engine).round(0)

# STEP 14: Clean 'Battery (kWh)' with rounding
df['Battery (kWh)'] = df['Battery (kWh)'].replace('--', pd.NA)
df['Battery (kWh)'] = df['Battery (kWh)'].str.replace(',', '.').str.replace('"', '')
df['Battery (kWh)'] = pd.to_numeric(df['Battery (kWh)'], errors='coerce')
avg_battery = df['Battery (kWh)'].mean()
df['Battery (kWh)'] = df['Battery (kWh)'].fillna(avg_battery).round(1)  # Keep one decimal place

# STEP 12: Clean 'Current Mileage'
df['Current Mileage'] = df['Current Mileage'].replace('--', pd.NA)
df['Current Mileage'] = df['Current Mileage'].str.replace('km', '', regex=False).str.replace(' ', '')
df['Current Mileage'] = pd.to_numeric(df['Current Mileage'], errors='coerce')
avg_mileage = df['Current Mileage'].mean()
df['Current Mileage'] = df['Current Mileage'].fillna(avg_mileage).round(0)

# Convert numeric columns to integers (remove decimal points)
df['Price ($)'] = df['Price ($)'].astype(int)
df['Current Mileage'] = df['Current Mileage'].astype(int)
df['Max Speed (km/h)'] = df['Max Speed (km/h)'].astype(int)
df['Engine Capacity (p.h.)'] = df['Engine Capacity (p.h.)'].astype(int)

# STEP 16: Convert 'First Registration' to integer year
df['First Registration'] = pd.to_numeric(df['First Registration'], errors='coerce')  # Coerce invalid values to NaN
# Replace NULL values in 'First Registration' with 2024
df['First Registration'] = df['First Registration'].fillna(2024).astype(int)
df['First Registration'] = df['First Registration'].fillna(0).astype(int)

# STEP 17: Save cleaned data
df.to_csv('cleaned_electric_cars_final.csv', index=False)

# STEP 18: Download cleaned CSV
files.download('cleaned_electric_cars_final.csv')

"""Transformation of data type"""

from google.colab import files
uploaded = files.upload()
import pandas as pd
# Load your dataset (replace with your actual file path)
df = pd.read_csv('electric_cars_dashboard_ready.csv')

# Convert columns to decimal (float) format
df['MilesPerDollar'] = df['MilesPerDollar'].astype(float)
df['RangePerKWh'] = df['RangePerKWh'].astype(float)

# Display the transformed data
print(df[['name', 'MilesPerDollar', 'RangePerKWh']].head())

# Optional: Save the cleaned data
df.to_csv('electric_cars_decimal_transformed.csv', index=False)
files.download('electric_cars_decimal_transformed.csv')

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.scatter(df['MilesPerDollar'], df['RangePerKWh'], alpha=0.5)
plt.title('Efficiency: Miles per Dollar vs. Range per kWh')
plt.xlabel('Miles per Dollar (decimal)')
plt.ylabel('Range per kWh (decimal)')
plt.grid(True)
plt.show()

"""# **Tranformation** ;  by adding 5 new columns"""

import pandas as pd

# Load your CSV file (you can use Google Drive or upload manually)
df = pd.read_csv("cleaned_electric_cars_final.csv")  # Update with the correct path

# Create the Battery Efficiency column
df["Battery Efficiency"] = df["Battery (kWh)"] / df["Engine Capacity (p.h.)"]
df["Price per kWh"] = df["Price ($)"] / df["Battery (kWh)"]
df["Performance Index"] = (df["Battery (kWh)"] * df["Engine Capacity (p.h.)"]) / df["Price ($)"]
df["Car Age"] = 2025 - df["First Registration"]
df["Engine Category"] = pd.cut(df["Engine Capacity (p.h.)"],
                               bins=[0, 150, 300, 600, 1200],
                               labels=["Small", "Medium", "High", "Extreme"])


# Display the updated DataFrame
df[["Car name", "Battery (kWh)", "Engine Capacity (p.h.)", "Battery Efficiency" , "Price per kWh" , "Performance Index" ,"Car Age" , "Engine Category"]].head()

df.to_csv("cleaned_electric_cars_transform.csv", index=False)
files.download('cleaned_electric_cars_transform.csv')

from google.colab import files
uploaded = files.upload()  # this will open a file chooser to upload your file
import pandas as pd

df = pd.read_csv('cleaned_electric_cars_transform.csv')
columns_to_transform = ['Battery Efficiency', 'Price per kWh', 'Performance Index']

for col in columns_to_transform:
    # Make sure column exists
    if col in df.columns:
        # Convert float to string and replace '.' with ','
        df[col] = df[col].astype(str).str.replace('.', ',', regex=False)

df.head()
df.to_csv('transformed_data.csv', index=False)
files.download('transformed_data.csv')
