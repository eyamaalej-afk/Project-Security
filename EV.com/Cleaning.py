import pandas as pd
import numpy as np
from google.colab import files
import io
import re

# Prompt user to upload the CSV file
print("Please upload 'data.csv' from your PC")
uploaded = files.upload()

# Check if the file was uploaded and get its name
if not uploaded:
    raise FileNotFoundError("No file was uploaded. Please try again.")

# Assuming the uploaded file is 'data.csv'
# The uploaded file is stored as a dictionary with the file name as the key
csv_file_name = list(uploaded.keys())[0]  # Get the first uploaded file name
csv_path = io.BytesIO(uploaded[csv_file_name])  # Read the file into memory

# Load the CSV
df = pd.read_csv(csv_path)

print(df.head())  # View first 5 rows
print(df.info())  # Check data types and missing values
print(df.describe())  # Summary statistics
print(df.isnull().sum())  # Count missing values per column

# Count unique brands
unique_brands = df['Brand'].str.lower().str.strip().unique()  # Normalize case and whitespace
brand_count = len(unique_brands)
print("Number of unique brands:", brand_count)
print("Unique brands:", unique_brands)

# Step 1: Rename columns to include units
df = df.rename(columns={
    'Range': 'Range (mi)',
    'Fast Charging L3': 'Fast Charging L3 (minutes)',
    'Mileage': 'Mileage (mi)',
    'Performance': 'Performance (HP)',
    'Seats': 'Seats',
    'Battery size': 'Battery Size (kWh)',
    'Price': 'Price ($)',
    'Year': 'Year'
})

# Step 2: Split Battery warranty into two columns
df[['Battery Warranty (months)', 'Battery Warranty (miles)']] = df['Battery warranty'].str.split('/', expand=True)
df['Battery Warranty (months)'] = df['Battery Warranty (months)'].str.replace(' month', '').astype(float)

# Handle 'unlimited' in 'Battery Warranty (miles)'
df['Battery Warranty (miles)'] = df['Battery Warranty (miles)'].str.replace(' miles', '')  # Remove ' miles' suffix
df['Battery Warranty (miles)'] = pd.to_numeric(df['Battery Warranty (miles)'], errors='coerce')  # Convert to float, 'unlimited' becomes NaN

df = df.drop(columns=['Battery warranty'])

# Create a copy to preserve original data
df_original = df.copy()

# Define columns with units to clean
columns_with_units = {
    'Range (mi)': (' mi', lambda x: x),  # No preprocessing beyond unit removal
    'Fast Charging L3 (minutes)': (' minutes', lambda x: x),
    'Mileage (mi)': (' mi', lambda x: x),
    'Performance (HP)': (' HP', lambda x: x),
    'Battery Size (kWh)': (' kWh', lambda x: x),
    'Price ($)': ('$', lambda x: x.replace(',', '') if isinstance(x, str) else str(x)),  # Remove commas for Price
}

# Clean columns with units
for col, (unit, preprocess) in columns_with_units.items():
    if col in df.columns:
        # Store original values
        original_values = df[col].copy()
        # Apply preprocessing (e.g., remove commas for Price)
        df[col] = df[col].apply(lambda x: preprocess(x) if pd.notna(x) else x)
        # Remove unit and keep as string
        df[col] = df[col].astype(str).str.replace(unit, '', regex=False).str.strip()
        # Preserve "N\A" and handle failed unit removal
        mask = df[col].str.contains(unit, regex=False, na=False)
        if mask.any():
            print(f"Warning: Some values in '{col}' still contain '{unit}' after cleaning:")
            print(df.loc[mask, col].value_counts().head())
            # Revert to original values where unit removal failed
            df.loc[mask, col] = original_values[mask]
        # Ensure "N\A" is preserved
        df[col] = df[col].where(~df[col].str.lower().eq('nan'), original_values)

# Clean 'Seats'
if 'Seats' in df.columns:
    original_values = df['Seats'].copy()
    df['Seats'] = df['Seats'].astype(str).str.replace('Up to ', '', regex=False).str.replace(' seats', '', regex=False).str.strip()
    # Preserve "N\A" and revert failed cleanups
    mask = df['Seats'].str.contains('Up to | seats', regex=True, na=False)
    if mask.any():
        print(f"Warning: Some values in 'Seats' still contain unwanted text:")
        print(df.loc[mask, 'Seats'].value_counts().head())
        df.loc[mask, 'Seats'] = original_values[mask]
    df['Seats'] = df['Seats'].where(~df['Seats'].str.lower().eq('nan'), original_values)

# Clean 'Year'
if 'Year' in df.columns:
    original_values = df['Year'].copy()
    df['Year'] = df['Year'].astype(str).str.replace('.0', '', regex=False).str.strip()
    # Preserve "N\A" and revert failed cleanups
    mask = df['Year'].str.contains('.0', regex=False, na=False)
    if mask.any():
        print(f"Warning: Some values in 'Year' still contain '.0':")
        print(df.loc[mask, 'Year'].value_counts().head())
        df.loc[mask, 'Year'] = original_values[mask]
    df['Year'] = df['Year'].where(~df['Year'].str.lower().eq('nan'), original_values)

# Clean warranty columns
for col in ['Battery Warranty (months)', 'Battery Warranty (miles)']:
    if col in df.columns:
        original_values = df[col].copy()
        df[col] = df[col].astype(str).str.strip()
        # Preserve "N\A"
        df[col] = df[col].where(~df[col].str.lower().eq('nan'), original_values)

# Ensure non-numeric columns are strings
string_columns = ['Brand', 'Exterior color', 'Interior color', 'Charging type', 'VIN', 'URL', 'Title']
for col in string_columns:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip()
        # Preserve "N\A"
        df[col] = df[col].where(~df[col].str.lower().eq('nan'), df_original[col])

# Check data types and nulls after cleaning
print("\nData types after cleaning:")
print(df.dtypes)
print("\nNull values after cleaning:")
print(df.isna().sum())

# List of columns to convert
numeric_cols = [
    'Range (mi)',
    'Fast Charging L3 (minutes)',
    'Mileage (mi)',
    'Performance (HP)',
    'Seats',
    'Year',
    'Battery Size (kWh)',
    'Price ($)',
    'Battery Warranty (months)',
    'Battery Warranty (miles)'
]

# Remove any non-numeric characters (like $, commas, etc.) and convert
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col].replace('[^\d.]', '', regex=True), errors='coerce')
    
    
# Clean 'Title' column to keep only model and configuration (e.g., "3 Rear-Wheel Drive")
def clean_title(title):
    if pd.isna(title):
        return title
    title = str(title).strip()
    # Extract model and configuration (everything after "Model")
    match = re.search(r'Model\s*(\S+.*)', title, re.IGNORECASE)
    if match:
        model_config = match.group(1)
        # Add spaces before uppercase letters (e.g., "3Rear-Wheel" -> "3 Rear-Wheel")
        model_config = re.sub(r'(\d)([A-Z])', r'\1 \2', model_config)
        # Add spaces before certain keywords (e.g., "StandardRange" -> "Standard Range")
        model_config = re.sub(r'([a-z])([A-Z])', r'\1 \2', model_config)
        # Standardize capitalization (title case)
        model_config = model_config.title()
        # Remove extra spaces
        model_config = ' '.join(model_config.split())
        return model_config
    return title  # Return original if pattern not matched

if 'Title' in df.columns:
    df['Title'] = df['Title'].apply(clean_title)
    
# Rename 'Title' column to 'Model'
df = df.rename(columns={'Title': 'Model'})

# Define the new column order
new_order = [
    'Model',
    'Brand',
    'Price ($)',
    'Year',
    'Exterior color',
    'Interior color',
    'Seats',
    'Range (mi)',
    'Fast Charging L3 (minutes)',
    'Mileage (mi)',
    'Performance (HP)',
    'Charging type',
    'Battery Size (kWh)',
    'Battery Warranty (months)',
    'Battery Warranty (miles)',
    'VIN',
    'URL'
]

# Reorder the DataFrame
df1 = df[new_order]

# Check for duplicate rows (all columns must be identical)
duplicates = df1[df1.duplicated(keep=False)]

# Print the number of duplicate rows
print(f"Number of duplicate rows: {len(duplicates)}")

# If duplicates exist, display them
if len(duplicates) > 0:
    print("\nDuplicate rows:")
    print(duplicates)
else:
    print("No duplicate rows found.")

# Check for null values in each column
null_counts = df1.isna().sum()

# Print the number of null values per column
print("Null values per column:")
print(null_counts)

# Print total number of rows with any null values
print(f"\nTotal rows with at least one null value: {df1.isna().any(axis=1).sum()}")

# Filter rows where Mileage (mi) is 0 and count them
zero_mileage_count = len(df1[df1['Mileage (mi)'] == 0])

# Print the result
print(f"Number of cars with zero mileage: {zero_mileage_count}")

# 1. Price ($): Impute with median by Model, Brand, Year
df1['Price ($)'] = df1.groupby(['Model', 'Brand', 'Year'])['Price ($)'].transform(lambda x: x.fillna(x.median()))
df1['Price ($)'] = df1['Price ($)'].fillna(df1['Price ($)'].median())

# 2. Year: Impute with mode, with fallback to a default value (e.g., 2025)
def safe_mode(series):
    mode = series.mode()
    return mode[0] if not mode.empty else 2025  # Fallback to 2025 if mode is empty
df1['Year'] = df1['Year'].fillna(safe_mode(df1['Year']))

# 3. Seats: Impute with mode by Model, Brand, with fallback to overall mode or default (e.g., 5)
def safe_mode_by_group(group):
    mode = group.mode()
    return mode[0] if not mode.empty else df1['Seats'].mode()[0] if not df1['Seats'].mode().empty else 5
df1['Seats'] = df1.groupby(['Model', 'Brand'])['Seats'].transform(lambda x: x.fillna(safe_mode_by_group(x)))
df1['Seats'] = df1['Seats'].fillna(safe_mode(df1['Seats']))

# 4. Range (mi): Impute with median by Model, Brand, Year
df1['Range (mi)'] = df1.groupby(['Model', 'Brand', 'Year'])['Range (mi)'].transform(lambda x: x.fillna(x.median()))
df1['Range (mi)'] = df1['Range (mi)'].fillna(df1['Range (mi)'].median())

# 5. Fast Charging L3 (minutes): Impute with median by Model, Brand, Battery Size, or flag as Unknown
df1['Fast Charging L3 (minutes)'] = df1.groupby(['Model', 'Brand', 'Battery Size (kWh)'])['Fast Charging L3 (minutes)'].transform(lambda x: x.fillna(x.median()))
df1['Fast Charging L3 (minutes)'] = df1['Fast Charging L3 (minutes)'].fillna('Unknown')

# 6. Mileage (mi): Impute based on Year (12,000 miles/year, 0 for 2025 models)
current_year = 2025
avg_mileage_per_year = 12000
df1['Mileage (mi)'] = df1.apply(
    lambda row: 0 if pd.isna(row['Mileage (mi)']) and row['Year'] == current_year
    else (current_year - row['Year']) * avg_mileage_per_year if pd.isna(row['Mileage (mi)'])
    else row['Mileage (mi)'],
    axis=1
)

# 7. Performance (HP): Impute with median by Model, Brand
df1['Performance (HP)'] = df1.groupby(['Model', 'Brand'])['Performance (HP)'].transform(lambda x: x.fillna(x.median()))
df1['Performance (HP)'] = df1['Performance (HP)'].fillna(df1['Performance (HP)'].median())

# 8. Battery Size (kWh): Impute with median by Model, Brand, Year
df1['Battery Size (kWh)'] = df1.groupby(['Model', 'Brand', 'Year'])['Battery Size (kWh)'].transform(lambda x: x.fillna(x.median()))
df1['Battery Size (kWh)'] = df1['Battery Size (kWh)'].fillna(df1['Battery Size (kWh)'].median())

# 9. Battery Warranty (months) and (miles): Impute with mode by Model, Brand, with fallback
def safe_warranty_mode(group):
    mode = group.mode()
    return mode[0] if not mode.empty else df1['Battery Warranty (months)'].mode()[0] if not df1['Battery Warranty (months)'].mode().empty else 96
df1['Battery Warranty (months)'] = df1.groupby(['Model', 'Brand'])['Battery Warranty (months)'].transform(lambda x: x.fillna(safe_warranty_mode(x)))
df1['Battery Warranty (months)'] = df1['Battery Warranty (months)'].fillna(safe_mode(df1['Battery Warranty (months)']))

def safe_warranty_miles_mode(group):
    mode = group.mode()
    return mode[0] if not mode.empty else df1['Battery Warranty (miles)'].mode()[0] if not df1['Battery Warranty (miles)'].mode().empty else 100000
df1['Battery Warranty (miles)'] = df1.groupby(['Model', 'Brand'])['Battery Warranty (miles)'].transform(lambda x: x.fillna(safe_warranty_miles_mode(x)))
df1['Battery Warranty (miles)'] = df1['Battery Warranty (miles)'].fillna(safe_mode(df1['Battery Warranty (miles)']))

# Verify no nulls remain
print("Null values after imputation:")
print(df1.isna().sum())

# 10. Charging type: Impute with mode by Model and Brand, with fallback to global mode or 'Unknown'
def safe_charging_mode(group):
    mode = group.mode()
    return mode[0] if not mode.empty else df1['Charging type'].mode()[0] if not df1['Charging type'].mode().empty else 'Unknown'

df1['Charging type'] = df1.groupby(['Model', 'Brand'])['Charging type'].transform(lambda x: x.fillna(safe_charging_mode(x)))
df1['Charging type'] = df1['Charging type'].fillna('Unknown')

# Verify no nulls remain
print("Null values after imputation:")
print(df1.isna().sum())

# Count vehicles per brand
brand_counts = df1['Brand'].value_counts()

# Print the results
print("Number of vehicles per brand:")
print(brand_counts)

import re

def clean_model(model):
    if pd.isna(model):
        return model
    model = str(model).strip().lower()

    # Remove the year and brand (e.g., "2023volvo")
    brands = ['chevrolet', 'ford', 'hyundai', 'rivian', 'acura', 'alfa romeo',
              'audi', 'bmw', 'cadillac', 'chrysler', 'dodge', 'fiat', 'gmc',
              'genesis', 'honda', 'jeep', 'land rover', 'lexus', 'lucid', 'mini',
              'mazda', 'nissan', 'porsche', 'subaru', 'toyota', 'vinfast',
              'volkswagen', 'volvo']

    # Escape multi-word brands (like 'alfa romeo', 'land rover')
    brands_pattern = '|'.join(sorted(map(re.escape, brands), key=lambda b: -len(b)))

    # Full pattern to remove the year and brand (e.g., "2023volvo")
    model = re.sub(rf'^\d{{4}}({brands_pattern})', '', model)

    # Clean extra spaces and apply title case
    model = ' '.join(model.split()).title()

    return model

df1['Model'] = df1['Model'].apply(clean_model)

# Fill missing values in 'Exterior color' and 'Interior color' with 'Unknown'
df1['Exterior color'] = df1['Exterior color'].fillna('Unknown')
df1['Interior color'] = df1['Interior color'].fillna('Unknown')

# Drop rows where 'VIN' is missing
df1 = df1.dropna(subset=['VIN'])

# Check the null values after the operation
print(df1.isnull().sum())

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Assuming your dataframe is 'df1' and it's already cleaned

# Step 1: Select the price column for clustering
price_data = df1[['Price ($)']]

# Step 2: Standardize the price data for K-means clustering
scaler = StandardScaler()
price_scaled = scaler.fit_transform(price_data)

# Step 3: Apply K-means clustering (let's use 3 clusters for price ranges)
kmeans = KMeans(n_clusters=3, random_state=42)
df1['Cluster'] = kmeans.fit_predict(price_scaled)

# Step 4: Get the cluster centers (after inverse transformation to the original price scale)
cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)

# Sort the cluster centers to define price ranges
sorted_centers = sorted(cluster_centers[:, 0])  # Sorting the cluster centers to identify price ranges

# Step 5: Define price range labels based on sorted cluster centers
price_range_labels = {
    sorted_centers[0]: "Low",    # Lowest cluster center -> Low price range
    sorted_centers[1]: "Medium", # Middle cluster center -> Medium price range
    sorted_centers[2]: "High",   # Highest cluster center -> High price range
}

# Step 6: Create a function to classify cars into price ranges
def classify_price_range(price):
    if price <= sorted_centers[0]:
        return "Low"
    elif sorted_centers[0] < price <= sorted_centers[1]:
        return "Medium"
    else:
        return "High"

# Step 7: Apply the classification to the dataframe
df1['Price Range'] = df1['Price ($)'].apply(classify_price_range)

# Step 8: Check the results
print(df1[['Price ($)', 'Price Range']].head())

import matplotlib.pyplot as plt
import seaborn as sns

# Set the style for the plot
sns.set(style="whitegrid")

# Plot the clustering results
plt.figure(figsize=(10, 6))
plt.scatter(df1['Price ($)'], [0] * len(df1), c=df1['Cluster'], cmap='viridis', s=100, alpha=0.7, edgecolors='k')

# Add labels and title
plt.title('K-means Clustering of Car Prices', fontsize=16)
plt.xlabel('Price ($)', fontsize=12)
plt.yticks([])  # No y-axis labels, as it's a 1D plot
plt.colorbar(label='Cluster')

# Show the plot
plt.show()

# Remove the 'Cluster' column
df1.drop(columns=['Cluster'], inplace=True)

# Reorder columns by moving 'Price Range' after 'Price ($)'
price_col_index = df1.columns.get_loc('Price ($)')
df1.insert(price_col_index + 1, 'Price Range', df1.pop('Price Range'))

# Display the updated DataFrame
print(df1.head())

# Replace 'Unknown' with a fixed value (e.g., 30 minutes) in df1
df1['Fast Charging L3 (minutes)'] = df1['Fast Charging L3 (minutes)'].replace('Unknown', 30)

# Check the updated DataFrame
print(df1['Fast Charging L3 (minutes)'].head())

from google.colab import files
# Save the cleaned dataframe to a CSV file
df1.to_csv('/content/cleaned_data.csv', index=False)
# Trigger download
files.download('/content/cleaned_data.csv')
