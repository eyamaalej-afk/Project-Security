import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
df = pd.read_csv('C:/Users/salma/Downloads/poof/cars_data/cars_listings_20250509_181845.csv')
print(df.head())
df['Price'] = df['Price'].replace('[\$,]', '', regex=True).astype(float)
df['Mileage']=df['Mileage'].replace('[\,mi.]', '',regex=True).astype(float)
print(df.head())
df_clean=df.dropna(subset=['Dealer','Location','Rating'])
# Optionally reset index
df_clean.reset_index(drop=True, inplace=True)
# List of known compound car brands
compound_makes = [
    "Alfa Romeo", "Land Rover", "Aston Martin", "Rolls Royce",
    "Chevrolet Corvette", "Mercedes Benz", "Mini Cooper", "Genesis GV",
    "Ford Mustang", "BMW Alpina", "Jeep Grand", "RAM 1500",
    "GMC Sierra", "Dodge Charger", "Tesla Model"
]
def extract_make_model(title):
    parts = title.split()
    year = parts[0]
    rest = " ".join(parts[1:])
    
    for make in compound_makes:
        if rest.startswith(make):
            return pd.Series([year, make, rest[len(make):].strip()])
    
    # Fallback to one-word make
    return pd.Series([year, parts[1], " ".join(parts[2:])])

# Apply extraction
df_clean[['Year', 'Make', 'Model']] = df_clean['Title'].apply(extract_make_model)

# Show result
print(df_clean[['Title', 'Year', 'Make', 'Model']].head(10))
model_year_prices = (
    df_clean.groupby(['Make','Model', 'Year',])['Price']
    .mean()
    .reset_index()
    .sort_values(['Model', 'Year'])
)
print(model_year_prices.head())

import seaborn as sns
top_models = df_clean['Model'].value_counts().head(5).index.tolist()
filtered = model_year_prices[model_year_prices['Model'].isin(top_models)]
# Plot
plt.figure(figsize=(12, 6))
sns.lineplot(data=filtered, x='Year', y='Price', hue='Model', marker='o')
plt.title('Average Price of Car Models Over Different Release Years')
plt.ylabel('Average Price ($)')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()
avg_prices=df_clean.groupby("Make")["Price"].mean().sort_values(ascending=False)
print(avg_prices.head())

from sklearn.cluster import KMeans
import numpy as np

# Reshape price into a 2D array for sklearn
prices = df_clean['Price'].values.reshape(-1, 1)

# KMeans clustering into 3 clusters: Cheap, Midrange, Luxury
# Log-transform the prices to reduce skewness
prices_log = np.log(df_clean['Price'].values.reshape(-1, 1))
kmeans = KMeans(n_clusters=3, random_state=0)
df_clean['PriceCluster'] = kmeans.fit_predict(prices_log)

# Reorder cluster labels based on actual price
cluster_centers = kmeans.cluster_centers_.flatten()
sorted_indices = np.argsort(cluster_centers)
cluster_to_label = {sorted_indices[0]: 'Cheap',
                    sorted_indices[1]: 'Midrange',
                    sorted_indices[2]: 'Luxury'}

df_clean['PriceClass'] = df_clean['PriceCluster'].map(cluster_to_label)

print(df_clean['PriceClass'].value_counts())
import seaborn as sns
sns.boxplot(data=df_clean, x='PriceClass', y='Price')
plt.show()

import numpy as np

# Compute log-transformed values on the fly
log_mileage = np.log1p(df_clean['Mileage'])  # log1p avoids log(0)
log_price = np.log1p(df_clean['Price'])

# Correlation
log_corr = log_mileage.corr(log_price)
print(f'Log-transformed correlation: {log_corr}')
import seaborn as sns
import matplotlib.pyplot as plt

# Create a temporary DataFrame for plotting
temp_df = pd.DataFrame({
    'LogMileage': log_mileage,
    'LogPrice': log_price
})

# Scatterplot with regression line
sns.lmplot(data=temp_df, x='LogMileage', y='LogPrice', aspect=2, height=6)
plt.title("Log-Transformed Mileage vs Price (Regression)")
plt.show()



