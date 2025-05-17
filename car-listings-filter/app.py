from flask import Flask, render_template, request
import pandas as pd
import os

app = Flask(__name__)

# Mapping dataset names to their paths
DATASETS = {
    'cars.com': 'datasets/classified_cars.csv',
    'EV.com': 'datasets/cleaned_data (5).csv',
    'wattnow': 'datasets/another_site.csv',
    'electrifying': 'datasets/last.csv'
}

@app.route('/', methods=['GET', 'POST'])
def index():
    selected = request.form.get('dataset') or 'cars.com'
    df = pd.read_csv(DATASETS[selected])

    # Normalize any column containing 'price' to a consistent 'price'
    price_columns = [col for col in df.columns if 'price' in col.lower()]
    if price_columns:
        df.rename(columns={price_columns[0]: 'price'}, inplace=True)

    filters = {}
    filtered_df = df.copy()  # Create a copy to apply filters to
    
    if request.method == 'POST':
        # Handle price filters separately
        min_price = request.form.get('min_price')
        max_price = request.form.get('max_price')
        
        if min_price and min_price.strip():
            try:
                min_price_float = float(min_price)
                filtered_df = filtered_df[filtered_df['price'].astype(float) >= min_price_float]
                filters['min_price'] = min_price
            except (ValueError, TypeError):
                pass  # Invalid input, ignore filter
                
        if max_price and max_price.strip():
            try:
                max_price_float = float(max_price)
                filtered_df = filtered_df[filtered_df['price'].astype(float) <= max_price_float]
                filters['max_price'] = max_price
            except (ValueError, TypeError):
                pass  # Invalid input, ignore filter
        
        # Handle Make filter separately with exact matching
        make_filter = request.form.get('Make')
        if make_filter and make_filter.strip():
            filtered_df = filtered_df[filtered_df['Make'].astype(str).str.contains(make_filter, case=False, na=False)]
            filters['Make'] = make_filter
        
        # Apply other filters from the form
        for col in filtered_df.columns:
            if col not in ['price', 'Make']:  # Skip price and Make as we handle them separately
                val = request.form.get(col)
                if val and val.strip():
                    filters[col] = val
                    filtered_df = filtered_df[filtered_df[col].astype(str).str.contains(val, case=False, na=False)]

    # Debug information
    print(f"Total records before filtering: {len(df)}")
    print(f"Total records after filtering: {len(filtered_df)}")
    print(f"Applied filters: {filters}")
    
    return render_template('index.html', 
                          datasets=DATASETS.keys(), 
                          selected=selected, 
                          columns=filtered_df.columns, 
                          filters=filters, 
                          data=filtered_df.head(50).to_dict(orient='records'))

if __name__ == '__main__':
    app.run(debug=True)