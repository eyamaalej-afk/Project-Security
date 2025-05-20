from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import pandas as pd
import time

# Chrome options
options = Options()
options.add_argument("--headless")
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")

# Start driver
driver = webdriver.Chrome(options=options)

# Base URL for ALL EV listings (adjust this if you're targeting used only or specific makes)
base_url = "https://www.ev.com/vehicles?stock_type=used&page={}"

data = []

# Scrape first 20 pages
for page in range(1, 21):
    print(f"Scraping page {page}...")
    driver.get(base_url.format(page))
    time.sleep(2)  # Allow time for page to load

    soup = BeautifulSoup(driver.page_source, "html.parser")
    
    cars = soup.find_all("a", class_="vehicle-card-link")
    
    for car in cars:
        title = car.find("h2", class_="title")
        title_text = title.get_text(strip=True) if title else "N/A"

        mileage_tag = car.find_next("div", {"data-qa": "mileage"})
        mileage = mileage_tag.get_text(strip=True) if mileage_tag else "N/A"

        price_tag = car.find_next("span", {"data-qa": "primary-price"})
        price = price_tag.get_text(strip=True) if price_tag else "N/A"

        dealer_tag = car.find_next("strong")
        dealer = dealer_tag.get_text(strip=True) if dealer_tag else "N/A"

        location_tag = car.find_next("div", {"data-qa": "miles-from-user"})
        location = location_tag.get_text(strip=True) if location_tag else "N/A"

        data.append({
            "Title": title_text,
            "Mileage": mileage,
            "Price": price,
            "Dealer": dealer,
            "Location": location
        })

driver.quit()

# Create DataFrame
df = pd.DataFrame(data)

# Save to CSV
df.to_csv("ev_listings.csv", index=False)
print("Scraping complete. Data saved to ev_listings.csv")
