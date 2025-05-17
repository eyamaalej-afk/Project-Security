from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import time
import random
import json
import re
from urllib.parse import urljoin
import pandas as pd
import logging
import os

# Setup logging for debugging and error tracking
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Create output directory if it doesn't exist
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def setup_driver():
    """Configure Chrome WebDriver with optimized settings for headless browsing"""
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--no-sandbox")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)
    
    # Specify the correct path to the ChromeDriver executable
    chromedriver_path = r"C:\Users\EyaMaalej\Desktop\chromedriver-win64\chromedriver.exe"
    service = Service(executable_path=chromedriver_path)
    driver = webdriver.Chrome(service=service, options=options)
    
    driver.set_page_load_timeout(90)
    logging.info(f"ChromeDriver initialized with Chrome version: {driver.capabilities['browserVersion']}")
    return driver

def extract_vehicle_details(detail_url, idx, brand):
    """Extract vehicle details from a detail page with retry logic"""
    driver = setup_driver()
    details = {
        'Brand': brand,
        'Range': 'N/A',
        'Fast Charging L3': 'N/A',
        'Mileage': 'N/A',
        'Performance': 'N/A',
        'Seats': 'N/A',
        'Exterior color': 'N/A',
        'Interior color': 'N/A',
        'Charging type': 'N/A',
        'Battery warranty': 'N/A',
        'VIN': 'N/A',
        'Year': 'N/A',
        'Battery size': 'N/A',
        'URL': detail_url,
        'Title': 'N/A',
        'Price': 'N/A'
    }

    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            print(f"Extracting vehicle {idx} for {brand}: {detail_url} (Attempt {attempt + 1}/{max_attempts})")
            logging.info(f"Starting extraction for vehicle {idx} of {brand}: {detail_url} (Attempt {attempt + 1}/{max_attempts})")
            driver.get(detail_url)

            WebDriverWait(driver, 30).until(
                lambda d: d.execute_script("return document.readyState") == "complete"
            )
            WebDriverWait(driver, 20).until(
                EC.presence_of_element_located(
                    (By.XPATH, "//div[contains(@class, 'grid') or contains(@class, 'spec') or contains(@class, 'text-body1Light')]")
                )
            )
            time.sleep(random.uniform(3, 5))

            soup = BeautifulSoup(driver.page_source, 'html.parser')
            with open(os.path.join(OUTPUT_DIR, f'page_{brand}_{idx}.html'), 'w', encoding='utf-8') as f:
                f.write(driver.page_source)
            logging.debug(f"Saved page source for {brand} vehicle {idx}")

            # Extract grid section details
            grid_sections = soup.find_all("div", class_=lambda x: x and 'grid' in x)
            for grid in grid_sections:
                items = grid.find_all("div", recursive=False)
                for item in items:
                    text_lines = [line.strip() for line in item.get_text("\n").split("\n") if line.strip()]
                    if len(text_lines) >= 2:
                        spec_name = text_lines[0]
                        spec_value = "\n".join(text_lines[1:])
                        if any(x in spec_name.lower() for x in ['range', 'distance']):
                            details['Range'] = spec_value
                        elif 'fast charging' in spec_name.lower():
                            details['Fast Charging L3'] = spec_value
                        elif 'mileage' in spec_name.lower():
                            details['Mileage'] = spec_value
                        elif 'performance' in spec_name.lower():
                            details['Performance'] = spec_value
                        elif 'seats' in spec_name.lower():
                            details['Seats'] = spec_value
                        elif 'battery size' in spec_name.lower():
                            details['Battery size'] = spec_value

            # Extract list section details
            list_items = soup.find_all("div", class_=lambda x: x and 'text-body1Light' in x)
            for item in list_items:
                spec_name = item.get_text(strip=True)
                value_div = item.find_next("div", class_=lambda x: x and 'text-right' in x)
                if value_div:
                    spec_value = value_div.get_text(strip=True)
                    if 'exterior color' in spec_name.lower():
                        details['Exterior color'] = spec_value
                    elif 'interior color' in spec_name.lower():
                        details['Interior color'] = spec_value
                    elif 'charging type' in spec_name.lower():
                        details['Charging type'] = spec_value
                    elif 'battery warranty' in spec_name.lower():
                        details['Battery warranty'] = spec_value
                    elif 'vin' in spec_name.lower():
                        details['VIN'] = spec_value
                    elif 'year' in spec_name.lower():
                        details['Year'] = spec_value

            # Extract title
            details['Title'] = soup.find("h1").get_text(strip=True) if soup.find("h1") else "N/A"

            # Extract price
            price = "N/A"
            price_element = None
            for method in [
                lambda: soup.find("p", class_=lambda x: x and ('text-body1Regular' in x or 'ml:text-h4SemiBold' in x)),
                lambda: soup.find("div", class_=lambda x: x and 'text-h3SemiBold' in x),
                lambda: soup.find(class_=lambda x: x and any(k in x.lower() for k in ['price', 'amount', 'cost', 'msrp', 'total'])),
                lambda: soup.find(string=re.compile(r'^\$?\s*\d{1,3}(?:[,\s]\d{3})*(?:\.\d{2})?$')),
                lambda: driver.find_element(By.XPATH, "//*[contains(text(), '$') and (contains(@class, 'text-body1Regular') or contains(@class, 'ml:text-h4SemiBold') or contains(@class, 'text-h3SemiBold') or contains(@class, 'price') or contains(@class, 'amount') or contains(@class, 'cost'))]")
            ]:
                try:
                    result = method()
                    if result:
                        price_element = result
                        if isinstance(result, str):
                            price = result.strip()
                        elif hasattr(result, 'text'):
                            price = result.text.strip()
                        elif hasattr(result, 'get_text'):
                            price = result.get_text(strip=True)
                        logging.info(f"Price found for {brand} vehicle {idx}: {price} (from element: {str(price_element)})")
                        break
                except Exception as e:
                    logging.debug(f"Price extraction method failed: {str(e)}")
                    continue

            # Clean and validate price
            if price != "N/A":
                cleaned_price = price.replace('\u202f', ' ').replace('\xa0', ' ').replace('*', '').strip()
                cleaned_price = re.sub(r'[^\d,.$\s]', '', cleaned_price).strip()
                cleaned_price = cleaned_price.replace(' ', ',').replace('$', '').strip()
                try:
                    numeric_price = float(cleaned_price.replace(',', '').replace('$', ''))
                    if numeric_price >= 1000:
                        details['Price'] = f"${cleaned_price}"
                    else:
                        logging.warning(f"Price rejected (too low) for {brand} vehicle {idx}: {price}")
                        details['Price'] = "N/A"
                except ValueError:
                    logging.warning(f"Invalid price format for {brand} vehicle {idx}: {price}")
                    details['Price'] = "N/A"
            else:
                logging.warning(f"No price found for {brand} vehicle {idx}")

            price_area = soup.find_all(string=re.compile(r'\$\s*\d'))
            if price_area:
                logging.debug(f"Potential price elements for {brand} vehicle {idx}: {[str(p.parent) for p in price_area[:3]]}")

            print(f"Finished extracting vehicle {idx} for {brand} successfully")
            logging.info(f"Completed extraction for {brand} vehicle {idx} with no errors")
            return details

        except Exception as e:
            logging.error(f"Error processing {brand} vehicle {idx} on attempt {attempt + 1}: {str(e)}")
            if attempt < max_attempts - 1:
                time.sleep(5)
                continue
            driver.save_screenshot(os.path.join(OUTPUT_DIR, f'error_{brand}_{idx}.png'))
            with open(os.path.join(OUTPUT_DIR, f'error_page_{brand}_{idx}.html'), 'w', encoding='utf-8') as f:
                f.write(driver.page_source if driver.page_source else "No page source available")
            print(f"Failed to extract vehicle {idx} for {brand} after {max_attempts} attempts")
            return details

        finally:
            if attempt == max_attempts - 1 or 'details' in locals():
                driver.quit()

def process_listing_page(listings_url, brand, page_num, vehicle_idx, processed_urls):
    """Process a listing page and extract details for each vehicle"""
    driver = setup_driver()
    vehicle_data = []

    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            print(f"Processing {brand} page {page_num}: {listings_url} (Attempt {attempt + 1}/{max_attempts})")
            logging.info(f"Processing {brand} page {page_num}: {listings_url} (Attempt {attempt + 1}/{max_attempts})")
            driver.get(listings_url)
            WebDriverWait(driver, 30).until(
                lambda d: d.execute_script("return document.readyState") == "complete"
            )
            WebDriverWait(driver, 20).until(
                EC.presence_of_element_located(
                    (By.XPATH, "//div[contains(@class, 'border-1') and contains(@class, 'flex')]")
                )
            )
            time.sleep(random.uniform(1, 2))

            soup = BeautifulSoup(driver.page_source, 'html.parser')
            listing_divs = soup.find_all("div", class_=lambda x: x and 'border-1' in x and 'flex' in x)

            if not listing_divs:
                print(f"No listings found on {brand} page {page_num}")
                logging.warning(f"No listings found on {brand} page {page_num}")
                return vehicle_data, vehicle_idx

            for div in listing_divs:
                link = div.find("a", href=re.compile(r'^/vehicle/[\w]+'))
                if link and 'href' in link.attrs:
                    href = link['href']
                    full_url = urljoin("https://ev.com", href)
                    if full_url not in processed_urls:
                        processed_urls.add(full_url)
                        print(f"Found vehicle URL for {brand}: {full_url}")
                        details = extract_vehicle_details(full_url, vehicle_idx, brand)
                        if details:
                            vehicle_data.append(details)
                            vehicle_idx += 1
                            print(f"Added vehicle {vehicle_idx - 1} for {brand}. Page total: {len(vehicle_data)}")

            print(f"Completed {brand} page {page_num} successfully. Processed {len(vehicle_data)} vehicles")
            logging.info(f"Completed {brand} page {page_num} with no errors. Processed {len(vehicle_data)} vehicles")
            return vehicle_data, vehicle_idx

        except Exception as e:
            logging.error(f"Error processing {brand} page {page_num} on attempt {attempt + 1}: {str(e)}")
            if attempt < max_attempts - 1:
                time.sleep(5)
                continue
            driver.save_screenshot(os.path.join(OUTPUT_DIR, f'listings_error_{brand}_page_{page_num}.png'))
            with open(os.path.join(OUTPUT_DIR, f'listings_error_page_{brand}_{page_num}.html'), 'w', encoding='utf-8') as f:
                f.write(driver.page_source if driver.page_source else "No page source available")
            print(f"Failed to process {brand} page {page_num} after {max_attempts} attempts")
            return vehicle_data, vehicle_idx

        finally:
            if attempt == max_attempts - 1 or vehicle_data:
                driver.quit()

def generate_page_url(base_url, brand, page_num):
    """Generate the URL for a specific page of a brand"""
    if page_num == 1:
        return f"{base_url}/make/{brand}"
    else:
        return f"{base_url}/distance/200/make/{brand}/sort/rankedscore_desc/has_ev_incentives/false/incentive_zip/90013/page/{page_num}"

def save_intermediate_results(all_vehicle_data, brand):
    """Save intermediate results to CSV and JSON after each brand"""
    df = pd.DataFrame(all_vehicle_data)
    output_csv = os.path.join(OUTPUT_DIR, f'intermediate_vehicle_data_{brand}.csv')
    output_json = os.path.join(OUTPUT_DIR, f'intermediate_vehicle_data_{brand}.json')
    df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(all_vehicle_data, f, indent=4)
    print(f"Saved intermediate results for {brand} to {output_csv} and {output_json}")
    logging.info(f"Saved intermediate results for {brand} to {output_csv} and {output_json}")

if __name__ == "__main__":
    # Base search URL
    base_search_url = "https://ev.com/search/postal/90013"

    # List of brands to scrape
    brands = ['Tesla', 'Chevrolet', 'Kia', 'Ford', 'Hyundai', 'Rivian', 'Acura', 'Alfa Romeo', 'Audi', 'BMW',
              'Bentley', 'Cadillac', 'Chrysler', 'Dodge', 'FIAT', 'GMC', 'Genesis', 'Honda', 'Jeep',
              'Lamborghini', 'Land Rover', 'Lexus', 'Lincoln', 'Lucid', 'MINI', 'Maserati', 'Mazda',
              'Mercedes-Benz', 'Nissan', 'Polestar', 'Porsche', 'Subaru', 'Toyota', 'VinFast',
              'Volkswagen', 'Volvo']

    # Initialize data storage
    all_vehicle_data = []
    processed_urls = set()
    max_pages = 30

    print(f"Starting scraping for {len(brands)} brands: {', '.join(brands)}")
    logging.info(f"Starting scraping process for {len(brands)} brands")

    # Process each brand
    for brand in brands:
        print(f"\n=== Starting {brand} ===")
        logging.info(f"--- Starting processing for brand: {brand} ---")
        vehicle_idx = 1
        for page_num in range(1, max_pages + 1):
            page_url = generate_page_url(base_search_url, brand, page_num)
            print(f"Fetching {brand} page {page_num}: {page_url}")
            logging.info(f"Generated URL for {brand} page {page_num}")

            page_data, vehicle_idx = process_listing_page(page_url, brand, page_num, vehicle_idx, processed_urls)
            all_vehicle_data.extend(page_data)
            print(f"Finished {brand} page {page_num}. Total vehicles scraped: {len(all_vehicle_data)}")
            logging.info(f"Finished {brand} page {page_num}. Total vehicles: {len(all_vehicle_data)}")

            if not page_data:
                print(f"No more listings for {brand} after page {page_num}. Moving to next brand.")
                logging.info(f"No more listings found for {brand} after page {page_num}")
                break

        save_intermediate_results(all_vehicle_data, brand)
        print(f"=== Completed {brand}. Total vehicles: {len(all_vehicle_data)} ===\n")
        logging.info(f"Completed processing for {brand} with no fatal errors")

    print("\n=== Scraping completed ===")
    logging.info(f"Finished scraping all brands. Total vehicles: {len(all_vehicle_data)}")

    # Create and save final DataFrame
    df = pd.DataFrame(all_vehicle_data)
    print(df.to_string())  # Use print for console output in VSCode

    final_json = os.path.join(OUTPUT_DIR, 'all_vehicle_data.json')
    final_csv = os.path.join(OUTPUT_DIR, 'all_vehicle_data.csv')
    with open(final_json, 'w', encoding='utf-8') as f:
        json.dump(all_vehicle_data, f, indent=4)
    df.to_csv(final_csv, index=False, encoding='utf-8-sig')
    print(f"Saved final results to {final_json} and {final_csv}")
    logging.info("Saved final results")

    print(f"Done! Scraped {len(all_vehicle_data)} vehicles across {len(brands)} brands")
    logging.info(f"Scraping completed successfully. Total vehicles: {len(all_vehicle_data)}")
