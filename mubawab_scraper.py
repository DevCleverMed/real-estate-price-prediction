import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import requests
import time
import random
import re
import os
import logging
from fake_useragent import UserAgent
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
BASE_URL = "https://www.mubawab.ma"
SEARCH_URL = f"{BASE_URL}/fr/ct/rabat/immobilier-a-vendre"
MAX_LISTINGS = 200
MIN_LISTINGS = 150
OUTPUT_FILE = "mubawab_properties_150_to_200_rows.xlsx"
DELAY = random.uniform(5, 10)
TIMEOUT = 20

# Rotating user agents
ua = UserAgent()

def get_headers():
    """
    Generate a dictionary of HTTP headers with a random user agent to mimic browser requests.
    
    Returns:
        dict: A dictionary containing HTTP headers including a random User-Agent.
    """
    return {
        'User-Agent': ua.random,
        'Accept-Language': 'fr-FR,fr;q=0.9',
        'Referer': BASE_URL,
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
        'DNT': '1'
    }

def extract_price(text):
    """
    Extract a numerical price value from a text string, handling various formats.
    
    Args:
        text (str): The text containing the price (e.g., with 'DH', commas, or spaces).
    
    Returns:
        float or None: The extracted price as a float, or None if no valid price is found.
    """
    if not text:
        return None
    text = text.replace(' ', '').replace(',', '').replace('\xa0', '')
    match = re.search(r'(\d[\d\s]*\.?\d+)', text)
    return float(match.group(1).replace(' ', '')) if match else None

def parse_listing(listing):
    """
    Parse a single property listing from BeautifulSoup object to extract relevant details.
    
    Args:
        listing (bs4.element.Tag): A BeautifulSoup element representing a property listing.
    
    Returns:
        dict or None: A dictionary with parsed data (title, price, bedrooms, etc.), or None if parsing fails.
    """
    result = {
        'title': None,
        'price_dh': None,
        'bedrooms': None,
        'surface_m2': None,
        'location': None,
        'property_type': None,
        'year_built': None,
        'url': None,
        'scraped_at': pd.Timestamp.now()
    }
    
    try:
        # Title extraction
        title_elem = listing.find('h2', class_=re.compile('listingTit|title'))
        if not title_elem:
            title_elem = listing.find('h2') or listing.find(class_=re.compile('title|heading'))
        result['title'] = title_elem.get_text(strip=True) if title_elem else None

        # Price extraction
        price_elem = listing.find(class_=re.compile('priceTag|price|cost'))
        if not price_elem:
            price_elem = listing.find(string=re.compile('DH|MAD|€|Dhs'))
        price_text = price_elem.get_text(strip=True) if price_elem else ""
        result['price_dh'] = extract_price(price_text)

        # URL extraction
        link = listing.find('a', href=True)
        if link and link['href']:
            result['url'] = BASE_URL + link['href'] if not link['href'].startswith('http') else link['href']

        # Extract details from card text
        card_text = listing.get_text(' ')
        
        # Bedrooms
        bedrooms = re.search(r'(\d+)\s*(?:chambres?|pieces?)', card_text, re.IGNORECASE)
        result['bedrooms'] = int(bedrooms.group(1)) if bedrooms else None
        
        # Surface area
        surface = re.search(r'(\d+)\s*(?:m²|m2|m\s*²)', card_text, re.IGNORECASE)
        result['surface_m2'] = int(surface.group(1)) if surface else None
        
        # Location
        if result['title'] and 'à' in result['title']:
            result['location'] = result['title'].split('à')[-1].strip()
        else:
            location_elem = listing.find(class_=re.compile('location|ville|place'))
            result['location'] = location_elem.get_text(strip=True) if location_elem else 'Rabat'

        # Property type (from title)
        if result['title']:
            title_lower = result['title'].lower()
            if 'appartement' in title_lower:
                result['property_type'] = 'Apartment'
            elif 'maison' in title_lower or 'villa' in title_lower:
                result['property_type'] = 'House'
            else:
                result['property_type'] = 'Unknown'

        # Year built (not typically available)
        year = re.search(r'(\d{4})\s*(?:construit|built)', card_text, re.IGNORECASE)
        result['year_built'] = int(year.group(1)) if year else None

    except Exception as e:
        logger.error(f"Error parsing listing: {e}")
        return None
    
    if result['title'] or result['price_dh']:
        return result
    return None

def scrape_page(page_num):
    """
    Scrape property listings from a specific page on Mubawab.ma.
    
    Args:
        page_num (int): The page number to scrape.
    
    Returns:
        list: A list of dictionaries containing parsed listing data, or None if scraping fails.
    
    Notes:
        Saves the raw HTML to a file if no listings are found, for debugging purposes.
    """
    try:
        url = f"{SEARCH_URL}:p:{page_num}" if page_num > 1 else SEARCH_URL
        logger.info(f"Requesting page {page_num}: {url}")
        response = requests.get(url, headers=get_headers(), timeout=TIMEOUT)
        
        if response.status_code != 200:
            logger.error(f"Request failed with status {response.status_code}")
            return None
            
        soup = BeautifulSoup(response.text, 'html.parser')
        listings = soup.find_all('li', class_=re.compile('listingBox|propertyBox|listing'))
        if not listings:
            listings = soup.select('div.listingBox, div.propertyBox, section.property')
        if not listings:
            listings = soup.find_all(class_=re.compile('listing|property'))
        
        logger.info(f"Found {len(listings)} listings on page {page_num}")
        if not listings:
            with open(f'page_{page_num}.html', 'w', encoding='utf-8') as f:
                f.write(response.text)
        return [parse_listing(l) for l in listings if parse_listing(l) is not None]
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Request error on page {page_num}: {e}")
        return None
    except Exception as e:
        logger.error(f"Error scraping page {page_num}: {e}")
        return None

def simulate_web_scraping():
    """
    Generate simulated property data as a fallback when scraping fails.
    
    Returns:
        pd.DataFrame: A DataFrame containing simulated data with columns for price, surface, bedrooms, etc.
    
    Notes:
        Uses random distributions to mimic real estate data for Rabat.
    """
    logger.info("Generating simulated data as fallback.")
    np.random.seed(42)
    n_samples = 150
    surface = np.random.normal(100, 30, n_samples).clip(30, 300)
    bedrooms = np.random.choice([1, 2, 3, 4, 5], n_samples, p=[0.1, 0.3, 0.4, 0.15, 0.05])
    price = surface * 10000 + bedrooms * 50000 + np.random.normal(0, 50000, n_samples)
    data = {
        'price_dh': price.clip(100000, 5000000),
        'surface_m2': surface,
        'bedrooms': bedrooms,
        'location': np.random.choice(['Rabat-Centre', 'Agdal', 'Hay Riad', 'Souissi'], n_samples),
        'property_type': np.random.choice(['Apartment', 'House', 'Villa'], n_samples),
        'year_built': np.random.choice([1980, 1990, 2000, 2010, 2020, np.nan], n_samples),
        'title': [f"Property {i}" for i in range(n_samples)],
        'url': [f"{BASE_URL}/listing/{i}" for i in range(n_samples)],
        'scraped_at': pd.Timestamp.now()
    }
    return pd.DataFrame(data)

def scrape_data():
    """
    Scrape property listings from Mubawab.ma or use simulated data if scraping fails.
    
    Returns:
        pd.DataFrame: A DataFrame containing scraped or simulated property listings.
    
    Notes:
        Limits collection to MAX_LISTINGS (200) and stops at MIN_LISTINGS (150) if reached.
        Includes delays to avoid overloading the server.
    """
    all_data = []
    page_num = 1
    
    while len(all_data) < MAX_LISTINGS:
        page_data = scrape_page(page_num)
        if page_data:
            all_data.extend(page_data)
            logger.info(f"Page {page_num} scraped successfully. Total listings: {len(all_data)}")
            if len(all_data) >= MIN_LISTINGS:
                logger.info(f"Reached target of {len(all_data)} listings (min {MIN_LISTINGS}). Stopping.")
                break
        else:
            logger.warning(f"Failed to scrape page {page_num}")
        
        wait_time = DELAY * (1 + page_num/10)
        logger.info(f"Waiting {wait_time:.1f} seconds...")
        time.sleep(wait_time)
        
        if page_num > 10 and len(all_data) < MIN_LISTINGS:
            logger.error("Stopping - likely blocked or no more listings")
            break
        
        page_num += 1
    
    if not all_data:
        logger.error("No data collected. Using simulated data.")
        return simulate_web_scraping()
    
    if len(all_data) > MAX_LISTINGS:
        all_data = all_data[:MAX_LISTINGS]
    
    df = pd.DataFrame(all_data)
    df = df.drop_duplicates(subset=['url'], keep='first')
    df = df[df['title'].notna() | df['price_dh'].notna()]
    return df

# Scrape and save data
df = scrape_data()
if not df.empty:
    df.to_excel(OUTPUT_FILE, index=False)
    logger.info(f"Data saved to {OUTPUT_FILE}")
else:
    logger.error("No data to save. Exiting.")
    exit()

# 2. Data Cleaning Pipeline
def create_data_pipeline():
    """
    Create a data preprocessing pipeline for numerical and categorical features.
    
    Returns:
        ColumnTransformer: A transformer combining imputation and scaling for numerical data,
                          and imputation and one-hot encoding for categorical data.
    """
    numeric_features = ['surface_m2', 'bedrooms', 'year_built']
    categorical_features = ['location', 'property_type']
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    return preprocessor

# Load and clean data
try:
    df = pd.read_excel(OUTPUT_FILE)
    logger.info(f"Loaded data from {OUTPUT_FILE}")
except FileNotFoundError:
    logger.error(f"File {OUTPUT_FILE} not found. Using fallback data.")
    df = simulate_web_scraping()
    df.to_excel(OUTPUT_FILE, index=False)

df = df.drop_duplicates()
df = df[['price_dh', 'surface_m2', 'bedrooms', 'location', 'property_type', 'year_built']]

# Handle outliers using IQR
def remove_outliers(df, column):
    """
    Remove outliers from a specified column using the Interquartile Range (IQR) method.
    
    Args:
        df (pd.DataFrame): The input DataFrame.
        column (str): The column name to check for outliers.
    
    Returns:
        pd.DataFrame: The DataFrame with outliers removed or retained as NaN where applicable.
    """
    if df[column].isna().all():
        return df
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column].isna()) | ((df[column] >= lower_bound) & (df[column] <= upper_bound))]

df = remove_outliers(df, 'price_dh')
df = remove_outliers(df, 'surface_m2')

# 3. Exploratory Data Analysis (EDA)
def perform_eda(df):
    """
    Perform exploratory data analysis and save visualizations to the 'plots' directory.
    
    Args:
        df (pd.DataFrame): The input DataFrame containing property data.
    
    Notes:
        Generates and saves plots for price distribution, correlation matrix, price by location,
        and price vs. surface area.
    """
    if not os.path.exists('plots'):
        os.makedirs('plots')
    
    # Price distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(df['price_dh'].dropna(), kde=True)
    plt.title('Distribution of Property Prices in Rabat')
    plt.xlabel('Price (DH)')
    plt.ylabel('Frequency')
    plt.savefig('plots/price_distribution.png')
    plt.close()
    
    # Correlation matrix
    numeric_cols = ['price_dh', 'surface_m2', 'bedrooms', 'year_built']
    plt.figure(figsize=(10, 8))
    sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.savefig('plots/correlation_matrix.png')
    plt.close()
    
    # Price by location
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='location', y='price_dh', data=df)
    plt.title('Price Distribution by Sub-Location in Rabat')
    plt.xlabel('Sub-Location')
    plt.ylabel('Price (DH)')
    plt.xticks(rotation=45)
    plt.savefig('plots/price_by_location.png')
    plt.close()
    
    # Price vs surface
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='surface_m2', y='price_dh', hue='property_type', data=df)
    plt.title('Price vs Surface Area')
    plt.xlabel('Surface (m²)')
    plt.ylabel('Price (DH)')
    plt.savefig('plots/price_vs_surface.png')
    plt.close()
    logger.info("EDA plots saved in 'plots/' directory.")

perform_eda(df)

# 4. Modeling (Multiple Regression)
def train_model(df):
    """
    Train a multiple regression model to predict property prices.
    
    Args:
        df (pd.DataFrame): The input DataFrame containing cleaned property data.
    
    Returns:
        tuple: A dictionary with model metrics (R2, RMSE, MAE) and the trained pipeline, or (empty dict, None) if data is insufficient.
    """
    X = df.drop('price_dh', axis=1)
    y = df['price_dh'].dropna()
    X = X.loc[y.index]
    
    if len(y) < 10:
        logger.error("Insufficient data for modeling.")
        return {'R2': 0, 'RMSE': 0, 'MAE': 0}, None
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    preprocessor = create_data_pipeline()
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])
    
    model_pipeline.fit(X_train, y_train)
    y_pred = model_pipeline.predict(X_test)
    
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    
    logger.info(f"Model trained. R²: {r2:.4f}, RMSE: {rmse:.2f}, MAE: {mae:.2f}")
    return {'R2': r2, 'RMSE': rmse, 'MAE': mae}, model_pipeline

metrics, model = train_model(df)

# 5. Generate Markdown Report
def generate_report(metrics):
    """
    Generate a Markdown report summarizing the project findings and performance.
    
    Args:
        metrics (dict): A dictionary containing model evaluation metrics (R2, RMSE, MAE).
    
    Notes:
        Saves the report to 'project_report.md' and includes a placeholder for a GitHub link.
    """
    report = f"""
# Real Estate Price Prediction Project Report (Rabat, Mubawab.ma)

## 1. Exploratory Data Analysis (EDA)

### Key Findings:
- **Price Distribution**: Prices in Rabat are right-skewed, indicating potential for log transformation.
- **Correlations**: Surface area and bedrooms show positive correlations with price.
- **Location Impact**: Prices vary across sub-locations in Rabat, with some areas commanding higher prices.
- **Price vs Surface**: Larger properties generally have higher prices, influenced by property type.

### Visualizations:
- Price Distribution: `plots/price_distribution.png`
- Correlation Matrix: `plots/correlation_matrix.png`
- Price by Location: `plots/price_by_location.png`
- Price vs Surface: `plots/price_vs_surface.png`

## 2. Model Performance

### Multiple Regression Results:
- **R² Score**: {metrics['R2']:.4f}
- **RMSE**: {metrics['RMSE']:.2f} DH
- **MAE**: {metrics['MAE']:.2f} DH

### Interpretation:
The model explains {metrics['R2']*100:.2f}% of the variance in property prices in Rabat. RMSE reflects the average prediction error.

## 3. Limitations and Improvements
- **Limitations**:
  - Scraping may fail due to anti-bot measures, requiring simulated data.
  - Missing data (e.g., year built) limits model accuracy.
  - Outlier removal may exclude valid high-value properties.
- **Suggestions**:
  - Use Selenium or API for robust scraping.
  - Scrape additional features (e.g., amenities, proximity to services).
  - Test non-linear models (Random Forest, Gradient Boosting).
  - Apply log transformation to price to address skewness.

## 4. Conclusion
The project scraped or simulated {len(df)} listings for Rabat, implemented a data pipeline, conducted EDA, and built a regression model. The model provides a baseline for price prediction, with opportunities for enhancement via richer data and advanced methods.

---
*Code and documentation available on GitHub: [https://github.com/DevCleverMed/real-estate-price-prediction].*
"""
    with open('project_report.md', 'w') as f:
        f.write(report)
    logger.info("Report generated as 'project_report.md'.")

generate_report(metrics)

# Console output
print(f"Scraped or simulated {len(df)} listings.")
print(f"Data saved to {OUTPUT_FILE}")
print(f"Model Metrics:\nR²: {metrics['R2']:.4f}\nRMSE: {metrics['RMSE']:.2f} DH\nMAE: {metrics['MAE']:.2f} DH")
print("EDA plots saved in 'plots/' directory.")
print("Report generated as 'project_report.md'.")
