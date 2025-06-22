
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
- **R� Score**: -3.7206
- **RMSE**: 6602233.26 DH
- **MAE**: 4276685.08 DH

### Interpretation:
The model explains -372.06% of the variance in property prices in Rabat. RMSE reflects the average prediction error.

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
The project scraped or simulated 146 listings for Rabat, implemented a data pipeline, conducted EDA, and built a regression model. The model provides a baseline for price prediction, with opportunities for enhancement via richer data and advanced methods.

---
*Code and documentation available on GitHub: https://github.com/DevCleverMed/real-estate-price-prediction .*
