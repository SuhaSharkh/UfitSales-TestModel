# UfitSales-TestModel
Final Capstone Project - Learning Experience
# Amazon Seller Data Analysis Project

## I. Introduction
This project aims to analyze data from an Amazon Seller account based in the UAE, focusing on a variety of products sold. The goal is to gain insights into sales trends, correlations between different variables, and net profits, with the potential to develop a predictive model.

## II. Project Objective
The primary objectives of this project are to:
- Create a user-friendly dashboard that displays sales trends, correlations, and forecasts sales.

## III. Data Description
- **Dataset(s):** Business data sourced from an Amazon Seller account.
- **Data Sources:** The data will be pulled from various documents including CSV, TXT, and Excel files.(for now only one csv file to study)
- **Data Structure:** The data consists of multiple formats that require loading, cleaning and potentially merging for comprehensive analysis.

## IV. Methodology
The approach to accomplish the project objectives includes:
1. **Data Loading:** Importing data from various sources (may require merging).
2. **Data Cleaning:** Preparing data for analysis by addressing inconsistencies and missing values.
3. **Data Analysis:** Exploring correlations between different variables using analytical techniques.
4. **Data Visualization:** Creating visual representations of the data trends and insights.
5. **Model Evaluation:** Potentially developing and evaluating a predictive model. Forecasting Sales.

## V. Expected Deliverables
- An interactive visual dashboard that provides insights into:
- Sales trends
- A comprehensive analysis report detailing the findings and visualizations.

## VI. Potential Challenges
- **Challenges:**
  - Data cleaning may present challenges due to inconsistencies.
  - Familiarizing with Streamlit for dashboard development. 
  
- **Proposed Solutions:**
  - Load and clean each dataset systematically before analysis.
  - Utilize resources and documentation to guide Streamlit usage. https://github.com/okld/streamlit-elements https://docs.streamlit.io/develop/api-reference

## VII. Libraries Used
The following libraries are utilized in this project:
- `pandas`: For data manipulation and analysis.
- `matplotlib` and `seaborn`: For data visualization.
- `statsmodels`: For statistical modeling and time series analysis.
- `numpy`: For numerical operations.
- `xgboost`: For predictive modeling.
- `sklearn`: For machine learning and model evaluation.
- `joblib`: For model loading.

## VIII. More work to be done
The following are ongoing unresolved issues
- Analyze two datasets: one focusing on sales and the other on net profits, to visualize and better understand the impact of Amazon's fees on product profitability.
- Identify which products yield the highest net profits for sellers, allowing for informed decision-making regarding product focus.
-  We are working with two documents- excel and data from amazon and/or other platforms. We want to be able to merge the data from different platforms to the excel sheet at the end of every working day, that is Monday to Saturday ( Inventory and calculate net profit.) Consolidate data from different platforms and automate process.
-  Re-work the prediction models for better results

## Live App
Check out the deployed version of the app [[here](https://your-streamlit-app-link)](https://ufitsales-testmodel-xh6wmc2nxdhrwwx9vkhrul.streamlit.app/)](https://ufitsales-testmodel-xh6wmc2nxdhrwwx9vkhrul.streamlit.app/).
