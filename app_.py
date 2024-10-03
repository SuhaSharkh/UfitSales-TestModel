import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
import xgboost
print(xgboost.__version__)
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
import joblib  # Import joblib for model loading
import streamlit as st
import statsmodels.api as sm  # Import statsmodels
import xgboost as xgb

# # Custom CSS for dark theme
# st.markdown(
#     """
#     <style>
#     body {
#         background-color: #000000 !important; /* Main background color */
#         color: #FFFFFF !important; /* Main text color */
#     }
#     .stButton>button {
#         background-color: #F63366; /* Button background color */
#         color: white; /* Button text color */
#     }
#     .sidebar .sidebar-content {
#         background-color: #1E1E1E; /* Sidebar background color */
#     }
#     </style>
#     """,
#     unsafe_allow_html=True
# )

st.title("Sales Analysis & Forecasting App")

# Sidebar for File Uploads
st.sidebar.title("Upload Your Sales File")
uploaded_file = st.sidebar.file_uploader("Choose CSV file", type=["csv"])

if uploaded_file:
    # Read the sales CSV file
    df_sales = pd.read_csv(uploaded_file, on_bad_lines='skip')
    #st.write("Sales Data Overview:", df_sales.head())

    # Clean 'Date' column
    if 'Date' in df_sales.columns:
        df_sales['Date'] = pd.to_datetime(df_sales['Date'], format='%d/%m/%Y', errors='coerce')
    else:
        st.sidebar.warning("'Date' column is missing from the sales data.")

    # Clean 'Ordered Product Sales' column
    df_sales['Ordered Product Sales'] = df_sales['Ordered Product Sales'].astype(str).str.replace(',', '', regex=False).str.strip()
    df_sales['Ordered Product Sales'] = df_sales['Ordered Product Sales'].str.replace(r'[^0-9.]', '', regex=True)
    df_sales['Ordered Product Sales'] = pd.to_numeric(df_sales['Ordered Product Sales'], errors='coerce')

    # Convert other columns
    columns_to_convert = ['Page Views - Total', 'Sessions - Total', 'Average Offer Count', 'Average Parent Items']
    percentage_columns = ['Featured Offer (Buy Box) Percentage', 'Unit Session Percentage']

    for col in columns_to_convert:
        df_sales[col] = df_sales[col].str.replace(',', '')
        df_sales[col] = pd.to_numeric(df_sales[col], errors='coerce').astype('Int64')

    for col in percentage_columns:
        df_sales[col] = df_sales[col].str.replace(',', '').str.replace('%', '')
        df_sales[col] = pd.to_numeric(df_sales[col], errors='coerce') / 100.0

    # Sort the DataFrame by 'Date' to ensure proper filtering
    df_sales = df_sales.sort_values('Date')

    # Add a slider to select a date range
    min_date = df_sales['Date'].min().date()  # Convert to date
    max_date = df_sales['Date'].max().date()  # Convert to date

    # Streamlit slider to choose date range
    start_date, end_date = st.slider(
        "Select a Date Range",
        min_value=min_date,
        max_value=max_date,
        value=(min_date, max_date),
        format="YYYY-MM-DD"
    )


        # Filter the DataFrame based on the selected date range
    filtered_df = df_sales[(df_sales['Date'] >= pd.Timestamp(start_date)) & (df_sales['Date'] <= pd.Timestamp(end_date))]


    # Display the filtered data (optional for debugging)
    #st.write("Filtered Sales Data", filtered_df)

    # Calculate the percentage change in sales over time
    filtered_df = filtered_df.set_index('Date')  # Set 'Date' as the index
    filtered_df['Sales Percentage Change'] = filtered_df['Ordered Product Sales'].pct_change() * 100  # Calculate percentage change

    # Reset index to include 'Date' as a column for plotting
    filtered_df = filtered_df.reset_index()
  

    # Plot the Ordered Product Sales over time
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(filtered_df['Date'], filtered_df['Ordered Product Sales'], label="Ordered Product Sales", color='blue')
    ax.set_title('Sales Over Time')
    ax.set_xlabel('Date')
    ax.set_ylabel('Ordered Product Sales')

    # Add percentage change on the same axis, but as a secondary y-axis
    ax2 = ax.twinx()
    ax2.plot(filtered_df['Date'], filtered_df['Sales Percentage Change'], label="Sales Percentage Change", color='red', linestyle='--')
    ax2.set_ylabel('Sales Percentage Change (%)')

    # Add legends
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')

    # Display the plot
    st.pyplot(fig)
    st.write("Numerical metrics: Provide more precise insights.")
    # Show a summary of sales and percentage change over time
    st.write("Sales Percentage Change Summary", filtered_df[['Date', 'Ordered Product Sales', 'Sales Percentage Change']])


    # Filter the DataFrame based on the selected date range
    filtered_df = df_sales[(df_sales['Date'] >= pd.Timestamp(start_date)) & (df_sales['Date'] <= pd.Timestamp(end_date))]
    
    # Display the filtered data (optional for debugging)
    st.write("Filtered Sales Data", filtered_df)


    # Calculate the Conversion Rate directly
    df_sales['Conversion Rate'] = df_sales['Units Ordered'] / df_sales['Page Views - Total'].astype(float)

    # Compute average conversion rates
    average_conversion_rate = df_sales['Conversion Rate'].mean()

    # Monthly average conversion rates
    df_sales.set_index('Date', inplace=True)
    monthly_conversion_rate = df_sales['Conversion Rate'].resample('M').mean()
    
    st.write("Conversion Rate Analysis")
    st.write(f"Average Conversion Rate: {average_conversion_rate:.2%}") 
    st.markdown("""Typical conversion rates for e-commerce sites range from 1% to 3%. However, high-performing sites can achieve 5% or higher. We have extremely low conversion rate. Less than 1%.""")

    # Plot Monthly Conversion Rates
    plt.figure(figsize=(10, 5))
    monthly_conversion_rate.plot(title='Average Monthly Conversion Rate', color='lightgreen')
    plt.xlabel('Month')
    plt.ylabel('Average Conversion Rate')
    plt.grid(True)
    st.pyplot(plt)

    # Decompose sales data using additive model (assuming seasonality)
    monthly_sales = df_sales['Ordered Product Sales'].resample('M').sum()  # Resample to monthly sales
    decomposition = seasonal_decompose(monthly_sales, model='additive', period=12)


    # Plot the Conversion Rate over time
    plt.figure(figsize=(10, 5))
    df_sales['Conversion Rate'].plot(title='Conversion Rate Over Time', color='lightblue')
    plt.xlabel('Date')
    plt.ylabel('Conversion Rate')
    plt.grid(True)
    st.pyplot(plt)


    # # Correlation matrix for selected columns
    # corr_matrix = df_sales[['Unit Session Percentage', 'Units Ordered']].corr()
    # st.write("Correlation Matrix:", corr_matrix)
    # corr_matrix1 = df_sales[['Featured Offer (Buy Box) Percentage', 'Ordered Product Sales']].corr()
    # st.write("Correlation Matrix:", corr_matrix1)
    # corr_matrix2 = df_sales[['Sessions - Total', 'Ordered Product Sales']].corr()
    # st.write("Correlation Matrix:", corr_matrix2)
    
#Let the user choose the variables for Correlation analysis instead of above

# Sidebar for user input
    st.sidebar.header("Select Variables for Correlation Analysis")
    variables = df_sales.columns.tolist()
    selected_vars = st.sidebar.multiselect("Choose variables to see correlation:", variables)


    if selected_vars:
        # Display selected variables correlation matrix
        corr_matrix = df_sales[selected_vars].corr()
        st.write("Correlation Matrix:", corr_matrix)

   
        # Optional: Display a heatmap for better visualization
        st.subheader("Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
        st.pyplot(fig)

        # Example Markdown description
        if 'Sessions - Total' in selected_vars and 'Ordered Product Sales' in selected_vars:
            st.markdown("""The correlation between "Sessions - Total" and "Ordered Product Sales" is 0.42, which indicates a moderate positive correlation. This suggests that as the number of sessions increases, sales tend to increase as well, but the relationship is not extremely strong. Other factors might be affecting sales more strongly. A correlation of 0.419 suggests that sessions do influence sales, but the relationship is not direct or linear. The scatter plot shows a weak relationship; points are more spread out, and the relationship is weaker. We could further explore how well sessions convert into sales by looking at Unit Session Percentage, which shows how many units were sold per session. This would give a better idea of whether higher sessions lead to better conversions or if just a few sessions are leading to the majority of the sales.""")
    else:
        st.subheader("Please select at least one variable, from the sidebar, to see the correlation.")

        # Define the most important features, dropping the Sales column
        important_features = ['Units Ordered', 'Featured Offer (Buy Box) Percentage', 'Average Offer Count', 'Sessions - Total', 'Page Views - Total']
    
        # Define the dependent variable (target)
        y = df_sales['Ordered Product Sales']

        # Function to create scatter plot with regression line
        def reg_plot_on_ax(feature, ax):
            sns.regplot(x=df_sales[feature], y=y, ax=ax, scatter_kws={"s": 10}, line_kws={"color": "red"})
            ax.set_title(f'{feature} vs. Ordered Product Sales')
            ax.set_xlabel(feature)
            ax.set_ylabel('Ordered Product Sales')

     # Create a 3x2 subplot (for 6 features)
        fig, ax = plt.subplots(3, 2, figsize=(14, 12))  # 3x2 grid for 6 features
        ax = ax.flatten()

        # Plot the most important features
        for i, feature in enumerate(important_features):
            if i < len(ax):  # Check to avoid index error
                reg_plot_on_ax(feature, ax[i])

        # Leave the last subplot empty
        ax[-1].axis('off')  # Turn off the last subplot since we have only 5 features

        # Adjust layout for better spacing
        plt.tight_layout()
    
        # Display the regression plots
        st.subheader("Feature Regression Plots")
        st.pyplot(fig)

    #Prediction Model
    #Issues with Date Column - Debugging for Date - Reloading the File
    # Load your DataFrame 
    df_sales = pd.read_csv('BusinessReport-9-25-24-SalesTraffic.csv')

    # Clean the 'Ordered Product Sales' column
    def clean_sales_data(sales_column):
        return sales_column.str.replace('AED', '', regex=False).str.replace(',', '', regex=False).astype(float)

    # Clean the sales data
    df_sales['Ordered Product Sales'] = clean_sales_data(df_sales['Ordered Product Sales'])

    # Convert 'Date' column to datetime
    df_sales['Date'] = pd.to_datetime(df_sales['Date'])
    df_sales.set_index('Date', inplace=True)
    
    # Create a monthly sales time series
    monthly_sales = df_sales.resample('M')['Ordered Product Sales'].sum()

    # Streamlit layout
    st.title("Sales Forecasting Dashboard")

    # Seasonal Decomposition
    st.subheader("Seasonal Decomposition of Sales Data")
    st.markdown("""Evaluate if there is a trend, seasonal pattern and the residuals to help analyze the decomposition if it captures the major trends and seasonal effects well.""")
    decomposition = seasonal_decompose(monthly_sales, model='additive')
    fig, axes = plt.subplots(4, 1, figsize=(10, 10))
    decomposition.observed.plot(ax=axes[0], title='Observed', xlabel='Date', ylabel='Sales')
    decomposition.trend.plot(ax=axes[1], title='Trend', xlabel='Date', ylabel='Sales')
    decomposition.seasonal.plot(ax=axes[2], title='Seasonal', xlabel='Date', ylabel='Sales')
    decomposition.resid.plot(ax=axes[3], title='Residual', xlabel='Date', ylabel='Sales')
    plt.tight_layout()
    st.pyplot(fig)

    # ADF Test
    result = adfuller(monthly_sales)
    st.write('ADF Statistic:', result[0])
    st.write('p-value:', result[1])


    # Streamlit Layout
    st.title("Sales Forecasting App")

    # Step 1: Split data into training and testing sets
    st.subheader("Train/Test Split")
    split_ratio = st.slider("Choose Train/Test Split Ratio", min_value=0.5, max_value=0.95, value=0.8)
    split_index = int(len(monthly_sales) * split_ratio)

    train_data = monthly_sales[:split_index]
    test_data = monthly_sales[split_index:]

    st.write(f"Training data: {len(train_data)} observations")
    st.write(f"Testing data: {len(test_data)} observations")

    # Step 2: ADF Test on training data
    result_log_diff = adfuller(train_data)
    st.write('ADF Statistic (Log Differenced):', result_log_diff[0])
    st.write('p-value (Log Differenced):', result_log_diff[1])

    # Step 3: Seasonal Differencing
    seasonal_differenced_sales = train_data.diff(12).dropna()  # Assuming monthly data with yearly seasonality
    result_seasonal_diff = adfuller(seasonal_differenced_sales)
    st.write('ADF Statistic (Seasonal Differenced):', result_seasonal_diff[0])
    st.write('p-value (Seasonal Differenced):', result_seasonal_diff[1])

    # Step 4: Further Differencing
    if result_seasonal_diff[1] > 0.05:
        further_differenced_sales = seasonal_differenced_sales.diff().dropna()
        result_further_diff = adfuller(further_differenced_sales)
        st.write('ADF Statistic (Further Differenced):', result_further_diff[0])
        st.write('p-value (Further Differenced):', result_further_diff[1])

    # Step 5: Fit SARIMA model if stationary
    if result_log_diff[1] <= 0.05 or result_seasonal_diff[1] <= 0.05:
        st.write("The log-differenced or seasonal differenced series is stationary. Fitting SARIMA model...")
    
       # Train the SARIMA model (adjust the (p,d,q)(P,D,Q,s) parameters as needed)
        model = sm.tsa.statespace.SARIMAX(train_data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
        results = model.fit()

        # Save the SARIMA model using joblib
        joblib.dump(results, 'new_sarima_model.pkl')

        print("SARIMA model trained and saved successfully.")

        # Print summary
        st.write(results.summary())
    
        # Forecasting
        forecast = results.get_forecast(steps=len(test_data))
        forecast_index = test_data.index
        forecast_values = forecast.predicted_mean

        # Plot forecast
        plt.figure(figsize=(12, 6))
        plt.plot(train_data, label='Train Data', color='blue')
        plt.plot(test_data, label='Test Data', color='orange')
        plt.plot(forecast_index, forecast_values, label='Forecast', color='green')
        plt.title('Sales Forecast')
        plt.xlabel('Date')
        plt.ylabel('Ordered Product Sales')
        plt.legend()
        st.pyplot(plt)

        # Calculate error metrics
        mse = mean_squared_error(test_data, forecast_values)
        rmse = np.sqrt(mse)
        st.write(f"Test RMSE: {rmse}")
        

        # Load the saved SARIMA model
        model = joblib.load('new_sarima_model.pkl')

        def make_prediction(n_steps):
        # Forecast with the loaded SARIMA model
            forecast = model.get_forecast(steps=n_steps)
            predicted_values = forecast.predicted_mean
            return predicted_values

    else:
        st.warning("The time series is non-stationary. Please check your data.")

    # Fit the XGBoost Model
    # Prepare data for XGBoost
    # def create_lagged_features(data, n_lags=1):
    #     for i in range(1, n_lags + 1):
    #         data[f'lag_{i}'] = data['y'].shift(i)
    #     return data.dropna()

    # data_lagged = create_lagged_features(monthly_sales.to_frame(name='y'), n_lags=12)
    # X = data_lagged.drop('y', axis=1)
    # y = data_lagged['y']
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # # Fit the XGBoost Model
    # xgb_model = XGBRegressor()
    # xgb_model.fit(X_train, y_train)

    # # Evaluate predictions for XGBoost
    # xgb_predictions = xgb_model.predict(X_test)
    # xgb_rmse = mean_squared_error(y_test, xgb_predictions, squared=False)
    # st.write(f'XGBoost RMSE: {xgb_rmse}')
    # Create a lag feature


    from sklearn.metrics import mean_squared_error
    import numpy as np
    # Check for missing values in Ordered Product Sales
    df_sales = df_sales.dropna(subset=['Ordered Product Sales'])

    # Train-test split for time series data
    train_size = int(len(monthly_sales) * 0.8)
    train_data, test_data = monthly_sales[:train_size], monthly_sales[train_size:]



    #XGBoost Prediction
    def create_lagged_features(data, n_lags=1):
        lagged_data = pd.concat([data.shift(i) for i in range(1, n_lags + 1)], axis=1)
        lagged_data.columns = [f"lag_{i}" for i in range(1, n_lags + 1)]
        return lagged_data

    # Prepare the features
    lagged_sales = create_lagged_features(monthly_sales, n_lags=3)  # Lag of 3 months
    lagged_sales.dropna(inplace=True)

    X = lagged_sales  # Lagged sales values as features
    y = monthly_sales.loc[lagged_sales.index]  # Corresponding target sales values

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # # Initialize and fit the model
    # xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1)
    # xgb_model.fit(X_train, y_train)
    
    
        # Train the XGBoost model
    xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, max_depth=5)
    xgb_model.fit(X_train, y_train)
    # Make predictions on the test set
    y_pred = xgb_model.predict(X_test)
    # Save the trained XGBoost model
    joblib.dump(xgb_model, 'xgboost_model.pkl')

    print("XGBoost model trained and saved successfully.")

    import joblib
    import pandas as pd

    # Load the saved XGBoost model
    xgb_model = joblib.load('xgboost_model.pkl')

    # Example function to make predictions using the XGBoost model
    def make_xgb_prediction(input_data):
        prediction = xgb_model.predict(input_data)
        return prediction

    # Evaluate the model using MSE or RMSE
    mse = mean_squared_error(y_test, y_pred)
    xgb_rmse = np.sqrt(mse) 

    st.write(f"Test RMSE: {xgb_rmse:.2f}")

    # Plot the actual vs predicted sales
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(monthly_sales.index[-len(y_test):], y_test, label="Actual Sales", color='blue')
    ax.plot(monthly_sales.index[-len(y_test):], y_pred, label="Predicted Sales", color='red')
    ax.set_title('Actual vs Predicted Sales')
    ax.set_xlabel('Date')
    ax.set_ylabel('Sales')
    ax.legend()

    st.pyplot(fig)



    # Adding text to the sidebar
    st.sidebar.header("About This App")
    st.sidebar.write(
    "This application performs sales forecasting using various models, including SARIMA and XGBoost. "
    "You can explore different forecasting techniques and see the results on the main page."
)

    # Add model selection to the sidebar
    st.sidebar.subheader("Model Selection")
    model_choice = st.sidebar.selectbox("Choose a model:", ["SARIMA", "XGBoost"])

    # Main content of your app
    if model_choice == "SARIMA":
        st.write("You have selected the SARIMA model.")
        st.write("Summary of SARIMA model:")
        st.write(results.summary())
        # Forecasting plot for SARIMA
        st.write("SARIMA Forecast:")
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(monthly_sales, label='Observed')
        ax.plot(forecast_index, forecast_values, label='Forecast', color='orange')
        ax.set_title('Sales Forecast')
        ax.set_xlabel('Date')
        ax.set_ylabel('Ordered Product Sales')
        ax.legend()
        st.pyplot(fig)

    elif model_choice == "XGBoost":
        st.write("You have selected the XGBoost model.")
        st.write("XGBoost RMSE:", xgb_rmse)
        st.write("XGBoost predictions (sample):", y_pred[:10])

    # Create a summary table
    st.write("Model Comparison:")
    comparison_df = pd.DataFrame({
        'Model': ['SARIMA', 'XGBoost'],
        'RMSE': [rmse, xgb_rmse]
})
    st.write(comparison_df)


    # Add a link to GitHub repo
    st.markdown('[Check out the project on GitHub](https://github.com/SuhaSharkh/UfitSales-TestModel)', unsafe_allow_html=True)

       



    # # Load your SARIMA model
    # model = joblib.load('sarima_model.pkl')

    # def make_prediction(n_steps):
    #     # Forecasting with the SARIMA model
    #     forecast = model.get_forecast(steps=n_steps)
    #     predicted_values = forecast.predicted_mean
    #     return predicted_values

    # Streamlit Layout
    # st.title("Sales Forecasting App")

    # # User input for forecasting
    # date_input = st.date_input("Select a date for prediction:")
    # n_steps = st.number_input("Number of months to forecast:", min_value=1, max_value=24, value=12)

    # if st.button("Make Prediction"):
    #     # Make prediction
    #     prediction = make_prediction(n_steps)

    #     # Create a date range for the forecast
    #     forecast_dates = pd.date_range(start=date_input, periods=n_steps, freq='M')

    #     # Display the predictions
    #     st.write("Predicted Sales for the next months:")
    #     prediction_df = pd.DataFrame({'Date': forecast_dates, 'Predicted Sales': prediction})
    #     st.line_chart(prediction_df.set_index('Date'))

    # # Placeholder for XGBoost prediction
    # st.subheader("XGBoost Model Prediction")

    # # Function to generate XGBoost prediction
    # def make_xgb_prediction(model, input_data):
    #     # Ensure input_data is a valid DataFrame or NumPy array
    #     prediction = model.predict(input_data)
    #     return prediction

    # # Assuming we need some input data for XGBoost
    # if st.button("Make XGBoost Prediction"):
    #     try:
    #         # Example: Preparing input data (adjust as per your dataset and features)
    #         input_data = data_lagged.drop('y', axis=1)  # Use your actual test data or features
        
    #         # Check if the model has been loaded
    #         xgb_model = joblib.load('xgboost_model.pkl')
        
    #         # Make prediction
    #         xgb_prediction = make_xgb_prediction(xgb_model, input_data)

    #         # Display prediction results
    #         st.write("Predicted Sales (XGBoost):")
    #         st.write(xgb_prediction[:10])  # Display first 10 predictions as a sample

    #     except Exception as e:
    #         st.error(f"An error occurred: {e}")

    # # Save and load models
    # if st.button("Save Models"):
    #     if model and 'xgb_model' in locals():
    #         joblib.dump(model, 'sarima_model.pkl')
    #         joblib.dump(xgb_model, 'xgboost_model.pkl')
    #         st.success("Models saved successfully.")
    #     else:
    #         st.error("Models are not trained or loaded yet.")

    # if st.button("Load Models"):
    #     sarima_model = joblib.load('sarima_model.pkl')
    #     xgb_model = joblib.load('xgboost_model.pkl')
    #     st.success("Models loaded successfully.")


    # try:
    #     model = joblib.load('sarima_model.pkl')
    # except Exception as e:
    #     print(f"Error loading model: {e}")

    import streamlit as st
    import pandas as pd

# Load the saved XGBoost model
    xgb_model = joblib.load('xgboost_model.pkl')

# Streamlit Layout
    if __name__ == "__main__":
        st.title("Sales Forecasting App")

        # User input for forecasting
        date_input = st.date_input("Select a date for prediction:")
        n_steps = st.number_input("Number of months to forecast:", min_value=1, max_value=24, value=12)
        st.subheader("Sarima Model Prediction")
        if st.button("Make Sarima Prediction"):
            # Prediction with SARIMA (assuming SARIMA is already set up)
            prediction = make_prediction(n_steps)
            forecast_dates = pd.date_range(start=date_input, periods=n_steps, freq='M')
            prediction_df = pd.DataFrame({'Date': forecast_dates, 'Predicted Sales': prediction})
            st.write("Predicted Sales for the next months:")
            st.line_chart(prediction_df.set_index('Date'))

        # Placeholder for XGBoost prediction
        st.subheader("XGBoost Model Prediction")

        # Example input data for XGBoost (this depends on your dataset)
        if st.button("Make XGBoost Prediction"):
            try:
                # Example: Preparing input data (adjust as per your dataset and features)
                # Assume 'input_data' contains the features to predict future sales
                input_data = X_test.head(10)  # Replace with actual input data

                # Make XGBoost prediction
                xgb_prediction = make_xgb_prediction(input_data)

                # Display prediction results
                #st.write("Predicted Sales (XGBoost):")
                #st.write(xgb_prediction[:10])  # Display first 10 predictions as a sample
                 # Create a DataFrame for visualization
                prediction_dates = pd.date_range(start=pd.Timestamp.now(), periods=len(xgb_prediction), freq='M')
                prediction_df = pd.DataFrame({'Date': prediction_dates, 'Predicted Sales': xgb_prediction})

                # Display the predictions in a line chart
                st.line_chart(prediction_df.set_index('Date'))

            except Exception as e:
                st.error(f"An error occurred: {e}")

    import joblib

    # Save and load models buttons in Streamlit
    if st.button("Save Models"):
        joblib.dump(results, 'sarima_model.pkl')  # SARIMA model
        joblib.dump(xgb_model, 'xgboost_model.pkl')  # XGBoost model
        st.success("Models saved successfully.")

    if st.button("Load Models"):
        sarima_model = joblib.load('sarima_model.pkl')
        xgb_model = joblib.load('xgboost_model.pkl')
        st.success("Models loaded successfully.")



else:
    # Display a warning if no file is uploaded
    st.warning("Please upload a file to start the analysis.")
