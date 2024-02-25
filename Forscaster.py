import requests
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
from sqlalchemy import create_engine

# Replace 'YOUR_SAP_API_KEY' and 'SAP_API_ENDPOINT' with your actual SAP API key and endpoint
headers = {
    'Authorization': 'Bearer YOUR_SAP_API_KEY',
    'Content-Type': 'application/json'
}

# Example: Fetch financial data
financial_api_endpoint = 'SAP_API_ENDPOINT/financial_data'
response_financial = requests.get(financial_api_endpoint, headers=headers)

# Example: Fetch labor cost data (hypothetical endpoint)
labor_cost_api_endpoint = 'SAP_API_ENDPOINT/labor_cost_data'
response_labor_cost = requests.get(labor_cost_api_endpoint, headers=headers)

# Check if the requests were successful (status code 200)
if response_financial.status_code == 200 and response_labor_cost.status_code == 200:
    financial_data = response_financial.json()
    labor_cost_data = response_labor_cost.json()
else:
    print(f"Error - Financial Data: {response_financial.status_code}")
    print(f"Error - Labor Cost Data: {response_labor_cost.status_code}")
    financial_data = None
    labor_cost_data = None

# Extract and transform financial data
financial_df = pd.DataFrame(financial_data)
financial_df = financial_df[['Year', 'Spending', 'Profits']]

# Extract and transform labor cost data
labor_cost_df = pd.DataFrame(labor_cost_data)
labor_cost_df = labor_cost_df[['Year', 'LaborCost']]

# Merge financial and labor cost data on 'Year'
merged_data = pd.merge(financial_df, labor_cost_df, on='Year')

# Save merged data to an SQLite database
engine = create_engine('sqlite:///financial_and_labor_data.db')
merged_data.to_sql('financial_and_labor_data', con=engine, index=False, if_exists='replace')

# Plotting financial and labor cost data over the years
plt.plot(merged_data['Year'], merged_data['Spending'], label='Spending')
plt.plot(merged_data['Year'], merged_data['Profits'], label='Profits')
plt.plot(merged_data['Year'], merged_data['LaborCost'], label='Labor Cost')
plt.xlabel('Year')
plt.ylabel('Amount')
plt.legend()
plt.show()

# Predict future trends for spending, profits, and labor cost
X = np.array(merged_data['Year']).reshape(-1, 1)
y_spending = np.array(merged_data['Spending'])
y_profits = np.array(merged_data['Profits'])
y_labor_cost = np.array(merged_data['LaborCost'])

model_spending = LinearRegression().fit(X, y_spending)
model_profits = LinearRegression().fit(X, y_profits)
model_labor_cost = LinearRegression().fit(X, y_labor_cost)

# Predict future trends
future_years = np.array([2025, 2026, 2027]).reshape(-1, 1)
predicted_spending = model_spending.predict(future_years)
predicted_profits = model_profits.predict(future_years)
predicted_labor_cost = model_labor_cost.predict(future_years)

# Plotting the predictions
plt.plot(merged_data['Year'], merged_data['Spending'], label='Spending')
plt.plot(merged_data['Year'], merged_data['Profits'], label='Profits')
plt.plot(merged_data['Year'], merged_data['LaborCost'], label='Labor Cost')
plt.plot(future_years, predicted_spending, 'o--', label='Predicted Spending')
plt.plot(future_years, predicted_profits, 'o--', label='Predicted Profits')
plt.plot(future_years, predicted_labor_cost, 'o--', label='Predicted Labor Cost')
plt.xlabel('Year')
plt.ylabel('Amount')
plt.legend()
plt.show()
