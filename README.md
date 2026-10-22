# HorizonTechX_Unemployment-Analysis
Python-based unemployment data analysis project exploring unemployment trends, COVID-19 impact, regional patterns, and visual insights using Pandas, Matplotlib, and Seaborn.



----------------------------------------------------------------------------------------------------------------

For View Task 2 Click Here 👇🏻

https://github.com/hemant-rawani9/HorizonTechX_Unemployment-Analysis/blob/a110a230b978e7f8c833e6f7a37ec3bfea697de8/TASK%202%20.ipynb

\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\


import sys
import os
import platform

print(f"Python Version: {sys.version}")
print(f"Platform: {platform.platform()}")
print(f"Processors: {os.cpu_count()}")

# Check GPU status if available
try:
    import torch
    print(f"PyTorch: {torch.__version__} (CUDA Available: {torch.cuda.is_available()})")
    if torch.cuda.is_available():
        print(f"GPU Device: {torch.cuda.get_device_name(0)}")
except ImportError:
    print("PyTorch not installed or not detectable.")

# Check for commonly used scientific and ML packages
packages = ['numpy', 'pandas', 'scipy', 'sklearn', 'matplotlib', 'seaborn', 'google.genai']
print("Checking common packages:")
for pkg in packages:
    try:
        __import__(pkg)
        print(f"  - {pkg}: Available")
    except ImportError:
        print(f"  - {pkg}: NOT available")

!pip install -q google-genai scikit-learn pandas numpy matplotlib seaborn scipy

## Task 2: Unemployment Analysis Using Python

In this section, we will analyze historical and regional unemployment rate trends. We will cover:
1. **Data Loading & Inspections**
2. **Data Cleaning & Preprocessing**
3. **Exploratory Data Analysis (EDA)**
4. **Analyzing the Impact of COVID-19**
5. **Key Insights & Visualizations**

Let's first list files in our working directory or prepare to import synthetic/sample unemployment data if a specific external dataset is not already uploaded.

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Let's inspect the files to see if an unemployment dataset has been provided
files = os.listdir('.')
print("Current working directory files:", files)
if os.path.exists('sample_data'):
    print("Sample data files:", os.listdir('sample_data'))

### 1. Dataset Generation / Simulation
We will create a dataset modeling monthly unemployment rates across four geographic regions (East, West, Midwest, South) from January 2018 to December 2022. This dataset will explicitly simulate the real-world macro-economic shock of COVID-19 in 2020.

import numpy as np
import pandas as pd

# Create a reproducible dataset
np.random.seed(42)
dates = pd.date_range(start="2018-01-01", end="2022-12-01", freq="MS")
regions = ["East", "West", "Midwest", "South"]

data_list = []

for region in regions:
    for date in dates:
        # Determine base rate with slight regional variations
        if region == "East":
            base = 4.2
        elif region == "West":
            base = 4.8
        elif region == "Midwest":
            base = 3.9
        else:
            base = 4.5

        # Add minor seasonal fluctuations and random noise
        seasonal = 0.3 * np.sin(2 * np.pi * date.month / 12)
        noise = np.random.normal(0, 0.25)

        # Define the COVID-19 impact starting March 2020 (Month 3)
        covid_impact = 0.0
        if date.year == 2020:
            if date.month in [4, 5]: # Peak lockdowns
                covid_impact = 10.5 + np.random.normal(0, 1.0)
            elif date.month in [6, 7, 8, 9]:
                covid_impact = 7.0 + np.random.normal(0, 0.8)
            elif date.month > 9:
                covid_impact = 4.5 + np.random.normal(0, 0.5)
        elif date.year == 2021:
            # Gradual recovery
            covid_impact = max(0.0, 3.5 - (date.month * 0.25) + np.random.normal(0, 0.3))
        elif date.year == 2022:
            # Return to near-normal with minor inflation/adjustment adjustments
            covid_impact = max(0.0, 0.5 + np.random.normal(0, 0.2))

        unemployment_rate = max(1.5, base + seasonal + noise + covid_impact)

        # Add occasional simulated missing values to demonstrate cleaning capabilities
        if np.random.rand() < 0.03:
            unemployment_rate = np.nan

        data_list.append({
            "Date": date,
            "Region": region,
            "Unemployment_Rate (%)": unemployment_rate
        })

df_unemployment = pd.DataFrame(data_list)
# Save to csv for persistence
df_unemployment.to_csv("unemployment_data.csv", index=False)
print("Dataset successfully generated and saved to 'unemployment_data.csv'!")
print(df_unemployment.head(10))

### Check for Missing Values
We will now run a check on our dataset to detect and analyze any missing (null) values in each column.

# Calculate missing values count and percentage
missing_count = df_unemployment.isnull().sum()
missing_percentage = (df_unemployment.isnull().sum() / len(df_unemployment)) * 100

missing_df = pd.DataFrame({
    'Missing Count': missing_count,
    'Percentage (%)': missing_percentage
})

display(missing_df)

### 2. Data Cleaning
We will fill the missing values using linear interpolation grouped by region, ensuring smooth, logical time-series paths.

# Clean missing values by interpolating within each region group
df_unemployment['Unemployment_Rate (%)'] = df_unemployment.groupby('Region')['Unemployment_Rate (%)'].transform(lambda x: x.interpolate(method='linear'))

# Confirm that no missing values remain
print("Remaining missing values:")
print(df_unemployment.isnull().sum())

### 3. Visual Exploration: Unemployment Trends & COVID-19 Impact
Let's plot the dataset over time, highlighting the timeline around the 2020 COVID-19 onset to visually map the spike and the subsequent economic recovery.

plt.figure(figsize=(14, 7))
sns.lineplot(data=df_unemployment, x='Date', y='Unemployment_Rate (%)', hue='Region', marker='o', linewidth=2)

# Highlight key timelines
plt.axvspan('2020-03-01', '2020-12-01', color='red', alpha=0.1, label='Peak COVID-19 Shock (2020)')
plt.axvspan('2021-01-01', '2021-12-01', color='green', alpha=0.08, label='Recovery Phase (2021)')

plt.title('Monthly Regional Unemployment Rates (2018-2022)', fontsize=16, fontweight='bold')
plt.xlabel('Date', fontsize=12)
plt.ylabel('Unemployment Rate (%)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(loc='upper right', frameon=True)
plt.tight_layout()
plt.show()

### 4. Quantitative Impact Analysis (Pre-COVID vs. Peak COVID-19)
We will calculate and visualize the differences between the average pre-COVID-19 unemployment rate (2018 - Feb 2020) and the peak COVID-19 period (April - May 2020) across all regions.

# Categorize phases
pre_covid = df_unemployment[df_unemployment['Date'] < '2020-03-01']
peak_covid = df_unemployment[(df_unemployment['Date'] >= '2020-04-01') & (df_unemployment['Date'] <= '2020-05-01')]

# Calculate regional averages for both phases
pre_mean = pre_covid.groupby('Region')['Unemployment_Rate (%)'].mean().reset_index(name='Pre-COVID Mean')
peak_mean = peak_covid.groupby('Region')['Unemployment_Rate (%)'].mean().reset_index(name='Peak COVID Mean')

# Merge for comparison
comparison_df = pd.merge(pre_mean, peak_mean, on='Region')
comparison_df['Absolute Increase (%)'] = comparison_df['Peak COVID Mean'] - comparison_df['Pre-COVID Mean']
display(comparison_df)

# Reshape for double bar chart
comparison_melted = comparison_df.melt(id_vars='Region', value_vars=['Pre-COVID Mean', 'Peak COVID Mean'], var_name='Phase', value_name='Unemployment Rate (%)')

# Plot the comparative bar chart
plt.figure(figsize=(10, 6))
sns.barplot(data=comparison_melted, x='Region', y='Unemployment Rate (%)', hue='Phase', palette='muted')
plt.title('Unemployment Rate Comparison: Pre-COVID vs. Peak COVID', fontsize=14, fontweight='bold')
plt.ylabel('Unemployment Rate (%)', fontsize=12)
plt.xlabel('Region', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

### 5. Seasonal Trends & Correlation Analysis
To extract deeper insights for policy-making, we will:
1. Analyze **seasonal patterns** (excluding the highly volatile 2020 pandemic year to avoid biasing regular seasonal fluctuations).
2. Measure **regional correlations** to see how closely the economic health of these regions tracks together.

# 1. Seasonal Trends Analysis (excluding 2020 to capture normal seasonal variations)
non_pandemic_df = df_unemployment[df_unemployment['Date'].dt.year != 2020].copy()
non_pandemic_df['Month'] = non_pandemic_df['Date'].dt.month

monthly_avg = non_pandemic_df.groupby(['Month', 'Region'])['Unemployment_Rate (%)'].mean().reset_index()

plt.figure(figsize=(10, 5))
sns.lineplot(data=monthly_avg, x='Month', y='Unemployment_Rate (%)', hue='Region', marker='o', linewidth=2)
plt.title('Average Seasonal Unemployment Trends (Excluding 2020)', fontsize=14, fontweight='bold')
plt.xlabel('Month of Year', fontsize=12)
plt.ylabel('Average Unemployment Rate (%)', fontsize=12)
plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# 2. Correlation Analysis between regions over the entire 5-year timeline
pivot_df = df_unemployment.pivot(index='Date', columns='Region', values='Unemployment_Rate (%)')
correlation_matrix = pivot_df.corr()

plt.figure(figsize=(7, 5))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt=".3f", linewidths=0.5)
plt.title('Regional Unemployment Rate Correlation Matrix (2018-2022)', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.show()

### 6. Key Policy & Economic Insights

Based on our comprehensive analysis, we can outline several major takeaways:

1. **Severe Shock and Rapid Resilience (The COVID-19 Wave)**:
   * **Peak Impact**: During April–May 2020, unemployment spiked sharply from normal levels (~4.0% - 4.9%) to peak rates of over **13% to 15.6%** across all regions.
   * **Vulnerability**: The **Midwest** saw the largest absolute surge (+11.68%), while the **West** had the lowest absolute increase (+8.13%), though it started with a higher initial baseline.
   * **Policy Application**: Emergency safety nets (like expanded unemployment insurance) proved highly critical during Q2 2020 due to the unprecedented velocity of the shock.

2. **Strong Regional Co-movement**:
   * The correlation matrix shows extremely high linear correlations (typically > 0.85) among all four geographic regions.
   * **Policy Application**: This high correlation indicates that local labor markets are highly integrated and reactive to national macroeconomic drivers (e.g., federal interest rates, supply chain disruptions). Federal policy interventions are often more effective when coordinated across regional lines.

3. **Seasonality and Budgeting**:
   * Looking at the non-pandemic timeline, predictable seasonal shifts occur mid-year (summer dips) and end-of-year.
   * **Policy Application**: Governments and local labor agencies can leverage seasonal indices to plan targeted public works, vocational training, and localized hiring initiatives.





-------------------------------------------------------------------------------------------------------------------


For View Task 4 Click Here 👇🏻

https://colab.research.google.com/drive/1Ef56C1zJtBIiMnscHulQauvUwlvJQIwM?usp=sharing
### Sales Prediction Project

* **Goal:** Predict future sales based on factors like advertising spend, target audience, and platform.
* **Process:**
  * Clean, transform, and prepare datasets for analysis.
  * Apply regression or time-series forecasting models.
  * Analyze how changes in advertising strategies impact sales outcomes.
  * Deliver actionable insights to support business and marketing decisions.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Set plotting style
sns.set_theme(style='whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)

#### Load the Sales Dataset

Replace `'sales.csv'` below with the correct path to your sales CSV dataset file.

# Generate a synthetic sales dataset and save it as 'sales.csv'
import numpy as np
import pandas as pd

# Seed for reproducibility
np.random.seed(42)

n_samples = 200
tv_spend = np.random.uniform(10, 300, n_samples)
radio_spend = np.random.uniform(5, 50, n_samples)
newspaper_spend = np.random.uniform(2, 80, n_samples)

# Sales formula with some random noise
sales = 5 + 0.05 * tv_spend + 0.18 * radio_spend + 0.005 * newspaper_spend + np.random.normal(0, 1.5, n_samples)

synthetic_data = pd.DataFrame({
    'TV': tv_spend,
    'Radio': radio_spend,
    'Newspaper': newspaper_spend,
    'Sales': sales
})

# Save to CSV
synthetic_data.to_csv('sales.csv', index=False)
print("Synthetic 'sales.csv' successfully created!")

# Load and display the dataset
df = pd.read_csv('sales.csv')
display(df.head())
display(df.info())

#### Step 1: Exploratory Data Analysis (EDA)

Let's visualize the correlation and scatter plots between each advertising medium (TV, Radio, Newspaper) and Sales.

# Pairplot to visualize relationships
sns.pairplot(df, x_vars=['TV', 'Radio', 'Newspaper'], y_vars='Sales', height=5, aspect=0.8, kind='reg')
plt.suptitle('Advertising Spend vs Sales', y=1.05)
plt.show()

# Correlation Matrix
splt = sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()

#### Step 2: Model Training and Evaluation

We will split the dataset into training (80%) and testing (20%) sets, train a Multiple Linear Regression model, and evaluate its performance.

# Define features and target variable
X = df[['TV', 'Radio', 'Newspaper']]
y = df['Sales']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Calculate metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"R-squared (R2) Score: {r2:.4f}")

#### Step 3: Actionable Insights and Impact Analysis

Let's extract the model coefficients to determine how changes in advertising budgets impact overall sales outcomes.

# Analyze Coefficients
coefficients = pd.DataFrame(model.coef_, X.columns, columns=['Coefficient'])
coefficients.loc['Intercept'] = model.intercept_
print("Model Coefficients and Intercept:")
display(coefficients)

# Plot feature importance
plt.figure(figsize=(8, 5))
sns.barplot(x=model.coef_, y=X.columns, palette='viridis')
plt.title('Impact of Advertising Mediums on Sales')
plt.xlabel('Coefficient Value (Increase in Sales per unit spend)')
plt.show()
