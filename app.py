# Importing necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

@st.cache_data
def load_data():
    df = pd.read_csv('CO2 Emissions.csv', encoding='latin-1')
    return df

df = load_data()

st.title("CO2 Emissions Dashboard (HW 3/4)")
st.subheader("This is a basic analysis of global CO2 emissions over time.")
st.write("By Jesus Barrios, Discovery Analyst")

st.header("Dataset Overview")
st.write("This is an glimpse into what the dataset looks like.")
st.dataframe(df.head(10))

st.header("Filter the Data (Country and Year Range)")

countries = sorted(df['Country'].unique())

selected_country = st.selectbox("Select a country:", countries)

country_data = df[df['Country'] == selected_country].copy()

min_year = int(df['Year'].min())
max_year = int(df['Year'].max())
year_range = st.slider("Select year range:", min_year, max_year, (1900, 2020))

filtered_data = country_data[(country_data['Year'] >= year_range[0]) & (country_data['Year'] <= year_range[1])]

st.write(f"Showing data for **{selected_country}** from {year_range[0]} to {year_range[1]}")
st.write(f"Total data points: {len(filtered_data)}")

st.header(f"Linear Regression: CO2 Emissions Over Time ({selected_country})")

if len(filtered_data) > 1:
    X = filtered_data['Year'].values.reshape(-1, 1)
    y = filtered_data['CO2 emission (Tons)'].values
    
    model = LinearRegression()
    model.fit(X, y)
    
    y_pred = model.predict(X)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(X, y, alpha=0.5, label='Actual Data', color='blue')
    ax.plot(X, y_pred, color='red', linewidth=2, label='Linear Regression')
    ax.set_xlabel('Year')
    ax.set_ylabel('CO2 Emissions (Tons)')
    ax.set_title(f'CO2 Emissions Over Time: {selected_country}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    st.pyplot(fig)
    
    st.write(f"**Slope:** {model.coef_[0]:.2f} tons per year")
    st.write(f"**Intercept:** {model.intercept_:.2f} tons")
    
    from sklearn.metrics import r2_score
    r2 = r2_score(y, y_pred)
    st.write(f"**R² Score:** {r2:.4f}")
else:
    st.warning("Not enough data points for linear regression. Please adjust your filters.")

st.header(f"CO2 Emissions Trend ({selected_country})")

chart_type = st.radio("Select chart type:", ["Line Chart", "Bar Chart", "Area Chart"])

if len(filtered_data) > 0:
    chart_data = filtered_data[['Year', 'CO2 emission (Tons)']].set_index('Year')
    
    if chart_type == "Line Chart":
        st.line_chart(chart_data)
    elif chart_type == "Bar Chart":
        st.bar_chart(chart_data)
    elif chart_type == "Area Chart":
        st.area_chart(chart_data)
else:
    st.warning("No data available for the selected filters.")

st.header("Compare CO2 Emissions of Multiple Countries")

selected_countries = st.multiselect("Select countries to compare:", countries, default=[selected_country])

if len(selected_countries) > 0:
    comparison_data = df[df['Country'].isin(selected_countries)]
    comparison_data = comparison_data[(comparison_data['Year'] >= year_range[0]) & (comparison_data['Year'] <= year_range[1])]
    
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    
    for country in selected_countries:
        country_subset = comparison_data[comparison_data['Country'] == country]
        ax2.plot(country_subset['Year'], country_subset['CO2 emission (Tons)'], label=country, marker='o', markersize=3)
    
    ax2.set_xlabel('Year')
    ax2.set_ylabel('CO2 Emissions (Tons)')
    ax2.set_title('CO2 Emissions Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    st.pyplot(fig2)

st.write("---")

st.header(f"Statistics Summary ({selected_country})")

if len(filtered_data) > 0:
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Average CO2 Emissions", f"{filtered_data['CO2 emission (Tons)'].mean():.2f} tons")
    
    with col2:
        st.metric("Maximum CO2 Emissions", f"{filtered_data['CO2 emission (Tons)'].max():.2f} tons")
    
    with col3:
        st.metric("Minimum CO2 Emissions", f"{filtered_data['CO2 emission (Tons)'].min():.2f} tons")

st.write("---")
st.markdown(
    'Data source: [**CO2 Emissions dataset**]'
    '(https://drive.google.com/drive/folders/1pKat7CVPFHTe76WJVam8wn5VVn84yxKP)'
)