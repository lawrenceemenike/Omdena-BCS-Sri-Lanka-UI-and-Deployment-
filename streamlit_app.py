import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from datetime import datetime

# Generate dummy data
def load_data():
    np.random.seed(0)
    dates = pd.date_range('2021-01-01', periods=100, freq='D')
    data = pd.DataFrame({
        'Date': np.tile(dates, 5),
        'Keyword': np.random.choice(['Education', 'Employment', 'Healthcare'], size=500),
        'Gender': np.random.choice(['Male', 'Female', 'Non-binary'], size=500),
        'SocialGroup': np.random.choice(['Group A', 'Group B', 'Group C'], size=500),
        'Sector': np.random.choice(['Health', 'Education', 'Technology'], size=500),
        'District': np.random.choice(['District A', 'District B', 'Province C'], size=500),
        'Role': np.random.choice(['Leadership', 'Technical', 'Administrative'], size=500),
        'Enrollment': np.random.choice(['Enrolled', 'Not Enrolled'], size=500),
        'Mentions': np.random.randint(10, 100, size=500)
    })
    return data

# Visualization functions (dummy implementations)
def visualize_data(chart_type, data, title):
    if chart_type == "time_series":
        chart = alt.Chart(data).mark_line().encode(x='Date', y='Mentions', color='Keyword').properties(title=title)
    elif chart_type == "bar":
        chart = alt.Chart(data).mark_bar().encode(x='Gender', y='Mentions', color='Gender').properties(title=title)
    else:
        chart = alt.Chart(data).mark_circle().encode(x='Sector', y='Mentions', color='Sector').properties(title=title)
    st.altair_chart(chart, use_container_width=True)

# Streamlit app layout
st.title('GESI Analysis Dashboard')

# Sidebar filters
data = load_data()
st.sidebar.header('Filters')
start_date, end_date = st.sidebar.date_input('Select Date Range', [datetime(2021, 1, 1), datetime(2021, 4, 10)])
keyword = st.sidebar.selectbox('Select Keyword', ['All'] + ['Education', 'Employment', 'Healthcare'])
gender = st.sidebar.multiselect('Select Gender', ['Male', 'Female', 'Non-binary'])
social_group = st.sidebar.multiselect('Select Social Group', ['Group A', 'Group B', 'Group C'])
sector = st.sidebar.multiselect('Select Sector', ['Health', 'Education', 'Technology'])
district_province = st.sidebar.multiselect('Select District/Province', ['District A', 'District B', 'Province C'])
role_position = st.sidebar.multiselect('Select Role/Position', ['Leadership', 'Technical', 'Administrative'])
education_enrollment = st.sidebar.radio('Select Education Enrollment Status', ['Enrolled', 'Not Enrolled'])

# Main content - Visualizations in a 2 x 2 grid
col1, col2 = st.columns(2)
with col1:
    st.header('Time Series Analysis on GESI Keyword Usage')
    visualize_data("time_series", data, 'Keyword Mentions Over Time')

with col2:
    st.header('Sentiment Analysis Results')
    visualize_data("bar", data, 'Sentiment Analysis by Gender')

col3, col4 = st.columns(2)
with col3:
    st.header('Sector-wise Analysis')
    visualize_data("bar", data, 'Sector Analysis')

with col4:
    st.header('District/Province Level Analysis')
    visualize_data("bar", data, 'District Analysis')
