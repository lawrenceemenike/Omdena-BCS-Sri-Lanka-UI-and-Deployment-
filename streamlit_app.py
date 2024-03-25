import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
import seaborn as sns
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go

# Set page configuration
st.set_page_config(layout="wide")

# Function to load and clean data
def load_and_clean_data():
    df1 = pd.read_csv("data/reviewed_social_media_english.csv")
    df2 = pd.read_csv("data/reviewed_news_english.csv")
    df3 = pd.read_csv("data/tamil_social_media.csv")  
    df4 = pd.read_csv("data/tamil_news.csv")       

    # Concatenate dataframes and clean data
    df_combined = pd.concat([df1, df2, df3, df4])
    
    # Replace 'nan' and 'None' with numpy NaN for removal
    df_combined['Domain'] = df_combined['Domain'].replace({"MUSLIM": "Muslim", "nan": pd.NA, "None": pd.NA, "Other-Ethnic": "Other-Ethnicity"})
    
    # Specific replacements for 'Sentiment' and 'Discrimination'
    df_combined['Sentiment'] = df_combined['Sentiment'].replace({"nan": pd.NA, "None": pd.NA, "No": pd.NA})
    df_combined['Discrimination'] = df_combined['Discrimination'].replace({"nan": pd.NA, "None": pd.NA, "No": pd.NA})
    
    # Drop rows with NA values in 'Domain', 'Sentiment', and 'Discrimination'
    df_combined.dropna(subset=['Domain', 'Sentiment', 'Discrimination'], inplace=True)

    return df_combined

df = load_and_clean_data()


# Page navigation setup
page_names = ["Overview", "Sentiment Analysis", "Discrimination Analysis", "Channel Analysis"]
page = st.sidebar.selectbox("Choose a page", page_names)

# Sidebar Filters
domain_options = df['Domain'].dropna().unique()
channel_options = df['Channel'].dropna().unique()
sentiment_options = df['Sentiment'].dropna().unique()
discrimination_options = df['Discrimination'].dropna().unique()

domain_filter = st.sidebar.multiselect('Select Domain', options=domain_options, default=domain_options)
channel_filter = st.sidebar.multiselect('Select Channel', options=channel_options, default=channel_options)
sentiment_filter = st.sidebar.multiselect('Select Sentiment', options=sentiment_options, default=sentiment_options)
discrimination_filter = st.sidebar.multiselect('Select Discrimination', options=discrimination_options, default=discrimination_options)

# Apply filters
df_filtered = df[(df['Domain'].isin(domain_filter)) & 
                 (df['Channel'].isin(channel_filter)) & 
                 (df['Sentiment'].isin(sentiment_filter)) & 
                 (df['Discrimination'].isin(discrimination_filter))]

# Define a color palette for consistent visualization styles
color_palette = px.colors.sequential.Viridis


# Visualisation for Domain Distribution
def create_pie_chart(df, column, title):
    fig = px.pie(df, names=column, title=title, hole=0.35)
    fig.update_layout(margin=dict(l=20, r=20, t=30, b=20), legend=dict(x=0.1, y=1), font=dict(size=12))
    fig.update_traces(marker=dict(colors=color_palette))
    return fig

# Visualization for Distribution of Gender versus Ethnicity
def create_gender_ethnicity_distribution_chart(df):
    df['GenderOrEthnicity'] = df['Domain'].apply(lambda x: "Gender: Women & LGBTQIA+" if x in ["Women", "LGBTQIA+"] else "Ethnicity")
    fig = px.pie(df, names='GenderOrEthnicity', title='Distribution of Gender versus Ethnicity', hole=0.35)
    fig.update_layout(margin=dict(l=20, r=20, t=30, b=20), legend=dict(x=0.1, y=1), font=dict(size=12))
    return fig

# Visualization for Sentiment Distribution Across Domains
def create_sentiment_distribution_chart(df):
    df['Discrimination'] = df['Discrimination'].replace({"Non Discriminative": "Non-Discriminative"})  # Assuming typo in the original script
    domain_counts = df.groupby(['Domain', 'Sentiment']).size().reset_index(name='counts')
    fig = px.bar(domain_counts, x='Domain', y='counts', color='Sentiment', title="Sentiment Distribution Across Domains", barmode='stack')
    fig.update_layout(margin=dict(l=20, r=20, t=40, b=20), xaxis_title="Domain", yaxis_title="Counts", font=dict(size=12))
    return fig

# Visualization for Correlation between Sentiment and Discrimination
def create_sentiment_discrimination_grouped_chart(df):
    # Creating a crosstab of 'Sentiment' and 'Discrimination'
    crosstab_df = pd.crosstab(df['Sentiment'], df['Discrimination'])
    
    # Check if 'Yes' and 'No' are in the columns after the crosstab operation
    value_vars = crosstab_df.columns.intersection(['Yes', 'No']).tolist()
    
    # If 'No' is not in columns, it will not be included in melting
    melted_df = pd.melt(crosstab_df.reset_index(), id_vars='Sentiment', value_vars=value_vars, var_name='Discrimination', value_name='Count')
    
    # Proceeding to plot only if we have data to plot
    if not melted_df.empty:
        fig = px.bar(melted_df, x='Sentiment', y='Count', color='Discrimination', barmode='group', title="Sentiment vs. Discrimination")
        fig.update_layout(margin=dict(l=20, r=20, t=40, b=20), xaxis_title="Sentiment", yaxis_title="Count", font=dict(size=12))
        return fig
    else:
        return "No data to display for the selected filters."

# Function for Top Domains with Negative Sentiment Chart
def create_top_negative_sentiment_domains_chart(df):
    domain_counts = df.groupby(['Domain', 'Sentiment']).size().unstack(fill_value=0)
    domain_counts.sort_values(by='Negative', ascending=False, inplace=True)
    domain_counts_subset = domain_counts.iloc[:3, [0]]
    domain_counts_subset = domain_counts_subset.rename(columns={domain_counts_subset.columns[0]: 'Count'})
    domain_counts_subset = domain_counts_subset.reset_index()
    colors = ['limegreen', 'crimson', 'darkcyan']
    fig = px.bar(domain_counts_subset, x='Count', y='Domain', title='Top Domains with Negative Sentiment', color='Domain',
                 orientation='h', color_discrete_sequence=colors)
    fig.update_layout(margin=dict(l=20, r=20, t=40, b=20), xaxis_title="Negative sentiment content Count", yaxis_title="Domain")
    return fig

# Function for Key Phrases in Negative Sentiment Content Chart
def create_key_phrases_negative_sentiment_chart(df):
    cv = CountVectorizer(ngram_range=(3,3), stop_words='english')
    trigrams = cv.fit_transform(df['Content'][df['Sentiment'] == 'Negative'])
    count_values = trigrams.toarray().sum(axis=0)
    ngram_freq = pd.DataFrame(sorted([(count_values[i], k) for k, i in cv.vocabulary_.items()], reverse=True))
    ngram_freq.columns = ['frequency', 'ngram']
    fig = px.bar(ngram_freq.head(10), x='frequency', y='ngram', orientation='h', title='Key phrases in Negative Sentiment Content')
    fig.update_layout(margin=dict(l=20, r=20, t=40, b=20), xaxis_title="Frequency", yaxis_title="Trigram")
    return fig

# Function for Prevalence of Discriminatory Content Chart
def create_prevalence_discriminatory_content_chart(df):
    domain_counts = df.groupby(['Domain', 'Discrimination']).size().unstack(fill_value=0)
    fig = px.bar(domain_counts, x=domain_counts.index, y=['Discriminative', 'Non-Discriminative'], barmode='group',
                 title='Prevalence of Discriminatory Content')
    fig.update_layout(margin=dict(l=20, r=20, t=40, b=20), xaxis_title="Domain", yaxis_title="Count")
    return fig

# Function for Top Domains with Discriminatory Content Chart
def create_top_discriminatory_domains_chart(df):
    domain_counts = df.groupby(['Domain', 'Discrimination']).size().unstack(fill_value=0)
    domain_counts.sort_values(by='Discriminative', ascending=False, inplace=True)
    domain_counts_subset = domain_counts.iloc[:3]
    domain_counts_subset = domain_counts_subset.rename(columns={'Discriminative': 'Count'})
    fig = px.bar(domain_counts_subset, x='Count', y=domain_counts_subset.index, orientation='h',
                 title='Top Domains with Discriminatory Content')
    fig.update_layout(margin=dict(l=20, r=20, t=40, b=20), xaxis_title="Discriminatory Content Count", yaxis_title="Domain")
    return fig

# Function for Channel-wise Sentiment Over Time Chart
def create_sentiment_distribution_by_channel_chart(df):
    sentiment_by_channel = df.groupby(['Channel', 'Sentiment']).size().reset_index(name='counts')
    fig = px.bar(sentiment_by_channel, x='Channel', y='counts', color='Sentiment', title="Sentiment Distribution by Channel", barmode='group')
    fig.update_layout(margin=dict(l=20, r=20, t=40, b=20), xaxis_title="Channel", yaxis_title="Counts", font=dict(size=12))
    return fig

# Function for Channel-wise Distribution of Discriminative Content Chart
def create_channel_discrimination_chart(df):
    channel_discrimination = df.groupby(['Channel', 'Discrimination']).size().unstack(fill_value=0)
    fig = px.bar(channel_discrimination, x=channel_discrimination.index, y=['Discriminative', 'Non-Discriminative'], barmode='group')
    fig.update_layout(title='Channel-wise Distribution of Discriminative Content', margin=dict(l=20, r=20, t=40, b=20))
    return fig

# Function for rendering dashboard
def render_dashboard(page, df_filtered):
    if page == "Overview":
        st.title("Overview Dashboard")
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(create_pie_chart(df_filtered, 'Domain', 'Distribution of Domains'))
        with col2:
            st.plotly_chart(create_gender_ethnicity_distribution_chart(df_filtered))

        col3, col4 = st.columns(2)
        with col3:
            st.plotly_chart(create_sentiment_distribution_chart(df_filtered))
        with col4:
            chart = create_sentiment_discrimination_grouped_chart(df_filtered)
            if isinstance(chart, str):
                st.write(chart)
            else:
                st.plotly_chart(chart)

    elif page == "Sentiment Analysis":
        st.title("Sentiment Analysis Dashboard")
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(create_sentiment_distribution_chart(df_filtered))
        with col2:
            st.plotly_chart(create_top_negative_sentiment_domains_chart(df_filtered))

        col3, col4 = st.columns(2)
        with col3:
            st.plotly_chart(create_key_phrases_negative_sentiment_chart(df_filtered))

    elif page == "Discrimination Analysis":
        st.title("Discrimination Analysis Dashboard")
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(create_prevalence_discriminatory_content_chart(df_filtered))
        with col2:
            st.plotly_chart(create_top_discriminatory_domains_chart(df_filtered))

    elif page == "Channel Analysis":
        st.title("Channel Analysis Dashboard")
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(create_sentiment_distribution_by_channel_chart(df_filtered))
        with col2:
            st.plotly_chart(create_channel_discrimination_chart(df_filtered))


# Render the selected dashboard page
render_dashboard(page, df_filtered)
