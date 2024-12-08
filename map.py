import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
import plotly.graph_objects as go

# Step 1: Load Dataset
file_path = r"C:\Users\iurch\OneDrive\DataCircle\project_data\twitterelection_dataanalysis\twitterelection_dataanalysis\df_for_dash.csv"
df = pd.read_csv(file_path)
df['created_at'] = pd.to_datetime(df['created_at'])

# Step 2: Calculate Preferred Candidate and Normalized Sentiment
state_candidate_sentiment = (
    df.groupby(['state_code', 'candidate'])
    .agg(total_weighted_sentiment=('weighted_sentiment', 'sum'))
    .reset_index()
)

state_comparison = state_candidate_sentiment.pivot(
    index='state_code',
    columns='candidate',
    values='total_weighted_sentiment'
).fillna(0)

state_comparison['preferred_candidate'] = np.where(                #Hier I create PREFERABLE CANDIDATE column up to who have higher weighted_sentiment in this state
    state_comparison['Joe Biden'] > state_comparison['Donald Trump'],
    'Joe Biden',
    'Donald Trump'
)

df = df.merge(state_comparison[['preferred_candidate']], on='state_code', how='left')

df['engagement'] = df['likes'] + df['retweet_count']
df['normalized_sentiment'] = np.where(
    df['engagement'] > 0,
    df['weighted_sentiment'] / df['engagement'],
    0
)
############################################################
#Page config
st.set_page_config(layout="wide")

##############################################################
# Summary Information
total_likes = df['likes'].sum()  # Total likes
total_tweets = len(df)  # Total number of tweets
total_retweets = df['retweet_count'].sum()  # Total retweets
num_rows, num_columns = df.shape  # Dataset size

# Find the main topic (most frequent value in representative_word column)
if 'representative_word' in df.columns:
    main_topic = df['representative_word'].value_counts().idxmax()
    main_topic_count = df['representative_word'].value_counts().max()
else:
    main_topic = "N/A"
    main_topic_count = 0

# Display the summary
st.header("Dataset Overview")

# Create columns for metrics
col1, col2, col3 = st.columns(3)
col1.metric(label="Total Tweets", value=f"{total_tweets:,}")
col2.metric(label="Total Retweets", value=f"{total_retweets:,}")
col3.metric(label="Country", value="USA")

# Additional summary metrics
col4, col5, col6 = st.columns(3)
col4.metric(label="Total Likes", value=f"{total_likes:,}")
col5.metric(label="Rows", value=f"{num_rows:,}")
col6.metric(label="Main Topic", value=f"{main_topic}")
#########################################################
# Sidebar Filters
st.sidebar.header("Filters")

# Candidate Filter (Allowing multiple candidates to be selected)
selected_candidates = st.sidebar.multiselect(
    "Select Candidates", 
    options=df['candidate'].dropna().unique(), 
    default=df['candidate'].dropna().unique()  # Default to show all candidates
)

# Sentiment Filter (Allowing selection of sentiment type)
sentiment_types = df['sentiment_vader'].dropna().unique()
selected_sentiment = st.sidebar.selectbox("Select Sentiment", sentiment_types, index=0)

# State Code Filter (Allowing multiple states to be selected)
state_codes = df['state_code'].dropna().unique()
selected_state_codes = st.sidebar.multiselect(
    "Select State Code", 
    options=state_codes, 
    default=state_codes  # Default to show all states
)

# Date Range Filter
min_date = df['created_at'].min().date()
max_date = df['created_at'].max().date()
selected_date_range = st.sidebar.slider(
    "Select Date Range",
    min_value=min_date,
    max_value=max_date,
    value=(min_date, max_date)
)

# Filter the dataset based on user input
filtered_data = df[
    (df['candidate'].isin(selected_candidates)) &  # Apply selected candidates filter
    (df['sentiment_vader'] == selected_sentiment) &  # Apply selected sentiment filter
    (df['state_code'].isin(selected_state_codes)) &  # Apply selected state filter
    (df['created_at'].dt.date >= selected_date_range[0]) &  # Apply selected date range filter
    (df['created_at'].dt.date <= selected_date_range[1])  # Apply selected date range filter
]

# Handle empty filtered data case
if filtered_data.empty:
    st.write("No data available for the selected filters.")
    st.stop()

filtered_data = filtered_data.sort_values(by='created_at', ascending=True)


#######################################################################

#USA Map with boarders

# Aggregate data at the state level
# state_summary = df.groupby('state_code').agg(
#     total_likes=('likes', 'sum'),
#     total_retweets=('retweet_count', 'sum'),
#     normalized_sentiment=('normalized_sentiment', 'mean'),
#     preferred_candidate=('preferred_candidate', lambda x: x.mode()[0])  # Most frequent candidate
# ).reset_index()

# # Step 4. Choropleth Map with Plotly
# fig = px.scatter_geo(
#     state_summary,
#     locations='state_code',  # Two-letter state codes
#     locationmode="USA-states",  # Built-in USA state mode
#     color='preferred_candidate',  # Color by candidate
#     size='normalized_sentiment',  # Bubble size by total likes
#     color_discrete_map={'Donald Trump': 'red', 'Joe Biden': 'blue'},  # Custom colors
#     scope='usa',  # Focus on USA
#     hover_name='state_code',  # Display state codes on hover
#     hover_data={
#         'normalized_sentiment': ':.2f',  # Sentiment data
#         'total_likes': True,  # Total likes
#         'preferred_candidate': True  # Preferred candidate
#     }
# )

# fig.update_layout(
#     title_text="USA Election 2020 Map",  # Title text
#     title_x=0.5,  # Center the title horizontally
#     title_y=0.95,  # Adjust the vertical placement of the title (optional)
#     title_font=dict(size=40),  # Set the font size
#     title_font_color="black",  # Optional: Set title color
#     margin=dict(t=60)  # Adjust top margin to avoid clipping the title
# )

# st.plotly_chart(fig, use_container_width=True)

##################################################################################
# Generate state_summary with animation frame

state_summary = filtered_data.groupby(['state_code', filtered_data['created_at'].dt.date]).agg(
    total_likes=('likes', 'sum'),
    total_retweets=('retweet_count', 'sum'),
    normalized_sentiment=('normalized_sentiment', 'mean'),
    preferred_candidate=('preferred_candidate', lambda x: x.mode()[0])
).reset_index()

state_summary['created_at'] = state_summary['created_at'].astype(str)  # Convert date to string

# Create the animated map
fig = px.scatter_geo(
    state_summary,
    locations='state_code',  # Two-letter state codes
    locationmode="USA-states",  # Built-in USA state mode
    color='preferred_candidate',  # Color by candidate
    size='normalized_sentiment',  # Bubble size by normalized sentiment
    color_discrete_map={'Donald Trump': 'red', 'Joe Biden': 'blue'},  # Custom colors
    scope='usa',  # Focus on USA
    hover_name='state_code',  # Display state codes on hover
    hover_data={
        'normalized_sentiment': ':.2f',
        'total_likes': True,
        'preferred_candidate': True
    },
    animation_frame='created_at',  # Add animation frame
    title="USA Election 2020 Map"
)

# Improve map appearance with borders
fig.update_geos(
    visible=True,
    showland=True,
    landcolor="lightgray",
    showlakes=True,
    lakecolor="blue",
    showcountries=True,
    countrycolor="black",
    showsubunits=True,
    subunitcolor="darkgray"
)

fig.update_layout(
    title_text="USA Election 2020 Map",
    title_x=0.5,
    title_y=0.95,
    title_font=dict(size=40),
    margin=dict(t=60)
)

# Display in Streamlit
st.plotly_chart(fig, use_container_width=True)
#########################################################

# Calculate Timeline Data
timeline_data = filtered_data.groupby(filtered_data['created_at'].dt.date).agg({
    'tweet_id': 'count',  # Number of tweets
    'likes': 'sum',       # Total likes
    'sentiment_score': 'mean'  # Average sentiment score
}).reset_index()

timeline_data.rename(columns={'created_at': 'Date'}, inplace=True)

# Timeline Visualization
st.subheader("Tweet Timeline")

fig_timeline = go.Figure()
fig_timeline.add_trace(go.Scatter(
    x=timeline_data['Date'], y=timeline_data['tweet_id'],
    mode='lines+markers', name='Number of Tweets',
    line=dict(color='blue')
))
fig_timeline.add_trace(go.Scatter(
    x=timeline_data['Date'], y=timeline_data['likes'],
    mode='lines+markers', name='Total Likes',
    line=dict(color='green')
))
fig_timeline.update_layout(
    xaxis_title="Date",
    yaxis_title="Count / Likes",
    legend_title="Metrics"
)

st.plotly_chart(fig_timeline, use_container_width=True)

########################################################

# Group data and include lat/long
state_summary = filtered_data.groupby(['state_code', filtered_data['created_at'].dt.date]).agg(
    total_likes=('likes', 'sum'),
    total_retweets=('retweet_count', 'sum'),
    normalized_sentiment=('normalized_sentiment', 'mean'),
    preferred_candidate=('preferred_candidate', lambda x: x.mode()[0])
).reset_index()

state_summary['created_at'] = state_summary['created_at'].astype(str)  # Convert date to string

# Add lat/long if not already included
if 'lat' not in state_summary.columns or 'long' not in state_summary.columns:
    state_summary = state_summary.merge(
        filtered_data[['state_code', 'lat', 'long']].drop_duplicates(),
        on='state_code',
        how='left'
    )

# Create the animated map
fig = px.scatter_geo(
    state_summary,
    locations='state_code',  # Two-letter state codes
    locationmode="USA-states",  # Built-in USA state mode
    color='preferred_candidate',  # Color by candidate
    size='normalized_sentiment',  # Bubble size by normalized sentiment
    color_discrete_map={'Donald Trump': 'red', 'Joe Biden': 'blue'},  # Custom colors
    scope='usa',  # Focus on USA
    animation_frame='created_at',  # Add animation frame
    hover_name='state_code',  # Display state codes on hover
    hover_data={
        'normalized_sentiment': ':.2f',
        'total_likes': True,
        'preferred_candidate': True
    },
    title="USA Election 2020 Map"
)

# Add visible state abbreviations (annotations)
fig.add_traces([
    dict(
        type="scattergeo",
        lon=state_summary['long'],
        lat=state_summary['lat'],
        text=state_summary['state_code'],  # State abbreviations
        mode="text",
        textfont=dict(
            family="Arial",
            size=10,
            color="black"
        ),
        showlegend=False
    )
])

# Update layout for better alignment and appearance
fig.update_geos(
    showframe=False,
    showcoastlines=True,
    coastlinecolor="lightgray",
    projection_type="albers usa"  # Focuses on USA
)

fig.update_layout(
    title_text="USA Election 2020 Map",
    title_x=0.5,
    title_y=0.95,
    title_font=dict(size=40),
    margin=dict(t=60)
)

# Display in Streamlit
st.plotly_chart(fig, use_container_width=True)
