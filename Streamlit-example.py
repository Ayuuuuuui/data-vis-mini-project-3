import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from random import randint

# Sample data representing tokens and their NER predictions
data = [
    ("123", "ADDR", 0.95),
    ("Sukhumvit", "ADDR", 0.98),
    ("Road", "ADDR", 0.96),
    ("Bangkok", "LOC", 0.99),
    ("10110", "POST", 0.93),
]

# Convert the data into a pandas DataFrame
df = pd.DataFrame(data, columns=["Token", "Entity", "Confidence"])

# Entity color mapping for visualization
entity_colors = {
    "LOC": "lightblue",
    "POST": "lightgreen",
    "ADDR": "orange",
    "O": "gray",
}

# Function to highlight tokens based on entity type
def highlight_tokens(row):
    return f'<span style="color:{entity_colors.get(row["Entity"], "black")}; font-weight: bold">{row["Token"]}</span>'

# Render the tokens with their respective colors
def render_tokens(df):
    html_tokens = df.apply(highlight_tokens, axis=1).str.cat(sep=" ")
    st.markdown(html_tokens, unsafe_allow_html=True)

# Function to render entity breakdown pie chart
def render_entity_breakdown(df):
    entity_counts = df["Entity"].value_counts()
    fig, ax = plt.subplots()
    ax.pie(entity_counts, labels=entity_counts.index, autopct='%1.1f%%', startangle=90, colors=[entity_colors[e] for e in entity_counts.index])
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    st.pyplot(fig)

# Render confidence scores as a bar chart
def render_confidence_scores(df):
    fig, ax = plt.subplots()
    ax.bar(df["Token"], df["Confidence"], color='skyblue')
    ax.set_ylabel('Confidence')
    ax.set_title('Token Confidence Scores')
    st.pyplot(fig)

# Streamlit application layout
st.title("Named Entity Recognition (NER) Visualization")

# Display the tokenized text with highlighted entities
st.subheader("Tokenized Text with Highlighted Entities")
render_tokens(df)

# Display entity breakdown pie chart
st.subheader("Entity Breakdown")
render_entity_breakdown(df)

# Display confidence scores for each token
st.subheader("Confidence Scores per Token")
render_confidence_scores(df)

# Optional: Add a slider to filter by confidence level (interactive feature)
min_conf = st.slider("Filter by Minimum Confidence", 0.0, 1.0, 0.8)
filtered_df = df[df["Confidence"] >= min_conf]

# Display filtered tokens if confidence slider is used
st.subheader(f"Filtered Tokens with Confidence >= {min_conf}")
render_tokens(filtered_df)