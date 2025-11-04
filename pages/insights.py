import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

st.set_page_config(page_title="Data Insights", page_icon="📊")
st.title("📊 Data Insights & Visualizations")
st.write("This page shows some important insights from the real-world credit card dataset used for this project.")

# --- Load Data ---
# Note: Loading large CSVs from GitHub is slow. 
# It's better to upload a small sample of the data (e.g., 'sample.csv')
# For this demo, we'll create a mock dataframe if the file isn't found.

@st.cache_data
def load_data():
    # --- IMPORTANT ---
    # You should create a small sample of 'creditcard.csv' (e.g., first 50k rows)
    # and upload it to your GitHub as 'sample.csv'. Then change the line below.
    data_url = 'creditcard.csv' # Or 'sample.csv'

    try:
        df = pd.read_csv(data_url)
        return df
    except FileNotFoundError:
        st.warning("Could not find the dataset. Displaying mock data instead.")
        # Create mock data if file is not found
        data = {'Class': [0]*284315 + [1]*492, 
                'Amount': pd.Series(range(0, 284807)).sample(284807).values,
                'Time': pd.Series(range(0, 172792)).sample(284807).values
               }
        df = pd.DataFrame(data)
        return df

df = load_data()


# --- 1. Data Imbalance ---
st.header("1. Data Imbalance (The Main Challenge)")
st.write("The primary challenge of this project is the severe imbalance between normal and fraudulent transactions.")

class_counts = df['Class'].value_counts()
st.dataframe(class_counts)

# Bar plot
try:
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.countplot(x='Class', data=df, ax=ax, palette=['#34A853', '#EA4335'])
    ax.set_title('Normal (0) vs. Fraud (1)')
    ax.set_xticklabels(['Normal (0)', 'Fraud (1)'])
    st.pyplot(fig)
    
    fraud_percentage = (class_counts.get(1, 0) / class_counts.sum()) * 100
    st.warning(f"Only **{fraud_percentage:.3f}%** of all transactions are fraudulent.")

except Exception as e:
    st.error(f"Could not draw plot: {e}")


# --- 2. Transaction Amount Distribution ---
st.header("2. Transaction Amount Distribution")
st.write("How does the transaction amount differ between normal and fraudulent cases?")

try:
    fig_amt, (ax1, ax2) = plt.subplots(ncols=2, figsize=(18, 6))

    # Normal transactions
    sns.histplot(df[df['Class'] == 0]['Amount'], bins=100, ax=ax1, color='green', kde=True)
    ax1.set_title('Normal Transaction Amount')
    ax1.set_xlabel('Amount')
    ax1.set_ylabel('Frequency')
    ax1.set_xlim(0, 5000) # Limiting x-axis to see details

    # Fraudulent transactions
    sns.histplot(df[df['Class'] == 1]['Amount'], bins=50, ax=ax2, color='red', kde=True)
    ax2.set_title('Fraudulent Transaction Amount')
    ax2.set_xlabel('Amount')
    ax2.set_ylabel('Frequency')

    st.pyplot(fig_amt)
    st.info("Note: Most transactions (both normal and fraudulent) are for small amounts.")

except Exception as e:
    st.error(f"Could not draw amount plot: {e}")
