import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

st.set_page_config(page_title="Data Insights", page_icon="📊")
st.title("📊 Data Insights & Visualizations")
st.write("This page shows insights from a sample of the credit card dataset.")

# --- Load Data ---
@st.cache_data
def load_data():
    data_url = 'card_sample_small.csv' 

    try:
        df = pd.read_csv(data_url)
        return df
    except FileNotFoundError:
        st.error(f"Error: '{data_url}' not found in your GitHub repository.")
        st.warning(f"Please ensure you have uploaded '{data_url}' to GitHub.")
        return None

df = load_data()

# --- 1. Data Imbalance ---
if df is not None:
    st.header("1. Data Imbalance (The Main Challenge)")
    st.write("The primary challenge is the severe imbalance between normal and fraudulent transactions.")

    class_counts = df['Class'].value_counts()
    st.dataframe(class_counts)

    # Bar plot
    try:
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.countplot(x='Class', data=df, ax=ax, palette=['#34A853', '#EA4335'])
        ax.set_title(f'Normal (0) vs. Fraud (1) (in {len(df)} sample)')
        ax.set_xticklabels(['Normal (0)', 'Fraud (1)'])
        st.pyplot(fig)
        
        fraud_percentage = (class_counts.get(1, 0) / class_counts.sum()) * 100
        st.warning(f"Only **{fraud_percentage:.3f}%** of transactions in this sample are fraudulent.")

    except Exception as e:
        st.error(f"Could not draw plot: {e}")


    # --- 2. Transaction Amount Distribution ---
    st.header("2. Transaction Amount Distribution")
    st.write("How does the *scaled* transaction amount differ between normal and fraudulent cases?")

    try:
        # --- এই সেকশনটি ঠিক করা হয়েছে ---
        # আমরা 'Amount' এর বদলে 'scaled_amount' ব্যবহার করবো
        
        fig_amt, (ax1, ax2) = plt.subplots(ncols=2, figsize=(18, 6))

        # Normal transactions
        sns.histplot(df[df['Class'] == 0]['scaled_amount'], bins=100, ax=ax1, color='green', kde=True)
        ax1.set_title('Normal Transaction (Scaled Amount)')
        ax1.set_xlabel('Scaled Amount')
        ax1.set_ylabel('Frequency')

        # Fraudulent transactions
        if 1 in df['Class'].values:
            sns.histplot(df[df['Class'] == 1]['scaled_amount'], bins=50, ax=ax2, color='red', kde=True)
            ax2.set_title('Fraudulent Transaction (Scaled Amount)')
        else:
            ax2.set_title('No Fraudulent Transactions in this sample')
            
        ax2.set_xlabel('Scaled Amount')
        ax2.set_ylabel('Frequency')

        st.pyplot(fig_amt)
        st.info("Note: 'Scaled Amount' হলো আসল Amount-এর একটি স্ট্যান্ডার্ডাইজড ভার্সন।")
        # --- ---

    except Exception as e:
        st.error(f"Could not draw amount plot: {e}")
else:
    st.info(f"Waiting for 'card_sample_small.csv' to be uploaded to GitHub...")
