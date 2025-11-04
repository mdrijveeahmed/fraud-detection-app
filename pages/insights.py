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
    # এইবার আমরা ছোট স্যাম্পল ফাইলটি লোড করবো
    data_url = 'card_sample.csv' 

    try:
        df = pd.read_csv(data_url)
        return df
    except FileNotFoundError:
        st.error(f"Error: '{data_url}' not found in your GitHub repository.")
        st.warning("Please ensure you have created and uploaded 'card_sample.csv' from Google Colab.")
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
        ax.set_title('Normal (0) vs. Fraud (1) (in 50k sample)')
        ax.set_xticklabels(['Normal (0)', 'Fraud (1)'])
        st.pyplot(fig)
        
        fraud_percentage = (class_counts.get(1, 0) / class_counts.sum()) * 100
        st.warning(f"Only **{fraud_percentage:.3f}%** of transactions in this sample are fraudulent.")

    except Exception as e:
        st.error(f"Could not draw plot: {e}")


    # --- 2. Transaction Amount Distribution ---
    st.header("2. Transaction Amount Distribution")
    st.write("How does the transaction amount differ between normal and fraudulent cases?")

    try:
        # Filter for rows where Amount > 0 to avoid log(0) issues if using log_scale
        df_plot = df[df['Amount'] > 0].copy()

        fig_amt, (ax1, ax2) = plt.subplots(ncols=2, figsize=(18, 6))

        # Normal transactions
        sns.histplot(df_plot[df_plot['Class'] == 0]['Amount'], bins=100, ax=ax1, color='green', kde=True)
        ax1.set_title('Normal Transaction Amount')
        ax1.set_xlabel('Amount')
        ax1.set_ylabel('Frequency')
        ax1.set_xlim(0, 3000) # Limiting x-axis to see details

        # Fraudulent transactions
        if 1 in df_plot['Class'].values:
            sns.histplot(df_plot[df_plot['Class'] == 1]['Amount'], bins=50, ax=ax2, color='red', kde=True)
            ax2.set_title('Fraudulent Transaction Amount')
        else:
            ax2.set_title('No Fraudulent Transactions in this sample')
            
        ax2.set_xlabel('Amount')
        ax2.set_ylabel('Frequency')

        st.pyplot(fig_amt)
        st.info("Note: Most transactions (both normal and fraudulent) are for small amounts.")

    except Exception as e:
        st.error(f"Could not draw amount plot: {e}")
else:
    st.info("Waiting for 'card_sample.csv' to be uploaded to GitHub...")
