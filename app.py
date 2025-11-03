import streamlit as st
import pickle
import numpy as np
import pandas as pd

# --- 1. Load Saved Model and Scalers ---
# @st.cache_resource ensures files are loaded only once, even on app reload
@st.cache_resource
def load_assets():
    """Loads the model and scaler files."""
    try:
        with open('fraud_detection_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('amount_scaler.pkl', 'rb') as f:
            amount_scaler = pickle.load(f)
        with open('time_scaler.pkl', 'rb') as f:
            time_scaler = pickle.load(f)
        return model, amount_scaler, time_scaler
    except FileNotFoundError:
        st.error("Required .pkl files not found. Please ensure they are in the same folder as app.py.")
        return None, None, None
    except Exception as e:
        st.error(f"An error occurred while loading files: {e}")
        return None, None, None

# Load the assets
model, amount_scaler, time_scaler = load_assets()

# --- 2. Webpage Title and Interface ---
st.set_page_config(page_title="Fraud Detection System", page_icon="💳")
st.title("💳 Credit Card Fraud Detection")

if model:
    st.write("""
    This app uses a Machine Learning model to predict if a transaction
    is 'Normal' or 'Fraudulent'. Please enter the two required features below:
    """)

    # --- 3. Get User Input ---
    st.header("Enter Transaction Details:")

    col1, col2 = st.columns(2)
    with col1:
        time_input = st.number_input(
            "Transaction Time (in seconds)", 
            min_value=0, 
            value=40000,
            help="Time in seconds since the first transaction in the dataset."
        )
    with col2:
        amount_input = st.number_input(
            "Transaction Amount", 
            min_value=0.0, 
            value=120.50, 
            format="%.2f",
            help="The amount of the transaction."
        )

    # --- 4. Prediction Button and Logic ---
    if st.button("🔍 Predict", type="primary"):
        
        try:
            # --- a. Input Preprocessing ---
            # 1. Scale the input values (must use .transform(), not .fit_transform())
            scaled_time = time_scaler.transform([[time_input]])[0][0]
            scaled_amount = amount_scaler.transform([[amount_input]])[0][0]
            
            # 2. Use '0' for V1-V28 features
            # The original model has 30 features (V1-V28, scaled_amount, scaled_time).
            # For this demo, we assume the V1-V28 features are '0' (their mean value).
            v_features = np.zeros(28) 
            
            # 3. Create the complete feature vector in the correct order
            # The order during training was: [V1...V28, scaled_amount, scaled_time]
            feature_vector = np.concatenate([v_features, [scaled_amount, scaled_time]])
            
            # --- b. Prediction ---
            prediction = model.predict([feature_vector])
            probability = model.predict_proba([feature_vector])[0] # [Prob(0), Prob(1)]

            # --- c. Display Results ---
            st.subheader("Result:")
            if prediction[0] == 1:
                st.error(f"**Alert! This might be a fraudulent transaction!**", icon="🚨")
                st.warning(f"Probability of Fraud: **{probability[1] * 100:.2f}%**")
            else:
                st.success(f"**This is a normal transaction.**", icon="✅")
                st.info(f"Probability of Normal Transaction: {probability[0] * 100:.2f}%")
                
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
else:
    st.warning("App could not be loaded. Please ensure the .pkl files are in the correct location.")