import streamlit as st
import pickle
import numpy as np
import pandas as pd

# --- ১. পেজ ডিজাইন সেটআপ ---
st.set_page_config(
    page_title="Fraud Detection System", 
    page_icon="💳",
    layout="wide"
)

# --- ২. কাস্টম CSS (ডিজাইন ভালো করার জন্য) ---
def local_css(file_name):
    try:
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        # যদি style.css ফাইল না থাকে, তবে একটি ডিফল্ট স্টাইল ব্যবহার করুন
        st.markdown("""
        <style>
        /* টাইটেলের রঙ পরিবর্তন */
        h1 {
            color: #4CAF50; /* সবুজ রঙ */
            text-align: center;
        }
        
        /* বাটন ডিজাইন */
        .stButton > button {
            border: 2px solid #4CAF50;
            background-color: #4CAF50;
            color: white;
            padding: 12px 28px;
            border-radius: 8px;
            font-size: 16px;
            width: 100%; /* বাটন চওড়া করা */
        }
        .stButton > button:hover {
            background-color: white;
            color: #4CAF50;
            border: 2px solid #4CAF50;
        }

        /* ইনপুট বক্সের চারপাশ */
        .stNumberInput {
            background-color: #f0f2f6;
            border-radius: 8px;
            padding: 10px;
        }
        
        /* সফল (Success) বক্সের ডিজাইন */
        .stSuccess {
            background-color: #e6f7ec;
            border: 1px solid #4CAF50;
            border-radius: 8px;
        }
        
        /* এরর (Error) বক্সের ডিজাইন */
        .stError {
            background-color: #fdecea;
            border: 1px solid #EA4335;
            border-radius: 8px;
        }
        </style>
        """, unsafe_allow_html=True)

# CSS ফাংশনটি কল করুন
local_css("style.css") # (এই ফাইলটির দরকার নেই, আমরা উপরের কোডটি ব্যবহার করছি)


# --- ৩. মডেল এবং স্কেলার লোড করা ---
@st.cache_resource
def load_assets():
    try:
        with open('fraud_detection_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('amount_scaler.pkl', 'rb') as f:
            amount_scaler = pickle.load(f)
        with open('time_scaler.pkl', 'rb') as f:
            time_scaler = pickle.load(f)
        return model, amount_scaler, time_scaler
    except FileNotFoundError:
        st.error("Required .pkl files not found.")
        return None, None, None
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None, None, None

model, amount_scaler, time_scaler = load_assets()

# --- ৪. মূল পেইজের কনটেন্ট ---
st.title("💳 Credit Card Fraud Detection System")
st.image("https://images.unsplash.com/photo-1555949963-ff9fe0c870eb?auto=format&fit=crop&w=1500")

st.header("Enter Transaction Details:")

# ইনপুট বক্সগুলোকে পাশাপাশি দেখানো (columns)
col1, col2 = st.columns(2)

with col1:
    time_input = st.number_input(
        "Transaction Time (in seconds)", 
        min_value=0, 
        value=40000,
        help="Time in seconds since the first transaction."
    )

with col2:
    amount_input = st.number_input(
        "Transaction Amount", 
        min_value=0.0, 
        value=120.50, 
        format="%.2f",
        help="The amount of the transaction."
    )

st.write("") # একটু ফাঁকা জায়গা

# --- ৫. প্রেডিকশন এবং ফলাফল ---
if st.button("🔍 Predict Transaction"):
    if model:
        try:
            scaled_time = time_scaler.transform([[time_input]])[0][0]
            scaled_amount = amount_scaler.transform([[amount_input]])[0][0]
            v_features = np.zeros(28) 
            feature_vector = np.concatenate([v_features, [scaled_amount, scaled_time]])
            
            prediction = model.predict([feature_vector])
            probability = model.predict_proba([feature_vector])[0] 

            st.header("Prediction Result:")
            if prediction[0] == 1:
                st.error(f"**Alert! Fraudulent Transaction Detected!**", icon="🚨")
                st.warning(f"Probability of Fraud: **{probability[1] * 100:.2f}%**")
                st.info("Recommended Action: Please contact your bank immediately.")
            else:
                st.success(f"**This is a Normal Transaction.**", icon="✅")
                st.info(f"Probability of Normal Transaction: {probability[0] * 100:.2f}%")
                
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
    else:
         st.error("Model could not be loaded. Please check the files.")
