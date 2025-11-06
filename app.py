import streamlit as st
import pickle
import numpy as np
import pandas as pd

# --- ১. পেজ ডিজাইন সেটআপ ---
# আমরা পেজটিকে "wide" মোডে সেট করবো এবং একটি সুন্দর আইকন দেবো
st.set_page_config(
    page_title="Fraud Detection System", 
    page_icon="💳",
    layout="wide"  # এটি আপনার অ্যাপকে পুরো স্ক্রিন জুড়ে দেখাবে
)

# --- ২. মডেল এবং স্কেলার লোড করা (আগের মতোই) ---
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
        st.error("Required .pkl files not found. Please ensure they are in the same folder as app.py.")
        return None, None, None
    except Exception as e:
        st.error(f"An error occurred while loading files: {e}")
        return None, None, None

model, amount_scaler, time_scaler = load_assets()

# --- ৩. সাইডবার (Sidebar) ডিজাইন ---
# আমরা সব ইনপুট বক্স সাইডবারে নিয়ে যাবো
st.sidebar.header("Enter Transaction Details:")

time_input = st.sidebar.number_input(
    "Transaction Time (in seconds)", 
    min_value=0, 
    value=40000,
    help="Time in seconds since the first transaction in the dataset."
)

amount_input = st.sidebar.number_input(
    "Transaction Amount", 
    min_value=0.0, 
    value=120.50, 
    format="%.2f",
    help="The amount of the transaction."
)

predict_button = st.sidebar.button("🔍 Predict Transaction", type="primary")


# --- ৪. মূল পেইজের কনটেন্ট ---
st.title("💳 Credit Card Fraud Detection System")
st.write("""
This app uses a Machine Learning model to predict if a transaction is 'Normal' or 'Fraudulent'. 
Enter the details in the sidebar on the left to get a real-time prediction.
""")

# একটি সুন্দর ব্যানার ইমেজ যোগ করা (এর জন্য কোনো ফাইল আপলোড করতে হবে না)
st.image("https://images.unsplash.com/photo-1555949963-ff9fe0c870eb?auto=format&fit=crop&w=1500", 
         caption="Real-time Fraud Analysis")

# ফলাফল দেখানোর জন্য একটি খালি জায়গা তৈরি করা
result_placeholder = st.empty()


# --- ৫. প্রেডিকশন এবং নতুন ফিচার (সুপারিশ) ---
if predict_button and model:
    try:
        # প্রেডিকশন লজিক (আগের মতোই)
        scaled_time = time_scaler.transform([[time_input]])[0][0]
        scaled_amount = amount_scaler.transform([[amount_input]])[0][0]
        v_features = np.zeros(28) 
        feature_vector = np.concatenate([v_features, [scaled_amount, scaled_time]])
        
        prediction = model.predict([feature_vector])
        probability = model.predict_proba([feature_vector])[0] 

        # ফলাফলগুলো মূল পেইজে দেখানো
        with result_placeholder.container():
            st.header("Prediction Result:")
            
            if prediction[0] == 1:
                st.error(f"**Alert! This might be a fraudulent transaction!**", icon="🚨")
                st.warning(f"Probability of Fraud: **{probability[1] * 100:.2f}%**")
                
                # --- নতুন ফিচার: কী করণীয় ---
                st.subheader("Recommended Action:")
                st.info("Please contact your bank or credit card provider immediately to report this suspicious activity. Do not share any personal information if you receive a call.")
                
            else:
                st.success(f"**This is a normal transaction.**", icon="✅")
                st.info(f"Probability of Normal Transaction: {probability[0] * 100:.2f}%")

                # --- নতুন ফিচার: কী করণীয় ---
                st.subheader("Recommended Action:")
                st.info("No action needed. Your transaction appears to be secure.")
                
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
elif not model:
     st.error("Model could not be loaded. Please check the files.")
