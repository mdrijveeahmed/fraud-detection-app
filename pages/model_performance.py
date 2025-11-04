import streamlit as st
import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc

st.set_page_config(page_title="Model Performance", page_icon="📈")
st.title("📈 Model Performance & Explainability")
st.write("On this page, we'll explore how our XGBoost model works behind the scenes.")

# --- Load Required Files ---
@st.cache_data
def load_assets():
    try:
        with open('fraud_detection_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('test_data.pkl', 'rb') as f:
            test_data = pickle.load(f)
        with open('feature_names.pkl', 'rb') as f:
            feature_names = pickle.load(f)
        
        X_test = test_data['X_test']
        y_test = test_data['y_test']
        return model, X_test, y_test, feature_names
    except FileNotFoundError:
        st.error("Required files (.pkl) not found. Please check if all files are uploaded to GitHub.")
        return None, None, None, None

model, X_test, y_test, feature_names = load_assets()

if model:
    # --- Generate model predictions ---
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] # Probability of fraud

    # --- 1. Feature Importance ---
    st.header("1. Feature Importance (Model Explainability)")
    st.write("Which features does the model prioritize when detecting fraud?")

    try:
        importance = model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance
        }).sort_values(by='Importance', ascending=False)

        # Show top 10 features
        fig_imp, ax_imp = plt.subplots(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(10), ax=ax_imp)
        ax_imp.set_title("Top 10 Most Important Features", fontsize=16)
        st.pyplot(fig_imp)

    except Exception as e:
        st.warning(f"Could not load Feature Importance chart: {e}")


    # --- 2. Confusion Matrix ---
    st.header("2. Confusion Matrix")
    st.write("A detailed breakdown of how many predictions were correct and incorrect.")

    cm = confusion_matrix(y_test, y_pred)
    fig_cm, ax_cm = plt.subplots(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'Fraud'], 
                yticklabels=['Normal', 'Fraud'], ax=ax_cm)
    ax_cm.set_xlabel("Predicted Class")
    ax_cm.set_ylabel("Actual Class")
    ax_cm.set_title("Confusion Matrix", fontsize=16)
    st.pyplot(fig_cm)
    
    st.info(f"""
    * **True Positive (TP):** {cm[1][1]} fraudulent cases correctly identified.
    * **False Negative (FN):** {cm[1][0]} fraudulent cases **missed** by the model. (This is the most harmful!)
    * **False Positive (FP):** {cm[0][1]} normal cases incorrectly flagged as 'Fraud'.
    """)

    # --- 3. ROC and Precision-Recall Curves ---
    st.header("3. Advanced Performance Curves")
    
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("ROC Curve")
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        
        fig_roc, ax_roc = plt.subplots()
        ax_roc.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
        ax_roc.plot([0, 1], [0, 1], 'k--') # Random guess line
        ax_roc.set_xlabel('False Positive Rate')
        ax_roc.set_ylabel('True Positive Rate')
        ax_roc.set_title('Receiver Operating Characteristic')
        ax_roc.legend(loc='lower right')
        st.pyplot(fig_roc)

    with col2:
        st.subheader("Precision-Recall Curve")
        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        pr_auc = auc(recall, precision)

        fig_pr, ax_pr = plt.subplots()
        ax_pr.plot(recall, precision, label=f'AUC = {pr_auc:.2f}')
        ax_pr.set_xlabel('Recall')
        ax_pr.set_ylabel('Precision')
        ax_pr.set_title('Precision-Recall Curve')
        ax_pr.legend(loc='lower left')
        st.pyplot(fig_pr)

    st.markdown("""
    **How to read these charts:**
    * **ROC Curve:** The closer the curve is to the top-left corner, the better the model.
    * **Precision-Recall Curve:** For imbalanced datasets like this, this curve is more important. The closer it is to the top-right corner, the better the model.
    """)

else:
    st.error("Required files to load the app were not found.")
