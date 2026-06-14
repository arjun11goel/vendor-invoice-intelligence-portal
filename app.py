import streamlit as st
import pandas as pd

from inference.predict_freight import predict_freight_cost
from inference.predict_invoice_flag import predict_invoice_flag

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Vendor Invoice Intelligence Portal", layout="wide")

# ---------------- HEADER ----------------
st.title("Vendor Invoice Intelligence Portal")

st.markdown("""
An internal ML-powered system designed to assist in:

- 🚚 Freight cost prediction  
- ⚠️ Invoice risk detection    

---
""")

# ---------------- SIDEBAR ----------------
st.sidebar.title("Navigation")

module = st.sidebar.radio(
    "Choose Module",
    ("🏠 Overview", "🚚 Freight Cost Prediction", "⚠️ Invoice Risk Flagging")
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Business Impact")

st.sidebar.markdown("""
- Better cost estimation  
- Faster invoice processing  
- Detection of abnormal patterns  
- Reducing manual effort and financial errors
""")

# ---------------- OVERVIEW ----------------
if module == "🏠 Overview":
    st.header(" How This System Works")

    st.markdown("""
This portal uses Machine Learning models trained on historical vendor invoice data.

### Freight Cost Prediction

Freight cost depends mainly on:
- **Dollars** → Total invoice value  
- **Quantity** → Number of items  

The model learns patterns between these values and historical freight charges.

---

### Invoice Risk Flagging

Invoice risk is determined using multiple business-related factors:

- **invoice_quantity** → Number of units in invoice  
- **invoice_dollars** → Total invoice amount  
- **Freight** → Shipping cost  
- **total_brands** → Variety of brands in invoice  
- **days_po_to_invoice** → Delay between purchase order and invoice  
- **total_item_quantity** → Total quantity of items  
- **total_item_dollars** → Total value of items  

The model identifies abnormal combinations of these features to flag risky invoices.

---

### Models Used

- Linear Regression  
- Decision Tree Regression  
- Random Forest Regression 
- Classification models for invoice flagging  

The best-performing model is selected based on evaluation metrics.

---

Use the modules on the left to try predictions interactively.
""")

# ---------------- FREIGHT COST ----------------
elif module == "🚚 Freight Cost Prediction":
    st.header("Freight Cost Prediction")

    st.markdown("""
    ### How This Works

    The freight cost prediction model estimates shipping cost based on historical patterns between invoice value and quantity.

    The model learns how freight cost varies with:
    - Order size  
    - Shipment value    

    Using this relationship, it predicts the expected freight cost for new inputs.
    """)
    
    st.markdown("### Sample Input Format")

    sample_df = pd.DataFrame({
        "Dollars": [18500, 9000, 3000, 200],
        "Quantity": [10, 5, 2, 1]
    })

    st.dataframe(sample_df)

    st.markdown("### Enter Your Values For Freight Cost Prediction")

    with st.form("freight_form"):
        dollars = st.number_input("Dollars (Shipment Value)", min_value=0.0)
        quantity = st.number_input("Quantity (Order Size)", min_value=0.0)

        submit = st.form_submit_button("Predict Freight Cost")

    if submit:

        if dollars == 0 or quantity == 0:
            st.warning(" Please enter valid non-zero values")
        else:
            with st.spinner("🔄 Predicting freight cost..."):
                input_data = {
                    "Dollars": [dollars],
                    "Quantity": [quantity]
                    }

                result = predict_freight_cost(input_data)

            st.success("Prediction Completed")

            st.write("### Result")
            st.dataframe(result)

# ---------------- INVOICE FLAG ----------------
elif module == "⚠️ Invoice Risk Flagging":
    st.header("Invoice Manual Approval Flagging")

    st.markdown("""
### How This Works

The model analyzes invoice patterns and flags invoices that appear unusual based on:
- Cost inconsistencies  
- Freight vs invoice ratio  
- Delivery delays  
- Item distribution patterns  
""")

    st.markdown("### Sample Input Format")

    sample_df = pd.DataFrame({
        "invoice_quantity": [10, 2, 8],
        "invoice_dollars": [8200, 75000, 16000],
        "Freight": [400, 150, 1000],
        "total_brands": [2, 1, 3],
        "days_po_to_invoice": [3, 12, 5],
        "total_item_quantity": [10, 3, 8],
        "total_item_dollars": [8000, 50000, 15000],
        "Type": ["🟢 SAFE", "🔴 MANUAL", "⚖️ BORDERLINE"]
    })

    st.dataframe(sample_df)

    st.markdown("### Enter Invoice Details For Invoice Risk Detection")

    with st.form("invoice_form"):
        invoice_quantity = st.number_input("Invoice Quantity (Number of units in invoice)", min_value=0.0)
        invoice_dollars = st.number_input("Invoice Dollars (Total invoice amount)", min_value=0.0)
        freight = st.number_input("Freight (Shipping cost)", min_value=0.0)
        total_brands = st.number_input("Total Brands(Variety of brands in invoice)", min_value=0.0)
        days_po_to_invoice = st.number_input("Days PO to Invoice (Delay between purchase order and invoice)", min_value=0.0)
        total_item_quantity = st.number_input("Total Item Quantity (Total quantity of items)", min_value=0.0)
        total_item_dollars = st.number_input("Total Item Dollars (Total value of items)", min_value=0.0)

        submit = st.form_submit_button("Evaluate Invoice Risk")

    if submit:

        if invoice_dollars == 0 or invoice_quantity == 0:
            st.warning(" Please fill required fields properly")
        else:
            with st.spinner("🔄 Evaluating Invoice Flagging..."):
                input_data = {
                    "invoice_quantity": [invoice_quantity],
                    "invoice_dollars": [invoice_dollars],
                    "Freight": [freight],
                    "total_brands": [total_brands],
                    "days_po_to_invoice": [days_po_to_invoice],
                    "total_item_quantity": [total_item_quantity],
                    "total_item_dollars": [total_item_dollars]
                }
                
                result = predict_invoice_flag(input_data)

            st.success(" Evaluation Completed")

            st.write("### Result")
            st.dataframe(result)

            prediction = result["Predicted_Invoice_Flag"].values[0]

            if prediction == 1:
                st.error("Invoice requires **MANUAL APPROVAL**")
            else:
                st.success("Invoice is **SAFE for Auto-Approval**")