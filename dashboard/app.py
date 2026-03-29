import streamlit as st
import requests
import json
from PIL import Image
import io
import cv2
import numpy as np

# Adjust base URL according to your local backend config
API_URL = "http://localhost:8000/analyze-shelf"

st.set_page_config(page_title="Smart Shelf Intelligence Dashboard", layout="wide", page_icon="🛒")

st.title("🛒 Smart Retail Shelf Intelligence System")
st.markdown("Upload a snapshot of your retail shelf to analyze product positioning, detect empty slots, and verify shelf health.")

uploaded_file = st.file_uploader("Choose a shelf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Set up layout split for image viewing and analytics
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Shelf Snapshot")
        image = Image.open(uploaded_file)
        st.image(image, use_column_width=True)

    with col2:
        st.subheader("Intelligent Analysis & Health Metrics")
        if st.button("Run Shelf Analysis"):
            with st.spinner('Contacting computer vision inference engine...'):
                # Reset file pointer and prepare file for API
                uploaded_file.seek(0)
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                
                try:
                    response = requests.post(API_URL, files=files)
                    
                    if response.status_code == 200:
                        results = response.json()
                        
                        st.success("Intelligent Analysis Complete!")
                        st.metric(label="Overall Shelf Health Score", value=f"{results['shelf_health_score']}%")
                        
                        m1, m2, m3 = st.columns(3)
                        m1.metric(label="Detected Products", value=results['total_products_detected'])
                        m2.metric(label="Empty Slots", value=results['empty_slots_detected'])
                        m3.metric(label="Misplaced Items", value=results['misplaced_items_detected'])
                        
                        st.markdown("---")
                        st.markdown("### Compliance Alerts")
                        if results['empty_slots_detected'] > 0:
                            st.warning(f"⚠️ Action Required: **{results['empty_slots_detected']}** empty slots observed.")
                        if results['misplaced_items_detected'] > 0:
                            st.error(f"🚨 Action Required: **{results['misplaced_items_detected']}** items appear to be in the wrong location")
                        
                        if results['empty_slots_detected'] == 0 and results['misplaced_items_detected'] == 0:
                            st.info("✅ Shelf is fully compliant! No anomalies detected.")
                            
                        with st.expander("Review Raw API Telemetry"):
                            st.json(results)
                            
                    else:
                        st.error(f"Service Error {response.status_code}: {response.text}")
                except Exception as e:
                    st.error(f"Inference service is unreachable. Ensure 'src/api/main.py' is running. Root Cause: {str(e)}")

st.markdown("---")
st.caption("*Retail Shelf Intelligence Dashboard Powered by YOLOv8 (Object Detection) & Isolation Forests (Anomaly Det).*")
