import streamlit as st
import pickle
import numpy as np
import pandas as pd

import gdown
import os

# Google Drive URLs
urls = {
    "pipe.pkl": "https://drive.google.com/file/d/15SIt-ncD5sME0xtIAX3MzJV-7doLa3gs/view?usp=sharing",
    "df.pkl":   "https://drive.google.com/file/d/1F9dQNiOC4IU8y3mbWjmroxAOF-EDvTb3/view?usp=sharing"
}

# Download files if they don't exist
for filename, url in urls.items():
    if not os.path.exists(filename):
        gdown.download(url, filename, quiet=False, fuzzy=True)

# Load pipeline and dataset
with open("pipe.pkl", "rb") as f:
    pipe = pickle.load(f)

with open("df.pkl", "rb") as f:
    df = pickle.load(f)

# Page Config
st.set_page_config(page_title="Laptop Cost Analyzer", page_icon="💻", layout="centered")
st.title("💻 Laptop Cost Analyzer")
st.markdown("### Predict the price of your dream laptop with AI 🚀")

# Layout: Two columns for better UI
col1, col2 = st.columns(2)

with col1:
    company = st.selectbox('🏢 Brand', df['Company'].unique())
    type = st.selectbox('💻 Type', df['TypeName'].unique())
    ram = st.selectbox('🔋 RAM (GB)', [2, 4, 8, 16, 24, 32, 64])
    weight = st.number_input('⚖️ Weight (Kg)', min_value=0.5, max_value=5.0, value=1.5)
    touchscreen = st.selectbox('🖐️ Touchscreen', ['No', 'Yes'])
    ips = st.selectbox('🖼️ IPS Display', ['No', 'Yes'])

with col2:
    screen_size = st.slider('📏 Screen Size (inches)', 10.0, 18.0, 13.0)
    resolution = st.selectbox('🖥️ Resolution',
                              ['1920x1080', '1366x768', '1600x900', '3840x2160',
                               '3200x1800', '2880x1800', '2560x1600', '2560x1440', '2304x1440'])
    cpu = st.selectbox('🧠 CPU', df['Cpu brand'].unique())
    hdd = st.selectbox('💾 HDD (GB)', [0, 128, 256, 512, 1024, 2048])
    ssd = st.selectbox('⚡ SSD (GB)', [0, 8, 128, 256, 512, 1024])
    gpu = st.selectbox('🎮 GPU', df['Gpu brand'].unique())
    os = st.selectbox('🖥️ OS', df['os'].unique())

# Prediction Button
if st.button('🎯 Predict Price'):
    # Convert binary features
    touchscreen = 1 if touchscreen == 'Yes' else 0
    ips = 1 if ips == 'Yes' else 0

    # PPI Calculation
    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])
    ppi = ((X_res ** 2) + (Y_res ** 2)) ** 0.5 / screen_size

    # Prepare query DataFrame with correct column names
    columns = ['Company', 'TypeName', 'Ram', 'Weight', 'Touchscreen', 'Ips',
           'ppi', 'Cpu brand', 'HDD', 'SSD', 'Gpu brand', 'os']

    query = pd.DataFrame([[company, type, ram, weight, touchscreen, ips,
                       ppi, cpu, hdd, ssd, gpu, os]], columns=columns)

    # Predict Price
    predicted_price = np.exp(pipe.predict(query)[0])

    # Display in Style
    st.success(f"💰 **Predicted Laptop Price: ₹ {int(predicted_price):,}**")
    st.balloons()

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using **Streamlit** & **Machine Learning**")
