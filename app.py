# app.py

import streamlit as st
from generator import generate_images

st.set_page_config(page_title="Digit Generator (DCGAN)", layout="centered")
st.title("🧠 DCGAN Handwritten Digit Generator")
st.markdown("Generate high-quality MNIST-style digits using a Conditional DCGAN.")

digit = st.selectbox("🎯 Choose a digit to generate (0–9):", list(range(10)))
count = st.slider("🖼️ Number of images:", 1, 10, 5)

if st.button("🚀 Generate"):
    with st.spinner("Generating images..."):
        images = generate_images(digit, count)

    st.success("✅ Images generated!")
    st.subheader(f"Digit: {digit}")

    cols = st.columns(count)
    for idx, col in enumerate(cols):
        col.image(images[idx], width=100, clamp=True, caption=f"#{idx+1}")
