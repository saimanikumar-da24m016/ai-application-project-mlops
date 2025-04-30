import streamlit as st
import requests, os
from PIL import Image

# ---------------------------------------------------------------- config
API_URL = os.getenv("BACKEND_URL", "http://backend:8000/predict")
CLASSES = ["Benign", "Early", "Pre", "Pro"]

st.set_page_config(
    page_title="Lymphoblastic Leukemia Cancer Classifier",
    page_icon="🩸",
    layout="wide",
)

# minimal CSS for a card panel and full-width button
st.markdown(
    """
    <style>
      .card { background: white; padding: 1rem; border-radius: 8px;
              box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin: 1rem 0; }
      .stButton>button { width: 100%; }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Model status")
    st.success("Up-to-date")
    st.markdown("**Classes:** Benign · Early · Pre · Pro")
    st.markdown("---")
    st.caption("Upload a WBC image and click **Predict**.")

# ---------------------------------------------------------------- main card
st.markdown('<div class="card">', unsafe_allow_html=True)

st.header("🎯 ALL Leukemia Classifier")

# split into two columns
col1, col2 = st.columns([2, 1])

with col1:
    uploaded = st.file_uploader("Choose a JPG/PNG file", type=["jpg", "png"])
    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, caption="Preview", width=300)

        if st.button("Predict", key="predict"):
            with st.spinner("Running inference…"):
                resp = requests.post(
                    API_URL,
                    files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type)},
                    timeout=60,
                )
            if resp.ok:
                st.session_state["pred"] = resp.json()
            else:
                st.error(f"Error {resp.status_code}")

with col2:
    st.subheader("Prediction")
    if "pred" in st.session_state:
        idx = st.session_state["pred"]["predicted_class"]
        label = CLASSES[idx] if idx < len(CLASSES) else f"Class {idx}"
        st.success(label)
    else:
        st.info("No prediction yet")

    st.markdown("---")
    st.subheader("Feedback")
    if "pred" in st.session_state:
        feedback = st.selectbox(
            "If prediction is wrong, select the correct label:",
            options=CLASSES,
            index=CLASSES.index(label) if "label" in locals() and label in CLASSES else 0,
            key="feedback_select"
        )
        if st.button("Submit Feedback", key="submit_feedback"):
            # Here you would send feedback back to your backend or store it
            st.success(f"Feedback received: {feedback}")
    else:
        st.write("Upload & predict first")

st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------- footer
st.markdown(
    '<footer style="text-align:center; color:#888; margin-top:2rem;">'
    "© 2025 MLOps Demo</footer>",
    unsafe_allow_html=True,
)
