# frontend_streamlit/app.py

import requests
import streamlit as st
from PIL import Image

# ── FULL, ABSOLUTE ENDPOINTS ───────────────────────────────────────────
# (bypass any dodgy BACKEND_URL env var)
PREDICT_URL  = "http://backend:8000/predict"
FEEDBACK_URL = "http://backend:8000/feedback"

CLASSES = ["Benign", "Early", "Pre", "Pro"]

st.set_page_config(
    page_title="Cancer Classifier",
    page_icon="🩸",
    layout="wide",
)

# ── simple CSS ────────────────────────────────────────────────────────
st.markdown("""
<style>
  .card { background:white;padding:1rem;border-radius:8px;
          box-shadow:0 2px 4px rgba(0,0,0,0.1);margin:1rem 0; }
  .stButton>button { width:100%; }
</style>""", unsafe_allow_html=True)

# ── sidebar ──────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Model status")
    st.success("Up-to-date")
    st.markdown("**Classes:** Benign · Early · Pre · Pro")
    st.markdown("---")
    st.caption("Upload an image → Predict → Provide feedback (if wrong)")

# ── main card ────────────────────────────────────────────────────────
st.markdown('<div class="card">', unsafe_allow_html=True)
st.header("🎯 Leukemia Classifier")

col1, col2 = st.columns([2, 1])

with col1:
    uploaded = st.file_uploader("Choose a JPG/PNG file", type=["jpg", "png"])
    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, caption="Preview", width=300)

        if st.button("Predict", key="predict"):
            with st.spinner("Calling inference…"):
                resp = requests.post(
                    PREDICT_URL,
                    files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type)},
                    timeout=60,
                )
            if resp.ok:
                st.session_state["pred"] = resp.json()
            else:
                # show full URL and status
                st.error(f"Got {resp.status_code} from {PREDICT_URL}")

with col2:
    st.subheader("Prediction")
    if "pred" in st.session_state:
        idx   = st.session_state["pred"]["predicted_class"]
        label = CLASSES[idx]
        st.success(label)
    else:
        st.info("No prediction yet")

    st.markdown("---")
    st.subheader("Feedback")
    if "pred" in st.session_state:
        feedback = st.selectbox(
            "Correct label (if wrong):",
            options=CLASSES,
            index=idx,
            key="feedback_select"
        )
        if st.button("Submit Feedback", key="submit_feedback"):
            with st.spinner("Saving feedback…"):
                fb_resp = requests.post(
                    FEEDBACK_URL,
                    data={
                        "correct_label":   feedback,
                        "predicted_label": label,
                    },
                    files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type)},
                    timeout=60,
                )
            if fb_resp.ok:
                data = fb_resp.json()
                count = data.get("feedback_count")
                if count is not None:
                    st.success(f"Thanks! We’ve stored your feedback.")
                else:
                    st.error(f"Unexpected response:\n{data}")
            else:
                st.error(f"Got {fb_resp.status_code} from {FEEDBACK_URL}")

st.markdown('</div>', unsafe_allow_html=True)

# ── footer ───────────────────────────────────────────────────────────
st.markdown(
    '<footer style="text-align:center;color:#888;margin-top:2rem;">'
    "© 2025 MLOps Demo</footer>",
    unsafe_allow_html=True,
)
