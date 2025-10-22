# app.py
import numpy as np
import streamlit as st
import cv2
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score

# ---------- Page config ----------
st.set_page_config(page_title="LOOCV & RGB Analyzer", page_icon="🧪", layout="wide")

# ---------- Background controls (UI) ----------
st.sidebar.header("🎨 Background")
bg_mode = st.sidebar.selectbox("Background mode", ["Solid ocean blue", "Photo", "Video"], index=0)

# Defaults
photo_url_default = "https://images.unsplash.com/photo-1507525428034-b723cf961d3e"
video_url_default = "https://cdn.coverr.co/videos/coverr-ocean-waves-1536/1080p.mp4"

if bg_mode == "Photo":
    photo_url = st.sidebar.text_input("Photo URL", value=photo_url_default)
elif bg_mode == "Video":
    video_url = st.sidebar.text_input("Video MP4 URL (autoplay, loop, muted)", value=video_url_default)

# ---------- Inject CSS / HTML for background ----------
if bg_mode == "Solid ocean blue":
    st.markdown("""
    <style>
    .main { background-color: #0077be !important; }          /* bright ocean blue */
    [data-testid="stSidebar"] { background-color: #0a2f47; } /* deep blue sidebar */
    </style>
    """, unsafe_allow_html=True)

elif bg_mode == "Photo":
    st.markdown(f"""
    <style>
    .main {{
        background: url("{photo_url}") no-repeat center center fixed !important;
        background-size: cover !important;
    }}
    .main::before {{
        content:"";
        position: fixed; inset: 0;
        background: rgba(0, 60, 100, 0.40);   /* blue overlay for readability */
        z-index: -1;
    }}
    [data-testid="stSidebar"] {{
        background-color: rgba(10, 47, 71, 0.85);
    }}
    </style>
    """, unsafe_allow_html=True)

elif bg_mode == "Video":
    st.markdown("""
    <style>
    video.bgvid {
      position: fixed; right: 0; bottom: 0; min-width: 100%; min-height: 100%;
      z-index: -1; object-fit: cover; filter: brightness(0.85) saturate(1.1);
    }
    .main::before { content:""; position: fixed; inset: 0; z-index: -1; }
    [data-testid="stSidebar"] { background-color: rgba(10, 47, 71, 0.85); }
    </style>
    """, unsafe_allow_html=True)
    st.markdown(f"""
    <video class="bgvid" autoplay loop muted playsinline>
        <source src="{video_url}" type="video/mp4">
    </video>
    """, unsafe_allow_html=True)

# ---------- Title ----------
st.title("Machine Learning & Image RGB Analyzer")

# Tabs
tab1, tab2 = st.tabs(["🎨 RGB Image Analyzer", "🔎 Algorithm"])

# ---------- Helper: pretty metric card ----------
def metric_card(title, value, note="", accent="#00e676"):
    st.markdown(
        f"""
        <div style="
            background:#1b1b1b; border-left:6px solid {accent};
            padding:12px 16px; border-radius:10px; color:#e0e0e0; margin-top:8px;">
          <div style="font-weight:700">{title}</div>
          <div style="font-size:22px; margin-top:6px"><b>{value}</b></div>
          <div style="opacity:0.85; margin-top:4px">{note}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

# ---------------- Tab 2: LOOCV / Calibration ----------------
with tab2:
    st.header("📊 Calibration Samples")

    # Calibration data (R,G fixed; B varies)
    X = np.array([
        [254, 254,   6],
        [254, 254,  52],
        [254, 254,   3],
        [254, 254,  64],
        [254, 254,  72],
        [254, 254, 169]
    ], dtype=float)

    # NOTE: Units — your y looks like "ppm" or "mg/L" (numerically same). Keep consistent wording.
    y = np.array([599.814, 449.8605, 649.7985, 399.876, 299.907, 249.9225], dtype=float)

    # Model + LOOCV prediction
    model = RandomForestRegressor(n_estimators=400, random_state=42)
    loo = LeaveOneOut()
    y_pred = cross_val_predict(model, X, y, cv=loo)

    # Metrics
    mse = mean_squared_error(y, y_pred)
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y, y_pred))

    # Adjusted R^2 (p predictors = 3)
    n, p = X.shape
    if n > p + 1:
        r2_adj = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    else:
        r2_adj = float("nan")

    # Relative RMSE (as % of range)
    rng = float(np.ptp(y)) if np.ptp(y) > 0 else float("nan")
    rrmse = (rmse / rng * 100) if rng == rng else float("nan")  # safe NaN check

    # Display: Actual vs Predicted (compact)
    st.subheader("Actual vs Predicted (LOOCV)")
    for actual, pred in zip(y, y_pred):
        st.write(f"Actual: {actual:.6f} | Predicted: {pred:.6f}")

    # Display: metrics
    st.subheader("Model Performance")
    metric_card("RMSE", f"{rmse:.3f}", note="Root Mean Square Error")
    metric_card("R²", f"{r2:.3f}", note="Coefficient of determination")
    metric_card("Adjusted R²", f"{r2_adj:.3f}" if r2_adj==r2_adj else "N/A",
                note="Penalizes small sample size")
    metric_card("Relative RMSE", f"{rrmse:.1f}%" if rrmse==rrmse else "N/A",
                note="RMSE ÷ range(y) × 100")

    st.info("Note: With only n=6 samples and R,G constant, results are preliminary. "
            "Consider more standards and variability across R,G,B, plus replicates.")

    # Predict new sample
    st.subheader("Predict New Sample")
    colr, colg, colb = st.columns(3)
    with colr:
        r = st.number_input("R (0–255)", min_value=0, max_value=255, value=254, step=1, format="%d")
    with colg:
        g = st.number_input("G (0–255)", min_value=0, max_value=255, value=254, step=1, format="%d")
    with colb:
        b = st.number_input("B (0–255)", min_value=0, max_value=255, value=64, step=1, format="%d")

    new_rgb = np.array([[r, g, b]], dtype=float)
    model.fit(X, y)
    predicted_conc = model.predict(new_rgb)

    st.markdown(
        f"""
        <div style="
            background-color:#1b1b1b;
            border-left:6px solid #00e676;
            padding:12px 18px;
            border-radius:8px;
            color:#e0e0e0;
            font-family:monospace;
        ">
            <b>🧪 Predicted Chlorine Concentration</b><br>
            <span style='color:#81c784'>RGB:</span> {new_rgb.astype(int).tolist()}<br>
            <span style='color:#00e676'>Predicted:</span>
            <b>{predicted_conc[0]:.6f} mg/L</b>
        </div>
        """,
        unsafe_allow_html=True
    )

# ---------------- Tab 1: RGB Analyzer ----------------
with tab1:
    st.header("Average RGB Calculator")

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        # Read image using OpenCV
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Average RGB
        avg_rgb = img_rgb.mean(axis=(0, 1)).astype(int)
        st.write("Average RGB value:", avg_rgb.tolist())

        # Show image and average color side by side
        col1, col2 = st.columns(2)
        with col1:
            st.image(img_rgb, caption="Uploaded Image", use_container_width=True)
        with col2:
            color_patch = np.ones((100, 100, 3), dtype=np.uint8) * avg_rgb
            st.image(color_patch, caption=f"Avg Color {avg_rgb.tolist()}", use_container_width=False)
