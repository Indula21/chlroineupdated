# app.py
import base64
import numpy as np
import streamlit as st
import cv2
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score

# ---------- Page config ----------
st.set_page_config(page_title="LOOCV & RGB Analyzer", page_icon="🧪", layout="wide")

# ---------- Sidebar controls ----------
st.sidebar.header("🎨 Background Mode")
bg_mode = st.sidebar.selectbox(
    "Select background type:",
    ["Solid ocean blue", "Photo URL", "Upload photo", "Video URL", "Upload video"],
    index=0
)

# Defaults for URL modes
photo_url_default = "https://images.unsplash.com/photo-1507525428034-b723cf961d3e"
video_url_default = "https://cdn.coverr.co/videos/coverr-ocean-waves-1536/1080p.mp4"

photo_url = None
video_url = None
photo_b64 = None
video_b64 = None
photo_mime = "image/jpeg"   # fallback

# ---------- Inputs depending on mode ----------
if bg_mode == "Photo URL":
    photo_url = st.sidebar.text_input("Photo URL", value=photo_url_default)

elif bg_mode == "Video URL":
    video_url = st.sidebar.text_input("Video MP4 URL", value=video_url_default)

elif bg_mode == "Upload photo":
    up_img = st.sidebar.file_uploader("Upload background image", type=["jpg", "jpeg", "png"])
    if up_img is not None:
        photo_mime = up_img.type or "image/jpeg"
        img_bytes = up_img.read()
        photo_b64 = base64.b64encode(img_bytes).decode()

elif bg_mode == "Upload video":
    up_vid = st.sidebar.file_uploader("Upload background video (MP4 recommended, keep small)", type=["mp4", "webm", "ogg"])
    if up_vid is not None:
        video_mime = up_vid.type or "video/mp4"
        vid_bytes = up_vid.read()
        video_b64 = base64.b64encode(vid_bytes).decode()
        # Build a data URL
        video_url = f"data:{video_mime};base64,{video_b64}"

# ---------- Inject CSS / HTML for backgrounds ----------
if bg_mode == "Solid ocean blue":
    st.markdown("""
    <style>
    .main { background-color: #0077be !important; }  /* Ocean blue */
    [data-testid="stSidebar"] { background-color: #0a2f47; }
    </style>
    """, unsafe_allow_html=True)

elif bg_mode in ["Photo URL", "Upload photo"]:
    # Resolve the background image source
    if bg_mode == "Photo URL" and photo_url:
        bg_image = photo_url
    elif bg_mode == "Upload photo" and photo_b64:
        bg_image = f"data:{photo_mime};base64,{photo_b64}"
    else:
        bg_image = ""  # nothing yet

    if bg_image:
        st.markdown(f"""
        <style>
        .main {{
            background: url("{bg_image}") no-repeat center center fixed !important;
            background-size: cover !important;
        }}
        .main::before {{
            content:"";
            position: fixed; inset: 0;
            background: rgba(0, 60, 100, 0.45); /* blue tint overlay for readability */
            z-index: -1;
        }}
        [data-testid="stSidebar"] {{
            background-color: rgba(10, 47, 71, 0.85);
        }}
        </style>
        """, unsafe_allow_html=True)

elif bg_mode in ["Video URL", "Upload video"]:
    # Use video_url either from URL input or base64 data URL
    if video_url:
        st.markdown(f"""
        <style>
        video.bgvid {{
          position: fixed; right: 0; bottom: 0;
          min-width: 100%; min-height: 100%;
          z-index: -1; object-fit: cover;
          filter: brightness(0.8) saturate(1.15);
        }}
        .main::before {{
          content:""; position: fixed; inset: 0;
          background: rgba(0, 40, 70, 0.45); z-index: -1;
        }}
        [data-testid="stSidebar"] {{
          background-color: rgba(10, 47, 71, 0.8);
        }}
        </style>

        <video class="bgvid" autoplay loop muted playsinline>
            <source src="{video_url}" type="video/mp4">
        </video>
        """, unsafe_allow_html=True)
    else:
        st.warning("Upload a video or provide a valid MP4 URL to enable the background video.")

# ---------- App Title ----------
st.title("🌊 Machine Learning & Image RGB Analyzer")

# Tabs
tab1, tab2 = st.tabs(["🎨 RGB Image Analyzer", "🔎 Algorithm"])

# ---------- Helper: metric card ----------
def metric_card(title, value, note="", accent="#00e676"):
    st.markdown(
        f"""
        <div style="
            background:#1b1b1b; border-left:6px solid {accent};
            padding:12px 16px; border-radius:10px;
            color:#e0e0e0; margin-top:8px;">
          <div style="font-weight:700">{title}</div>
          <div style="font-size:22px; margin-top:6px"><b>{value}</b></div>
          <div style="opacity:0.85; margin-top:4px">{note}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

# ---------- Tab 2: Algorithm ----------
with tab2:
    st.header("📊 Calibration Samples (LOOCV Validation)")

    X = np.array([
        [254, 254,   6],
        [254, 254,  52],
        [254, 254,   3],
        [254, 254,  64],
        [254, 254,  72],
        [254, 254, 169]
    ], dtype=float)

    y = np.array([599.814, 449.8605, 649.7985, 399.876, 299.907, 249.9225], dtype=float)

    model = RandomForestRegressor(n_estimators=400, random_state=42)
    loo = LeaveOneOut()
    y_pred = cross_val_predict(model, X, y, cv=loo)

    mse = mean_squared_error(y, y_pred)
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y, y_pred))

    n, p = X.shape
    r2_adj = 1 - (1 - r2) * (n - 1) / (n - p - 1) if n > p + 1 else float("nan")
    rrmse = (rmse / np.ptp(y) * 100) if np.ptp(y) > 0 else float("nan")

    st.subheader("Actual vs Predicted (Leave-One-Out Cross Validation)")
    for actual, pred in zip(y, y_pred):
        st.write(f"Actual: {actual:.6f} | Predicted: {pred:.6f}")

    st.subheader("Model Performance Metrics")
    metric_card("RMSE", f"{rmse:.3f}", "Root Mean Square Error")
    metric_card("R²", f"{r2:.3f}", "Coefficient of Determination")
    metric_card("Adjusted R²", f"{r2_adj:.3f}" if r2_adj==r2_adj else "N/A", "Penalty for small sample size")
    metric_card("Relative RMSE", f"{rrmse:.2f}%" if rrmse==rrmse else "N/A", "RMSE ÷ range × 100")

    st.info("⚠️ Only 6 samples used (R,G fixed). Add more varied samples and replicates for better calibration.")

    st.subheader("Predict New Sample")
    colr, colg, colb = st.columns(3)
    with colr:
        r = st.number_input("R (0–255)", 0, 255, 254)
    with colg:
        g = st.number_input("G (0–255)", 0, 255, 254)
    with colb:
        b = st.number_input("B (0–255)", 0, 255, 64)

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
            font-family:monospace;">
            <b>🧪 Predicted Chlorine Concentration</b><br>
            <span style='color:#81c784'>RGB:</span> {new_rgb.astype(int).tolist()}<br>
            <span style='color:#00e676'>Predicted:</span>
            <b>{predicted_conc[0]:.6f} mg/L</b>
        </div>
        """,
        unsafe_allow_html=True
    )

# ---------- Tab 1: RGB Analyzer ----------
with tab1:
    st.header("📷 Average RGB Calculator")

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        # Read image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        avg_rgb = img_rgb.mean(axis=(0, 1)).astype(int)
        st.write("Average RGB value:", avg_rgb.tolist())

        col1, col2 = st.columns(2)
        with col1:
            st.image(img_rgb, caption="Uploaded Image", use_container_width=True)
        with col2:
            patch = np.ones((120, 120, 3), dtype=np.uint8) * avg_rgb
            st.image(patch, caption=f"Avg Color {avg_rgb.tolist()}", use_container_width=False)

