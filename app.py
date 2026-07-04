import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2

# -----------------------------------------------
# Configuration
# -----------------------------------------------
MODEL_PATH = r"defectdetector.pt"

try:
    model = YOLO(MODEL_PATH)
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# -----------------------------------------------
# Page Setup
# -----------------------------------------------
st.set_page_config(page_title="Steel Defect AI", layout="wide")

st.title("AI Quality Control Dashboard")
st.markdown("Real-time Surface Defect Detection for Manufacturing")

# -----------------------------------------------
# Sidebar Controls
# -----------------------------------------------
st.sidebar.header("Control Panel")
confidence = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05)
st.sidebar.info(f"Model: YOLOv8 Nano | 13ms Real-Time")

uploaded_file = st.sidebar.file_uploader(
    "Upload Steel Image", type=["jpg", "png", "bmp", "jpeg"]
)

# -----------------------------------------------
# Inference & Display
# -----------------------------------------------
if uploaded_file is not None:
    # 1. Read Image
    image = Image.open(uploaded_file)
    # Convert to RGB
    if image.mode != "RGB":
        image = image.convert("RGB")

    # 2. Run Inference
    results = model.predict(image, conf=confidence)

    # 3. Process Results
    result = results[0]
    res_plotted = result.plot()
    res_image = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)

    # 4. Display
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Feed")
        st.image(image, use_container_width=True)
    with col2:
        st.subheader("AI Detection Output")
        st.image(res_image, use_container_width=True)

    # 5. Metrics
    st.divider()
    boxes = result.boxes
    if len(boxes) > 0:
        defect_counts = {}
        for box in boxes:
            cls_id = int(box.cls[0])
            cls_name = result.names[cls_id]
            defect_counts[cls_name] = defect_counts.get(cls_name, 0) + 1

        c1, c2 = st.columns(2)
        c1.metric("Total Defects", len(boxes), delta="Action Required", delta_color="inverse")
        with c2:
            st.write("Identified Issues:")
            for name, count in defect_counts.items():
                st.warning(f"{name}: {count} detected")
    else:
        st.success("No Defects Detected. Material Approved.")

else:
    st.info("Upload an image from datasets/NEU-DET-YOLO/images/val to test the system.")
