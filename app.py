import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import torch

# Title
st.title("🧠 Waste Detection App with Fallback")
st.markdown("Custom YOLOv8 model (42 classes) + fallback to COCO model (80 classes)")

# Upload image
uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

# Confidence threshold slider
threshold = st.slider("⚙️ Confidence Threshold", 0.0, 1.0, 0.4, 0.05)

# Load both models
@st.cache_resource
def load_models():
    custom_model = YOLO("best.pt")         # Replace with correct path if needed
    coco_model = YOLO("yolov8n.pt")        # Pretrained YOLOv8 with COCO classes
    return custom_model, coco_model

custom_model, coco_model = load_models()

# Handle uploaded image
if uploaded_file is not None:
    # Display uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    # Save to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
        tmp_file.write(uploaded_file.read())
        temp_path = tmp_file.name

    # First: custom model
    st.info("🔍 Detecting with custom model (42 classes)...")
    results_custom = custom_model(temp_path)
    boxes_custom = results_custom[0].boxes
    confidences = boxes_custom.conf if boxes_custom is not None else []

    # Check if any confidence > threshold
    if len(confidences) > 0 and any(conf > threshold for conf in confidences):
        st.success("✅ Detected with custom model!")
        res_img = results_custom[0].plot()
        st.image(res_img, caption="Result (Custom Model)", use_container_width=True)
    else:
        st.warning(f"⚠️ No confident detection > {threshold*100:.0f}%. Trying COCO model...")
        results_coco = coco_model(temp_path)
        boxes_coco = results_coco[0].boxes

        if boxes_coco is not None and len(boxes_coco) > 0:
            st.success("✅ Detected with COCO model!")
            res_img = results_coco[0].plot()
            st.image(res_img, caption="Result (COCO Model)", use_container_width=True)
        else:
            st.error("❌ No objects detected in either model.")
