import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import altair as alt
import datetime
import json
import cv2
from io import BytesIO
import base64
import plotly.express as px
import plotly.graph_objects as go
from geopy.geocoders import Nominatim
import folium
from streamlit_folium import st_folium
import qrcode

st.set_page_config(
    page_title="Smart Waste Classification System",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Initialize session state variables if they don't exist
if 'detection_history' not in st.session_state:
    st.session_state.detection_history = []
if 'total_items_detected' not in st.session_state:
    st.session_state.total_items_detected = 0
if 'environmental_impact' not in st.session_state:
    st.session_state.environmental_impact = {
        'carbon_saved': 0,  # in kg
        'water_saved': 0,   # in liters
        'landfill_reduced': 0  # in kg
    }
if 'user_location' not in st.session_state:
    st.session_state.user_location = None
if 'challenge_points' not in st.session_state:
    st.session_state.challenge_points = 0
if 'badges_earned' not in st.session_state:
    st.session_state.badges_earned = []
if 'theme' not in st.session_state:
    st.session_state.theme = 'light'

# Load historical data if it exists
def load_data():
    try:
        with open('detection_history.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {'history': [], 'stats': {
            'total_items': 0,
            'carbon_saved': 0,
            'water_saved': 0,
            'landfill_reduced': 0
        }}

# Save data to disk
def save_data(data):
    with open('detection_history.json', 'w') as f:
        json.dump(data, f)

# Environmental impact calculations
def calculate_environmental_impact(waste_type, count=1):
    impact = {
        'carbon_saved': 0,
        'water_saved': 0,
        'landfill_reduced': 0
    }
    
    # Average values based on research (customize these with more accurate data)
    impacts = {
        'plastic': {'carbon': 0.5, 'water': 100, 'landfill': 0.03},
        'paper': {'carbon': 0.3, 'water': 50, 'landfill': 0.1},
        'glass': {'carbon': 0.4, 'water': 25, 'landfill': 0.4},
        'metal': {'carbon': 1.0, 'water': 40, 'landfill': 0.1},
        'organic': {'carbon': 0.2, 'water': 10, 'landfill': 0.5},
        'electronic': {'carbon': 20, 'water': 300, 'landfill': 0.5},
    }
    
    # Match waste type to category
    for category, values in impacts.items():
        if category in waste_type.lower():
            impact['carbon_saved'] = values['carbon'] * count
            impact['water_saved'] = values['water'] * count
            impact['landfill_reduced'] = values['landfill'] * count
            break
    
    return impact

# Generate recycling QR code
def generate_qr_code(data):
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=4,
    )
    qr.add_data(data)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    
    # Convert to bytes for display
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

# Waste sorting guide with detailed information
waste_guide = {
    'plastic': {
        'disposal': 'Recycle in blue bin',
        'preparation': 'Rinse, remove caps, flatten if possible',
        'facts': 'Takes up to 1000 years to decompose naturally',
        'alternatives': 'Reusable bottles, glass containers, biodegradable packaging',
        'image': 'https://via.placeholder.com/100?text=Plastic'
    },
    'paper': {
        'disposal': 'Recycle in green bin',
        'preparation': 'Remove staples, paperclips, and plastic covering',
        'facts': 'Can be recycled 5-7 times before fibers become too short',
        'alternatives': 'Digital documentation, reusable cloths',
        'image': 'https://via.placeholder.com/100?text=Paper'
    },
    'glass': {
        'disposal': 'Recycle in glass collection bins',
        'preparation': 'Rinse, remove lids, sort by color if required',
        'facts': 'Can be recycled indefinitely without loss of quality',
        'alternatives': 'Reusable glass containers',
        'image': 'https://via.placeholder.com/100?text=Glass'
    },
    'metal': {
        'disposal': 'Recycle in yellow bin',
        'preparation': 'Rinse, remove labels if possible',
        'facts': 'Aluminum recycling saves 95% of energy required for virgin production',
        'alternatives': 'Reusable metal containers',
        'image': 'https://via.placeholder.com/100?text=Metal'
    },
    'organic': {
        'disposal': 'Compost bin or organic waste collection',
        'preparation': 'Remove any non-compostable packaging',
        'facts': 'Composting reduces methane emissions from landfills',
        'alternatives': 'Home composting, waste reduction strategies',
        'image': 'https://via.placeholder.com/100?text=Organic'
    },
    'electronic': {
        'disposal': 'Special e-waste collection centers',
        'preparation': 'Remove batteries, wipe personal data',
        'facts': 'Contains valuable metals that can be recovered',
        'alternatives': 'Repair, donate, or sell working electronics',
        'image': 'https://via.placeholder.com/100?text=Electronic'
    },
    'hazardous': {
        'disposal': 'Special hazardous waste facilities',
        'preparation': 'Keep in original container if possible',
        'facts': 'Can contaminate water sources if disposed improperly',
        'alternatives': 'Use eco-friendly cleaning products',
        'image': 'https://via.placeholder.com/100?text=Hazardous'
    }
}

# Nearest recycling centers database (example - you would expand this)
recycling_centers = [
    {"name": "City Recycling Center", "lat": 40.7128, "lon": -74.0060, "accepts": ["plastic", "paper", "glass", "metal"]},
    {"name": "GreenTech Recycling", "lat": 40.7309, "lon": -73.9872, "accepts": ["electronic", "hazardous"]},
    {"name": "Community Compost Hub", "lat": 40.7489, "lon": -73.9680, "accepts": ["organic"]},
    {"name": "AllWaste Recycling", "lat": 40.7589, "lon": -73.9851, "accepts": ["plastic", "paper", "glass", "metal", "electronic"]}
]

# Define waste categories and their associated badge thresholds
badges = {
    "plastic": {"name": "Plastic Preventer", "threshold": 10, "icon": "🥤"},
    "paper": {"name": "Paper Protector", "threshold": 15, "icon": "📄"},
    "glass": {"name": "Glass Guardian", "threshold": 8, "icon": "🍾"},
    "metal": {"name": "Metal Master", "threshold": 12, "icon": "🥫"},
    "organic": {"name": "Compost Champion", "threshold": 20, "icon": "🍎"},
    "overall": {"name": "Waste Warrior", "threshold": 50, "icon": "🦸"}
}

# Page configuration with theme support
def apply_custom_theme():
    # Toggle themes
    if st.session_state.theme == 'dark':
        theme_bg = "#121212"
        theme_text = "#FFFFFF"
        theme_secondary = "#262730"
    else:
        theme_bg = "#FFFFFF"
        theme_text = "#31333F"
        theme_secondary = "#F0F2F6"
    
    st.markdown(f"""
    <style>
        .main-header {{
            font-size: 2.5rem;
            color: {theme_text};
            text-align: center;
            margin-bottom: 1rem;
            font-weight: bold;
        }}
        .sub-header {{
            font-size: 1.5rem;
            color: {theme_text};
            margin-bottom: 2rem;
            text-align: center;
            opacity: 0.8;
        }}
        .success-box {{
            background-color: #d4edda;
            color: #155724;
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 20px;
            border-left: 5px solid #155724;
        }}
        .warning-box {{
            background-color: #fff3cd;
            color: #856404;
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 20px;
            border-left: 5px solid #856404;
        }}
        .info-box {{
            background-color: #d1ecf1;
            color: #0c5460;
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 20px;
            border-left: 5px solid #0c5460;
        }}
        .error-box {{
            background-color: #f8d7da;
            color: #721c24;
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 20px;
            border-left: 5px solid #721c24;
        }}
        .stButton>button {{
            width: 100%;
            background-color: #4CAF50;
            color: white;
            border: none;
            padding: 12px 16px;
            border-radius: 8px;
            font-weight: bold;
            transition: all 0.3s;
        }}
        .stButton>button:hover {{
            background-color: #45a049;
            transform: translateY(-2px);
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .upload-section {{
            border: 2px dashed #4CAF50;
            padding: 30px;
            border-radius: 15px;
            text-align: center;
            transition: all 0.3s;
        }}
        .upload-section:hover {{
            border-color: #45a049;
            background-color: rgba(76, 175, 80, 0.05);
        }}
        .results-container {{
            margin-top: 30px;
            padding: 25px;
            border-radius: 15px;
            background-color: {theme_secondary};
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        }}
        .footer {{
            text-align: center;
            padding: 20px;
            color: {theme_text};
            opacity: 0.6;
            font-size: 0.9rem;
            margin-top: 40px;
        }}
        .metric-box {{
            background-color: {theme_secondary};
            padding: 20px;
            border-radius: 15px;
            text-align: center;
            transition: all 0.3s;
            border-left: 5px solid #4CAF50;
        }}
        .metric-box:hover {{
            transform: translateY(-5px);
            box-shadow: 0 8px 15px rgba(0,0,0,0.1);
        }}
        .badge {{
            display: inline-block;
            padding: 10px 15px;
            background-color: #4CAF50;
            color: white;
            border-radius: 30px;
            margin: 5px;
            font-weight: bold;
        }}
        .waste-card {{
            background-color: {theme_secondary};
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 15px;
            border-left: 4px solid #4CAF50;
        }}
        .challenge-card {{
            background-color: #e3f2fd;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 15px;
            border: 1px solid #90caf9;
        }}
        .tabs-container {{
            border-bottom: 1px solid #e0e0e0;
            padding-bottom: 10px;
            margin-bottom: 20px;
        }}
        .tab {{
            display: inline-block;
            padding: 10px 15px;
            cursor: pointer;
            border-bottom: 2px solid transparent;
        }}
        .tab.active {{
            border-bottom: 2px solid #4CAF50;
            font-weight: bold;
        }}
        .stProgress > div > div > div > div {{
            background-color: #4CAF50;
        }}
    </style>
    """, unsafe_allow_html=True)

# Main application
def main():
    apply_custom_theme()

    
    # Sidebar
    with st.sidebar:
        st.image("https://via.placeholder.com/150?text=EcoScan", width=150)
        st.markdown("## Settings")
        
        # Model selection
        model_option = st.radio(
            "Select Model Behavior",
            ["Auto (Use custom first, fallback to COCO)", "Custom Model Only", "COCO Model Only"]
        )
        
        # Confidence threshold slider
        threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.4, 0.05)
        
        # UI options
        show_labels = st.checkbox("Show Labels", value=True)
        show_conf = st.checkbox("Show Confidence Scores", value=True)
        
        # Advanced options section
        with st.expander("Advanced Options"):
            processing_res = st.select_slider(
                "Processing Resolution",
                options=["Low (Faster)", "Medium", "High (More Accurate)"],
                value="Medium"
            )
            
            apply_enhancement = st.checkbox("Apply Image Enhancement", value=False)
            
            detect_mode = st.radio(
                "Detection Mode",
                ["Standard", "High Precision (Slower)", "Fast (Less Accurate)"]
            )
            
            theme_choice = st.radio("Theme", ["Light", "Dark"])
            if theme_choice.lower() != st.session_state.theme:
                st.session_state.theme = theme_choice.lower()
                st.session_state['force_rerun'] = not st.session_state.get('force_rerun', False)

                
        # Get user location for recycling center recommendations
        with st.expander("Set Your Location"):
            location_input = st.text_input("City or Zip Code", "", key="location_input_map")
            if st.button("Update Location"):
                try:
                    geolocator = Nominatim(user_agent="waste_classification_app")
                    location = geolocator.geocode(location_input)
                    if location:
                        st.session_state.user_location = {
                            "lat": location.latitude,
                            "lon": location.longitude,
                            "address": location.address
                        }
                        st.success(f"Location set to: {location.address}")
                    else:
                        st.error("Location not found. Please try again.")
                except Exception as e:
                    st.error(f"Error setting location: {e}")
                    
        st.markdown("---")
        st.markdown("### About")
        st.info("""
        This advanced waste classification system helps you:
        
        🔍 Identify waste materials accurately
        ♻️ Learn proper recycling methods
        📊 Track your environmental impact
        🏆 Earn badges through eco-challenges
        🗺️ Find nearby recycling centers
        """)
    
    # Main content
    st.markdown('<h1 class="main-header">♻️ Smart Waste Classification & Sustainability Hub</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Upload an image to identify waste materials and get personalized recycling guidance</p>', unsafe_allow_html=True)
    
    # Load models
    @st.cache_resource
    def load_models():
        with st.spinner("Loading AI models... This might take a moment."):
            try:
                custom_model = YOLO("best.pt")
                coco_model = YOLO("yolov8n.pt")
                return custom_model, coco_model, True
            except Exception as e:
                st.error(f"Error loading models: {e}")
                return None, None, False
    
    custom_model, coco_model, models_loaded = load_models()
    
    if not models_loaded:
        st.error("Failed to load models. Please check your installation and try again.")
        return
    
    # Create tabs for main functionality
    tabs = ["📷 Image Analysis", "📊 Analytics Dashboard", "🗺️ Recycling Map", "🏆 Eco Challenges", "📚 Education Center"]
    
    # Display tabs
    st.markdown('<div class="tabs-container">', unsafe_allow_html=True)
    cols = st.columns(len(tabs))
    selected_tab = st.session_state.get('selected_tab', tabs[0])
    
    for i, tab in enumerate(tabs):
        with cols[i]:
            if st.button(tab, key=f"tab_{i}", help=f"Switch to {tab} tab"):
                selected_tab = tab
                st.session_state.selected_tab = selected_tab
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Tab 1: Image Analysis
    if selected_tab == "📷 Image Analysis":
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown('<div class="upload-section">', unsafe_allow_html=True)
            uploaded_file = st.file_uploader("Drop your waste image here or click to browse", type=["jpg", "jpeg", "png"])
            
            camera_option = st.checkbox("Or use your camera")
            if camera_option:
                camera_input = st.camera_input("Take a picture of waste")
                if camera_input is not None:
                    uploaded_file = camera_input
            
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col2:
            st.markdown('<div class="info-box">', unsafe_allow_html=True)
            st.markdown("""
            ### Tips for Best Results
            
            1. **Good lighting**: Ensure waste items are well-lit
            2. **Clear view**: Place items against a contrasting background
            3. **Multiple items**: Space them slightly apart
            4. **Angle**: Capture from above for best recognition
            5. **Focus**: Ensure the image is not blurry
            """)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Process the image
        if uploaded_file is not None:
            # Create tabs for results view
            analysis_tabs = st.tabs(["📸 Image Results", "♻️ Recycling Guide", "🌍 Environmental Impact"])
            
            # Display uploaded image
            with analysis_tabs[0]:
                st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
            
            # Save to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
                tmp_file.write(uploaded_file.read())
                temp_path = tmp_file.name
            
            # Apply image enhancement if selected
            if apply_enhancement:
                try:
                    img = cv2.imread(temp_path)
                    # Basic enhancement
                    img = cv2.detailEnhance(img, sigma_s=10, sigma_r=0.15)
                    # Improve contrast
                    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
                    l, a, b = cv2.split(lab)
                    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                    cl = clahe.apply(l)
                    enhanced_lab = cv2.merge((cl, a, b))
                    enhanced_img = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
                    cv2.imwrite(temp_path, enhanced_img)
                    
                    with analysis_tabs[0]:
                        st.markdown('<div class="info-box">', unsafe_allow_html=True)
                        st.markdown("✅ Image enhancement applied to improve detection accuracy")
                        st.markdown('</div>', unsafe_allow_html=True)
                except Exception as e:
                    with analysis_tabs[0]:
                        st.warning(f"Image enhancement failed: {e}")
            
            with analysis_tabs[0]:
                st.markdown('<div class="results-container">', unsafe_allow_html=True)
                
                # Display processing status
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Adjust model parameters based on detection mode
                iou_threshold = 0.45  # Default
                if detect_mode == "High Precision (Slower)":
                    iou_threshold = 0.65
                elif detect_mode == "Fast (Less Accurate)":
                    iou_threshold = 0.25
                
                # Adjust image size based on resolution setting
                img_size = 640  # Default (medium)
                if processing_res == "Low (Faster)":
                    img_size = 320
                elif processing_res == "High (More Accurate)":
                    img_size = 1280
                
                # Process with selected model option
                if model_option == "Custom Model Only" or model_option == "Auto (Use custom first, fallback to COCO)":
                    status_text.text("Processing with specialized waste detection model...")
                    for i in range(50):
                        time.sleep(0.01)
                        progress_bar.progress(i)
                    
                    # Run detection with optimized parameters
                    results_custom = custom_model(temp_path, conf=threshold, iou=iou_threshold, imgsz=img_size)
                    boxes_custom = results_custom[0].boxes
                    custom_detections = len(boxes_custom)
                    
                    for i in range(50, 100):
                        time.sleep(0.01)
                        progress_bar.progress(i)
                    
                    if custom_detections > 0 and any(conf > threshold for conf in boxes_custom.conf):
                        st.markdown('<div class="success-box">', unsafe_allow_html=True)
                        st.markdown(f"✅ **Success!** Detected {custom_detections} waste items with specialized model")
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Show metrics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.markdown('<div class="metric-box">', unsafe_allow_html=True)
                            st.metric("Items Detected", custom_detections)
                            st.markdown('</div>', unsafe_allow_html=True)
                        with col2:
                            avg_conf = sum(boxes_custom.conf) / len(boxes_custom.conf) if len(boxes_custom.conf) > 0 else 0
                            st.markdown('<div class="metric-box">', unsafe_allow_html=True)
                            st.metric("Avg. Confidence", f"{avg_conf:.2%}")
                            st.markdown('</div>', unsafe_allow_html=True)
                        with col3:
                            max_conf = max(boxes_custom.conf) if len(boxes_custom.conf) > 0 else 0
                            st.markdown('<div class="metric-box">', unsafe_allow_html=True)
                            st.metric("Max Confidence", f"{max_conf:.2%}")
                            st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Display results
                        res_img = results_custom[0].plot(conf=show_conf, labels=show_labels)
                        st.image(res_img, caption="Waste Detection Results", use_container_width=True)
                        
                        # Display classes detected
                        detected_classes = {}
                        for i in range(len(boxes_custom)):
                            cls_id = int(boxes_custom.cls[i])
                            cls_name = results_custom[0].names[cls_id]
                            conf = float(boxes_custom.conf[i])
                            
                            if cls_name in detected_classes:
                                detected_classes[cls_name]["count"] += 1
                                detected_classes[cls_name]["confidences"].append(conf)
                            else:
                                detected_classes[cls_name] = {
                                    "count": 1,
                                    "confidences": [conf]
                                }
                        
                        st.subheader("Detected Waste Categories:")
                        for cls_name, data in detected_classes.items():
                            avg_conf = sum(data["confidences"]) / len(data["confidences"])
                            st.markdown(f"- **{cls_name}**: {data['count']} items (avg. confidence: {avg_conf:.2%})")
                            
                        # Record detection in history
                        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        for cls_name, data in detected_classes.items():
                            # Calculate environmental impact
                            impact = calculate_environmental_impact(cls_name, data["count"])
                            
                            # Update global counters
                            st.session_state.total_items_detected += data["count"]
                            st.session_state.environmental_impact["carbon_saved"] += impact["carbon_saved"]
                            st.session_state.environmental_impact["water_saved"] += impact["water_saved"]
                            st.session_state.environmental_impact["landfill_reduced"] += impact["landfill_reduced"]
                            
                            # Add to history
                            st.session_state.detection_history.append({
                                "timestamp": timestamp,
                                "waste_type": cls_name,
                                "count": data["count"],
                                "confidence": avg_conf,
                                "environmental_impact": impact
                            })
                            
                            # Check for badges
                            waste_category = next((cat for cat in waste_guide.keys() if cat in cls_name.lower()), None)
                            if waste_category and waste_category in badges:
                                # Count total items of this type detected
                                type_count = sum(item["count"] for item in st.session_state.detection_history 
                                                if waste_category in item["waste_type"].lower())
                                
                                # Check if threshold reached for badge
                                if type_count >= badges[waste_category]["threshold"]:
                                    badge_name = badges[waste_category]["name"]
                                    if badge_name not in st.session_state.badges_earned:
                                        st.session_state.badges_earned.append(badge_name)
                                        st.balloons()
                                        st.success(f"🏆 New Badge Earned: {badges[waste_category]['icon']} {badge_name}!")
                            
                            # Check for overall badge
                            if st.session_state.total_items_detected >= badges["overall"]["threshold"]:
                                if badges["overall"]["name"] not in st.session_state.badges_earned:
                                    st.session_state.badges_earned.append(badges["overall"]["name"])
                                    st.balloons()
                                    st.success(f"🏆 Achievement Unlocked: {badges['overall']['icon']} {badges['overall']['name']}!")
                            
                            # Award challenge points
                            st.session_state.challenge_points += data["count"]
                        
                        # Save historical data
                        historical_data = load_data()
                        for item in st.session_state.detection_history[-len(detected_classes):]:
                            historical_data["history"].append(item)
                        
                        historical_data["stats"]["total_items"] = st.session_state.total_items_detected
                        historical_data["stats"]["carbon_saved"] = st.session_state.environmental_impact["carbon_saved"]
                        historical_data["stats"]["water_saved"] = st.session_state.environmental_impact["water_saved"]
                        historical_data["stats"]["landfill_reduced"] = st.session_state.environmental_impact["landfill_reduced"]
                        
                        save_data(historical_data)
                        
                    elif model_option == "Auto (Use custom first, fallback to COCO)":
                        st.markdown('<div class="warning-box">', unsafe_allow_html=True)
                        st.markdown(f"⚠️ No confident waste detections found. Trying general object detection...")
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Use COCO model as fallback
                        results_coco = coco_model(temp_path, conf=threshold, iou=iou_threshold, imgsz=img_size)
                        boxes_coco = results_coco[0].boxes
                        
                        if len(boxes_coco) > 0:
                            st.markdown('<div class="success-box">', unsafe_allow_html=True)
                            st.markdown(f"✅ **Success!** Detected {len(boxes_coco)} objects with general detection model")
                            st.markdown('</div>', unsafe_allow_html=True)
                            
                            # Show results
                            res_img = results_coco[0].plot(conf=show_conf, labels=show_labels)
                            st.image(res_img, caption="General Object Detection Results", use_container_width=True)
                            
                            # Display classes detected
                            detected_classes = {}
                            for i in range(len(boxes_coco)):
                                cls_id = int(boxes_coco.cls[i])
                                cls_name = results_coco[0].names[cls_id]
                                conf = float(boxes_coco.conf[i])
                                
                                if cls_name in detected_classes:
                                    detected_classes[cls_name]["count"] += 1
                                    detected_classes[cls_name]["confidences"].append(conf)
                                else:
                                    detected_classes[cls_name] = {
                                        "count": 1,
                                        "confidences": [conf]
                                    }
                            
                            st.subheader("Detected Objects:")
                            for cls_name, data in detected_classes.items():
                                avg_conf = sum(data["confidences"]) / len(data["confidences"])
                                st.markdown(f"- **{cls_name}**: {data['count']} items (avg. confidence: {avg_conf:.2%})")
                            
                            # Map COCO classes to waste categories when possible
                            waste_mapping = {
                                "bottle": "plastic",
                                "cup": "plastic",
                                "wine glass": "glass",
                                "fork": "metal",
                                "knife": "metal",
                                "spoon": "metal",
                                "bowl": "glass",
                                "banana": "organic",
                                "apple": "organic",
                                "sandwich": "organic",
                                "orange": "organic",
                                "broccoli": "organic",
                                "carrot": "organic",
                                "hot dog": "organic",
                                "pizza": "organic",
                                "donut": "organic",
                                "cake": "organic",
                                "chair": "furniture",
                                "couch": "furniture",
                                "potted plant": "organic",
                                "bed": "furniture",
                                "dining table": "furniture",
                                "toilet": "non-recyclable",
                                "tv": "electronic",
                                "laptop": "electronic",
                                "mouse": "electronic",
                                "remote": "electronic",
                                "keyboard": "electronic",
                                "cell phone": "electronic",
                                "book": "paper",
                                "clock": "electronic",
                                "vase": "glass",
                                "scissors": "metal",
                                "teddy bear": "fabric",
                                "hair drier": "electronic",
                                "toothbrush": "plastic",
                            }
                            
                            st.markdown('<div class="info-box">', unsafe_allow_html=True)
                            st.markdown("""
                            **Note**: General object detection is providing educated guesses about waste categories. 
                            For more accurate waste classification, consider taking a clearer photo of the waste items.
                            """)
                            st.markdown('</div>', unsafe_allow_html=True)
                            
                            # Map detected objects to waste categories
                            waste_categories = {}
                            for cls_name, data in detected_classes.items():
                                waste_type = waste_mapping.get(cls_name.lower(), "unknown")
                                if waste_type in waste_categories:
                                    waste_categories[waste_type] += data["count"]
                                else:
                                    waste_categories[waste_type] = data["count"]
                            
                            if "unknown" not in waste_categories or len(waste_categories) > 1:
                                st.subheader("Suggested Waste Categories:")
                                for waste_type, count in waste_categories.items():
                                    if waste_type != "unknown":
                                        st.markdown(f"- **{waste_type.title()}**: {count} items")
                                        
                                        # Update history and stats
                                        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                        impact = calculate_environmental_impact(waste_type, count)
                                        
                                        st.session_state.total_items_detected += count
                                        st.session_state.environmental_impact["carbon_saved"] += impact["carbon_saved"]
                                        st.session_state.environmental_impact["water_saved"] += impact["water_saved"]
                                        st.session_state.environmental_impact["landfill_reduced"] += impact["landfill_reduced"]
                                        
                                        st.session_state.detection_history.append({
                                            "timestamp": timestamp,
                                            "waste_type": waste_type,
                                            "count": count,
                                            "confidence": 0.7,  # Estimated confidence for mapped categories
                                            "environmental_impact": impact
                                        })
                                
                                # Save historical data
                                historical_data = load_data()
                                for item in st.session_state.detection_history[-len([k for k in waste_categories if k != "unknown"]):]:
                                    historical_data["history"].append(item)
                                
                                historical_data["stats"]["total_items"] = st.session_state.total_items_detected
                                historical_data["stats"]["carbon_saved"] = st.session_state.environmental_impact["carbon_saved"]
                                historical_data["stats"]["water_saved"] = st.session_state.environmental_impact["water_saved"]
                                historical_data["stats"]["landfill_reduced"] = st.session_state.environmental_impact["landfill_reduced"]
                                
                                save_data(historical_data)
                        else:
                            st.error("❌ No objects detected with either model.")
                    else:
                        st.warning(f"⚠️ No waste items detected with confidence above the threshold ({threshold:.0%}).")
                
                elif model_option == "COCO Model Only":
                    status_text.text("Processing with general object detection model...")
                    for i in range(100):
                        time.sleep(0.01)
                        progress_bar.progress(i)
                    
                    results_coco = coco_model(temp_path, conf=threshold, iou=iou_threshold, imgsz=img_size)
                    boxes_coco = results_coco[0].boxes
                    
                    if len(boxes_coco) > 0:
                        st.markdown('<div class="success-box">', unsafe_allow_html=True)
                        st.markdown(f"✅ **Success!** Detected {len(boxes_coco)} objects with general model")
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Show results
                        res_img = results_coco[0].plot(conf=show_conf, labels=show_labels)
                        st.image(res_img, caption="General Object Detection Results", use_container_width=True)
                        
                        # Same code as above for COCO model handling
                        detected_classes = {}
                        for i in range(len(boxes_coco)):
                            cls_id = int(boxes_coco.cls[i])
                            cls_name = results_coco[0].names[cls_id]
                            conf = float(boxes_coco.conf[i])
                            
                            if cls_name in detected_classes:
                                detected_classes[cls_name]["count"] += 1
                                detected_classes[cls_name]["confidences"].append(conf)
                            else:
                                detected_classes[cls_name] = {
                                    "count": 1,
                                    "confidences": [conf]
                                }
                        
                        st.subheader("Detected Objects:")
                        for cls_name, data in detected_classes.items():
                            avg_conf = sum(data["confidences"]) / len(data["confidences"])
                            st.markdown(f"- **{cls_name}**: {data['count']} items (avg. confidence: {avg_conf:.2%})")
                        
                        # Map COCO classes to waste categories (same mapping as above)
                        waste_mapping = {
                            "bottle": "plastic",
                            "cup": "plastic",
                            "wine glass": "glass",
                            # Same as above, abbreviated for clarity
                        }
                        
                        # Same code as above for mapping and recording stats
                        waste_categories = {}
                        for cls_name, data in detected_classes.items():
                            waste_type = waste_mapping.get(cls_name.lower(), "unknown")
                            if waste_type in waste_categories:
                                waste_categories[waste_type] += data["count"]
                            else:
                                waste_categories[waste_type] = data["count"]
                        
                        if "unknown" not in waste_categories or len(waste_categories) > 1:
                            st.subheader("Suggested Waste Categories:")
                            for waste_type, count in waste_categories.items():
                                if waste_type != "unknown":
                                    st.markdown(f"- **{waste_type.title()}**: {count} items")
                    else:
                        st.error("❌ No objects detected with the general model.")
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Tab 2: Recycling Guide based on detected items
            with analysis_tabs[1]:
                st.markdown('<div class="results-container">', unsafe_allow_html=True)
                st.subheader("Personalized Recycling Guide")
                
                # Check if we have detections
                if 'detected_classes' in locals() and detected_classes:
                    # For custom model detections
                    if model_option != "COCO Model Only" and custom_detections > 0 and any(conf > threshold for conf in boxes_custom.conf):
                        # Create recycling guides for each detected waste type
                        for cls_name in detected_classes.keys():
                            waste_type = None
                            # Map detected class to waste guide category
                            for category in waste_guide.keys():
                                if category in cls_name.lower():
                                    waste_type = category
                                    break
                            
                            if waste_type:
                                st.markdown(f'<div class="waste-card">', unsafe_allow_html=True)
                                col1, col2 = st.columns([1, 3])
                                with col1:
                                    st.image(waste_guide[waste_type]["image"], caption=waste_type.title())
                                with col2:
                                    st.markdown(f"### {waste_type.title()} Recycling Guide")
                                    st.markdown(f"**Disposal Method:** {waste_guide[waste_type]['disposal']}")
                                    st.markdown(f"**Preparation:** {waste_guide[waste_type]['preparation']}")
                                    st.markdown(f"**Did You Know?** {waste_guide[waste_type]['facts']}")
                                    st.markdown(f"**Eco Alternatives:** {waste_guide[waste_type]['alternatives']}")
                                st.markdown('</div>', unsafe_allow_html=True)
                            else:
                                st.markdown(f'<div class="waste-card">', unsafe_allow_html=True)
                                st.markdown(f"### {cls_name}")
                                st.markdown("We don't have specific recycling information for this item. Please check your local waste management guidelines.")
                                st.markdown('</div>', unsafe_allow_html=True)
                    
                    # For COCO model with waste mapping
                    elif 'waste_categories' in locals() and waste_categories:
                        for waste_type, count in waste_categories.items():
                            if waste_type != "unknown" and waste_type in waste_guide:
                                st.markdown(f'<div class="waste-card">', unsafe_allow_html=True)
                                col1, col2 = st.columns([1, 3])
                                with col1:
                                    st.image(waste_guide[waste_type]["image"], caption=waste_type.title())
                                with col2:
                                    st.markdown(f"### {waste_type.title()} Recycling Guide")
                                    st.markdown(f"**Disposal Method:** {waste_guide[waste_type]['disposal']}")
                                    st.markdown(f"**Preparation:** {waste_guide[waste_type]['preparation']}")
                                    st.markdown(f"**Did You Know?** {waste_guide[waste_type]['facts']}")
                                    st.markdown(f"**Eco Alternatives:** {waste_guide[waste_type]['alternatives']}")
                                st.markdown('</div>', unsafe_allow_html=True)
                    else:
                        st.info("No recyclable items detected or confident waste classifications available.")
                else:
                    st.info("Process an image first to get personalized recycling recommendations.")
                
                # Add downloadable recycling guide
                if 'detected_classes' in locals() and detected_classes:
                    st.markdown("### Download Complete Recycling Guide")
                    
                    # Generate waste information for PDF
                    waste_info = ""
                    if model_option != "COCO Model Only" and custom_detections > 0:
                        for cls_name in detected_classes.keys():
                            waste_info += f"- {cls_name}: {detected_classes[cls_name]['count']} items\n"
                    elif 'waste_categories' in locals() and waste_categories:
                        for waste_type, count in waste_categories.items():
                            if waste_type != "unknown":
                                waste_info += f"- {waste_type.title()}: {count} items\n"
                    
                    if waste_info:
                        # Generate QR code with recycling information
                        qr_data = f"""
                        Waste Classification Results:
                        Date: {datetime.datetime.now().strftime('%Y-%m-%d')}
                        Items Detected:
                        {waste_info}
                        
                        For more information, visit your local recycling center.
                        """
                        
                        qr_code_data = generate_qr_code(qr_data)
                        st.markdown(f"""
                        <div style="text-align: center; margin: 20px 0;">
                            <img src="data:image/png;base64,{qr_code_data}" width="200">
                            <p>Scan this QR code to save your recycling information</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Download button for comprehensive guide
                        st.download_button(
                            label="📄 Download PDF Recycling Guide",
                            data=qr_data,
                            file_name="recycling_guide.txt",
                            mime="text/plain",
                        )
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Tab 3: Environmental Impact
            with analysis_tabs[2]:
                st.markdown('<div class="results-container">', unsafe_allow_html=True)
                st.subheader("Your Environmental Impact")
                
                if 'detected_classes' in locals() and detected_classes:
                    # Calculate total impact from this analysis
                    current_impact = {
                        'carbon_saved': 0,
                        'water_saved': 0,
                        'landfill_reduced': 0
                    }
                    
                    # Get most recent entries from detection history
                    recent_items = st.session_state.detection_history[-len(detected_classes):] if detected_classes else []
                    
                    for item in recent_items:
                        current_impact['carbon_saved'] += item['environmental_impact']['carbon_saved']
                        current_impact['water_saved'] += item['environmental_impact']['water_saved']
                        current_impact['landfill_reduced'] += item['environmental_impact']['landfill_reduced']
                    
                    # Display current impact metrics
                    st.markdown("### Impact from this Analysis")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.markdown('<div class="metric-box">', unsafe_allow_html=True)
                        st.metric("CO₂ Emissions Saved", f"{current_impact['carbon_saved']:.2f} kg")
                        st.markdown('</div>', unsafe_allow_html=True)
                    with col2:
                        st.markdown('<div class="metric-box">', unsafe_allow_html=True)
                        st.metric("Water Saved", f"{current_impact['water_saved']:.1f} L")
                        st.markdown('</div>', unsafe_allow_html=True)
                    with col3:
                        st.markdown('<div class="metric-box">', unsafe_allow_html=True)
                        st.metric("Landfill Waste Reduced", f"{current_impact['landfill_reduced']:.2f} kg")
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Visual representation of impact
                    st.markdown("### Visualized Impact")
                    
                    # Create equivalent metrics
                    car_km = current_impact['carbon_saved'] * 6  # ~ 6 km per kg CO2
                    trees_day = current_impact['carbon_saved'] / 0.022  # ~ 22g CO2 per tree per day
                    shower_minutes = current_impact['water_saved'] / 10  # ~ 10L per minute of showering
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.markdown(f"""
                        ### 🚗 Car Travel Equivalent
                        Your recycling saves CO₂ equivalent to **{car_km:.1f} km** of car travel.
                        """)
                    with col2:
                        st.markdown(f"""
                        ### 🌳 Tree Absorption Equivalent
                        Your recycling equals what **{trees_day:.1f} trees** absorb in a day.
                        """)
                    with col3:
                        st.markdown(f"""
                        ### 🚿 Water Usage Equivalent
                        Your recycling saves water equivalent to **{shower_minutes:.1f} minutes** of showering.
                        """)
                    
                    # Cumulative impact
                    st.markdown("### Your Total Environmental Impact")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.markdown('<div class="metric-box">', unsafe_allow_html=True)
                        st.metric("Total CO₂ Saved", f"{st.session_state.environmental_impact['carbon_saved']:.2f} kg")
                        st.markdown('</div>', unsafe_allow_html=True)
                    with col2:
                        st.markdown('<div class="metric-box">', unsafe_allow_html=True)
                        st.metric("Total Water Saved", f"{st.session_state.environmental_impact['water_saved']:.1f} L")
                        st.markdown('</div>', unsafe_allow_html=True)
                    with col3:
                        st.markdown('<div class="metric-box">', unsafe_allow_html=True)
                        st.metric("Total Landfill Reduced", f"{st.session_state.environmental_impact['landfill_reduced']:.2f} kg")
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Add badges earned
                    if st.session_state.badges_earned:
                        st.markdown("### 🏆 Your Eco Badges")
                        badge_html = ""
                        for badge in st.session_state.badges_earned:
                            badge_icon = next((b['icon'] for k, b in badges.items() if b['name'] == badge), "🔰")
                            badge_html += f'<span class="badge">{badge_icon} {badge}</span> '
                        
                        st.markdown(f'<div style="margin: 20px 0;">{badge_html}</div>', unsafe_allow_html=True)
                else:
                    st.info("Process an image to see your environmental impact.")
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Clean up the temp file
            # os.unlink(temp_path)
    
    # Tab 2: Analytics Dashboard
    elif selected_tab == "📊 Analytics Dashboard":
        st.markdown('<h2 style="text-align: center;">Waste Management Analytics Dashboard</h2>', unsafe_allow_html=True)
        
        # Load historical data
        historical_data = load_data()
        
        if not historical_data["history"]:
            st.info("No data available yet. Process some images to generate analytics.")
        else:
            # Convert history to dataframe for easier analysis
            df = pd.DataFrame(historical_data["history"])
            
            # Metrics overview
            st.markdown("### Key Metrics")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown('<div class="metric-box">', unsafe_allow_html=True)
                st.metric("Total Items Processed", historical_data["stats"]["total_items"])
                st.markdown('</div>', unsafe_allow_html=True)
            with col2:
                st.markdown('<div class="metric-box">', unsafe_allow_html=True)
                st.metric("CO₂ Emissions Saved", f"{historical_data['stats']['carbon_saved']:.2f} kg")
                st.markdown('</div>', unsafe_allow_html=True)
            with col3:
                st.markdown('<div class="metric-box">', unsafe_allow_html=True)
                st.metric("Water Saved", f"{historical_data['stats']['water_saved']:.1f} L")
                st.markdown('</div>', unsafe_allow_html=True)
            with col4:
                st.markdown('<div class="metric-box">', unsafe_allow_html=True)
                st.metric("Landfill Reduced", f"{historical_data['stats']['landfill_reduced']:.2f} kg")
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Time series analysis
            st.markdown("### Recycling Activity Over Time")
            
            # Extract date and convert to datetime
            df['date'] = pd.to_datetime(df['timestamp'])
            df['day'] = df['date'].dt.date
            
            # Group by day and waste type
            daily_counts = df.groupby(['day', 'waste_type'])['count'].sum().reset_index()
            
            # Pivot for better visualization
            pivot_df = daily_counts.pivot_table(index='day', columns='waste_type', values='count', fill_value=0)
            
            # Create a time series chart
            fig = px.line(
                daily_counts, 
                x='day', 
                y='count', 
                color='waste_type',
                title='Daily Recycling Activity',
                labels={'day': 'Date', 'count': 'Items Recycled', 'waste_type': 'Waste Type'}
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Waste composition analysis
            st.markdown("### Waste Composition Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Group by waste type
                waste_composition = df.groupby('waste_type')['count'].sum().reset_index()
                
                # Create pie chart
                fig = px.pie(
                    waste_composition, 
                    values='count', 
                    names='waste_type',
                    title='Overall Waste Composition',
                    hole=0.4,
                    color_discrete_sequence=px.colors.qualitative.Pastel
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Environmental impact by waste type
                impact_by_type = df.groupby('waste_type').apply(
                    lambda x: pd.Series({
                        'carbon_saved': sum(item['environmental_impact']['carbon_saved'] for _, item in x.iterrows()),
                        'water_saved': sum(item['environmental_impact']['water_saved'] for _, item in x.iterrows()),
                        'landfill_reduced': sum(item['environmental_impact']['landfill_reduced'] for _, item in x.iterrows())
                    })
                ).reset_index()
                
                # Create bar chart for carbon impact
                fig = px.bar(
                    impact_by_type,
                    x='waste_type',
                    y='carbon_saved',
                    title='CO₂ Emissions Saved by Waste Type',
                    labels={'waste_type': 'Waste Type', 'carbon_saved': 'CO₂ Saved (kg)'},
                    color='waste_type'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Progress towards sustainability goals
            st.markdown("### Sustainability Goals Progress")
            
            # Example goals
            goals = {
                'carbon': {'target': 50, 'current': historical_data['stats']['carbon_saved']},
                'water': {'target': 1000, 'current': historical_data['stats']['water_saved']},
                'landfill': {'target': 25, 'current': historical_data['stats']['landfill_reduced']}
            }
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                carbon_progress = min(goals['carbon']['current'] / goals['carbon']['target'] * 100, 100)
                st.markdown("#### Carbon Reduction Goal")
                st.progress(carbon_progress / 100)
                st.markdown(f"{carbon_progress:.1f}% of {goals['carbon']['target']} kg CO₂ goal")
            
            with col2:
                water_progress = min(goals['water']['current'] / goals['water']['target'] * 100, 100)
                st.markdown("#### Water Conservation Goal")
                st.progress(water_progress / 100)
                st.markdown(f"{water_progress:.1f}% of {goals['water']['target']} L water goal")
            
            with col3:
                landfill_progress = min(goals['landfill']['current'] / goals['landfill']['target'] * 100, 100)
                st.markdown("#### Landfill Reduction Goal")
                st.progress(landfill_progress / 100)
                st.markdown(f"{landfill_progress:.1f}% of {goals['landfill']['target']} kg landfill goal")
            
            # Download report button
            st.markdown("### Export Analytics")
            
            col1, col2 = st.columns(2)
            with col1:
                # Generate CSV
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📊 Download Data as CSV",
                    data=csv,
                    file_name=f"waste_data_{datetime.datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                )
            
            with col2:
                # Generate summary report
                report = f"""
                # Waste Management Analytics Report
                Generated on: {datetime.datetime.now().strftime('%Y-%m-%d')}
                
                ## Summary Statistics
                - Total Items Processed: {historical_data["stats"]["total_items"]}
                - CO₂ Emissions Saved: {historical_data["stats"]["carbon_saved"]:.2f} kg
                - Water Saved: {historical_data["stats"]["water_saved"]:.1f} L
                - Landfill Waste Reduced: {historical_data["stats"]["landfill_reduced"]:.2f} kg
                
                ## Waste Composition
                {waste_composition.to_markdown()}
                
                ## Environmental Impact by Waste Type
                {impact_by_type.to_markdown()}
                
                ## Progress Towards Goals
                - Carbon Reduction: {carbon_progress:.1f}% of goal
                - Water Conservation: {water_progress:.1f}% of goal
                - Landfill Reduction: {landfill_progress:.1f}% of goal
                """
                
                st.download_button(
                    label="📝 Download Summary Report",
                    data=report,
                    file_name=f"waste_report_{datetime.datetime.now().strftime('%Y%m%d')}.md",
                    mime="text/markdown",
                )
    
    # Tab 3: Recycling Map
    elif selected_tab == "🗺️ Recycling Map":
        st.markdown('<h2 style="text-align: center;">Recycling Center Locator</h2>', unsafe_allow_html=True)
        
        # Check if location is set
        if not st.session_state.user_location:
            st.info("Please set your location in the sidebar to find recycling centers near you.")
            
            # Manual location input
            st.markdown("### Enter Location")
            location_input = st.text_input("City or Zip Code", "")
            if st.button("Find Recycling Centers"):
                try:
                    geolocator = Nominatim(user_agent="waste_classification_app")
                    location = geolocator.geocode(location_input)
                    if location:
                        st.session_state.user_location = {
                            "lat": location.latitude,
                            "lon": location.longitude,
                            "address": location.address
                        }
                        st.success(f"Location set to: {location.address}")
                        st.session_state['force_rerun'] = not st.session_state.get('force_rerun', False)

                    else:
                        st.error("Location not found. Please try again.")
                except Exception as e:
                    st.error(f"Error setting location: {e}")
        else:
            # Display the map
            st.markdown(f"### Recycling Centers near {st.session_state.user_location['address']}")
            
            # Create map centered on user location
            m = folium.Map(
                location=[st.session_state.user_location["lat"], st.session_state.user_location["lon"]],
                zoom_start=13
            )
            
            # Add user marker
            folium.Marker(
                [st.session_state.user_location["lat"], st.session_state.user_location["lon"]],
                popup="Your Location",
                icon=folium.Icon(color="blue", icon="user", prefix="fa")
            ).add_to(m)
            
            # Calculate distances and add recycling centers
            user_lat = st.session_state.user_location["lat"]
            user_lon = st.session_state.user_location["lon"]
            
            # Filter based on waste types if we have detection history
            waste_types = []
            if st.session_state.detection_history:
                for item in st.session_state.detection_history:
                    waste_type = item["waste_type"].lower()
                    for category in waste_guide.keys():
                        if category in waste_type and category not in waste_types:
                            waste_types.append(category)
            
            # Add centers to map
            added_centers = []
            for center in recycling_centers:
                # Simple distance calculation
                distance = ((center["lat"] - user_lat) ** 2 + (center["lon"] - user_lon) ** 2) ** 0.5 * 111  # Rough km conversion
                
                # If we have waste types, filter centers that accept them
                if waste_types:
                    accepts_detected_waste = any(waste_type in center["accepts"] for waste_type in waste_types)
                    if not accepts_detected_waste:
                        continue
                
                # Add marker
                popup_html = f"""
                <div style="width: 200px">
                    <h4>{center['name']}</h4>
                    <p><strong>Distance:</strong> {distance:.2f} km</p>
                    <p><strong>Accepts:</strong> {', '.join(center['accepts'])}</p>
                    <a href="https://maps.google.com/?q={center['lat']},{center['lon']}" target="_blank">
                        Get Directions
                    </a>
                </div>
                """
                
                folium.Marker(
                    [center["lat"], center["lon"]],
                    popup=folium.Popup(popup_html, max_width=300),
                    icon=folium.Icon(color="green", icon="recycle", prefix="fa")
                ).add_to(m)
                
                added_centers.append({
                    "name": center["name"],
                    "distance": distance,
                    "accepts": center["accepts"]
                })
            
            # Display the map
            st_folium(m)
            
            # Add table of centers
            if added_centers:
                st.markdown("### Nearest Recycling Centers")
                
                # Sort by distance
                added_centers.sort(key=lambda x: x["distance"])
                
                # Create a dataframe
                centers_df = pd.DataFrame(added_centers)
                centers_df["distance"] = centers_df["distance"].round(2).astype(str) + " km"
                centers_df["accepts"] = centers_df["accepts"].apply(lambda x: ", ".join(x))
                centers_df.columns = ["Name", "Distance", "Accepts"]
                
                st.dataframe(centers_df)
                
                # Filter options
                st.markdown("### Filter Centers")
                waste_filter = st.multiselect(
                    "Show centers that accept:", 
                    options=list(waste_guide.keys()),
                    default=waste_types if waste_types else None
                )
                
                if waste_filter:
                    filtered_centers = [
                        center for center in recycling_centers
                        if any(waste_type in center["accepts"] for waste_type in waste_filter)
                    ]
                    
                    if filtered_centers:
                        # Create new map with filtered centers
                        fm = folium.Map(
                            location=[st.session_state.user_location["lat"], st.session_state.user_location["lon"]],
                            zoom_start=13
                        )
                        
                        # Add user marker
                        folium.Marker(
                            [st.session_state.user_location["lat"], st.session_state.user_location["lon"]],
                            popup="Your Location",
                            icon=folium.Icon(color="blue", icon="user", prefix="fa")
                        ).add_to(fm)
                        
                        # Add filtered centers
                        for center in filtered_centers:
                            distance = ((center["lat"] - user_lat) ** 2 + (center["lon"] - user_lon) ** 2) ** 0.5 * 111
                            
                            popup_html = f"""
                            <div style="width: 200px">
                                <h4>{center['name']}</h4>
                                <p><strong>Distance:</strong> {distance:.2f} km</p>
                                <p><strong>Accepts:</strong> {', '.join(center['accepts'])}</p>
                                <a href="https://maps.google.com/?q={center['lat']},{center['lon']}" target="_blank">
                                    Get Directions
                                </a>
                            </div>
                            """
                            
                            folium.Marker(
                                [center["lat"], center["lon"]],
                                popup=folium.Popup(popup_html, max_width=300),
                                icon=folium.Icon(color="green", icon="recycle", prefix="fa")
                            ).add_to(fm)
                        
                        st.markdown(f"### Centers accepting {', '.join(waste_filter)}")
                        st_folium(fm)
                    else:
                        st.info("No centers found that accept the selected waste types.")
            else:
                st.info("No recycling centers found in your area that accept your waste types.")
            
            # Reset location
            if st.button("Reset Location"):
                st.session_state.user_location = None
                st.session_state['force_rerun'] = not st.session_state.get('force_rerun', False)

    
    # Tab 4: Eco Challenges
    elif selected_tab == "🏆 Eco Challenges":
        st.markdown('<h2 style="text-align: center;">Eco Challenges & Achievements</h2>', unsafe_allow_html=True)
        
        # Points and badges summary
        st.markdown("### Your Eco Stats")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="metric-box">', unsafe_allow_html=True)
            st.metric("Challenge Points", st.session_state.challenge_points)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-box">', unsafe_allow_html=True)
            st.metric("Badges Earned", len(st.session_state.badges_earned))
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Display badges
        if st.session_state.badges_earned:
            st.markdown("### Your Badges")
            badge_html = ""
            for badge in st.session_state.badges_earned:
                badge_icon = next((b['icon'] for k, b in badges.items() if b['name'] == badge), "🔰")
                badge_html += f'<span class="badge">{badge_icon} {badge}</span> '
            
            st.markdown(f'<div style="margin: 20px 0;">{badge_html}</div>', unsafe_allow_html=True)
        
        # Available challenges
        st.markdown("### Available Challenges")
        
        challenges = [
            {
                "title": "Plastic-Free Week",
                "description": "Avoid single-use plastics for one week",
                "points": 100,
                "icon": "🥤",
                "difficulty": "Medium"
            },
            {
                "title": "Recycling Streak",
                "description": "Recycle 5 days in a row",
                "points": 50,
                "icon": "♻️",
                "difficulty": "Easy"
            },
            {
                "title": "E-Waste Collection",
                "description": "Properly dispose of 3 electronic items",
                "points": 75,
                "icon": "💻",
                "difficulty": "Medium"
            },
            {
                "title": "Compost Master",
                "description": "Start a compost bin and maintain it for 2 weeks",
                "points": 125,
                "icon": "🍎",
                "difficulty": "Hard"
            },
            {
                "title": "Waste Audit",
                "description": "Track and categorize all household waste for 3 days",
                "points": 150,
                "icon": "📋",
                "difficulty": "Hard"
            }
        ]
        
        # Display challenges in cards
        for i, challenge in enumerate(challenges):
            st.markdown(f'<div class="challenge-card">', unsafe_allow_html=True)
            col1, col2, col3 = st.columns([1, 4, 1])
            
            with col1:
                st.markdown(f"<h1 style='text-align: center;'>{challenge['icon']}</h1>", unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"### {challenge['title']}")
                st.markdown(challenge['description'])
                st.markdown(f"**Difficulty:** {challenge['difficulty']}")
            
            with col3:
                st.markdown(f"<h3 style='text-align: center;'>{challenge['points']}</h3>", unsafe_allow_html=True)
                st.markdown("<p style='text-align: center;'>points</p>", unsafe_allow_html=True)
                if st.button("Accept", key=f"challenge_{i}"):
                    st.session_state[f"challenge_{i}_accepted"] = True
            
            # Show progress if challenge accepted
            if st.session_state.get(f"challenge_{i}_accepted", False):
                progress = st.session_state.get(f"challenge_{i}_progress", 0)
                st.progress(progress / 100)
                st.markdown(f"Progress: {progress}%")
                
                # Allow marking progress
                if progress < 100:
                    if st.button("Update Progress", key=f"update_{i}"):
                        new_progress = min(progress + 25, 100)
                        st.session_state[f"challenge_{i}_progress"] = new_progress
                        if new_progress == 100:
                            st.session_state.challenge_points += challenge["points"]
                            st.balloons()
                            st.success(f"🎉 Challenge completed! You earned {challenge['points']} points!")
                        st.session_state['force_rerun'] = not st.session_state.get('force_rerun', False)

                else:
                    st.success("✅ Challenge completed!")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Leaderboard (simulated)
        st.markdown("### Community Leaderboard")
        
        # Simulate some leaderboard data
        leaderboard_data = [
            {"rank": 1, "name": "EcoWarrior22", "points": 2500, "badges": 12},
            {"rank": 2, "name": "GreenThumb", "points": 2200, "badges": 10},
            {"rank": 3, "name": "RecycleKing", "points": 1800, "badges": 8},
            {"rank": 4, "name": "EarthSaver", "points": 1500, "badges": 7},
            {"rank": 5, "name": "You", "points": st.session_state.challenge_points, "badges": len(st.session_state.badges_earned)},
            {"rank": 6, "name": "EcoNewbie", "points": 700, "badges": 3},
            {"rank": 7, "name": "GreenLearner", "points": 500, "badges": 2},
        ]
        
        # Sort by points
        leaderboard_data.sort(key=lambda x: x["points"], reverse=True)
        
        # Update ranks
        for i, entry in enumerate(leaderboard_data):
            entry["rank"] = i + 1
        
        # Create dataframe
        leaderboard_df = pd.DataFrame(leaderboard_data)
        
        # Color your entry
        def highlight_you(s):
            is_you = s == "You"
            return ['background-color: #d4edda' if v else '' for v in is_you]
        
        # Display styled dataframe
        st.dataframe(leaderboard_df.style.apply(highlight_you, subset=['name']))
    
    # Tab 5: Education Center
    elif selected_tab == "📚 Education Center":
        st.markdown('<h2 style="text-align: center;">Waste Management Education Center</h2>', unsafe_allow_html=True)
        
        # Create subtabs for different educational content
        education_tabs = st.tabs(["Waste Guide", "Recycling Facts", "Educational Videos", "Quizzes", "Resources"])
        
        # Tab 1: Waste Guide
        with education_tabs[0]:
            st.markdown("### Comprehensive Waste Guide")
            st.markdown("""
            Understanding how to properly sort and dispose of waste is crucial for effective recycling 
            and minimizing environmental impact. Browse through our comprehensive guide below.
            """)
            
            # Create expandable sections for each waste type
            for waste_type, info in waste_guide.items():
                with st.expander(f"{waste_type.title()} Guide"):
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        st.image(info["image"], caption=waste_type.title())
                    with col2:
                        st.markdown(f"### {waste_type.title()}")
                        st.markdown(f"**Disposal Method:** {info['disposal']}")
                        st.markdown(f"**Preparation:** {info['preparation']}")
                        st.markdown(f"**Did You Know?** {info['facts']}")
                        st.markdown(f"**Eco Alternatives:** {info['alternatives']}")
        
        # Tab 2: Recycling Facts
        with education_tabs[1]:
            st.markdown("### Interesting Recycling Facts")
            
            facts = [
                {
                    "title": "Aluminum Facts",
                    "content": """
                    - Recycling aluminum saves 95% of the energy required to make aluminum from raw materials
                    - An aluminum can can be recycled and back on the shelf in just 60 days
                    - Aluminum can be recycled infinitely without degradation in quality
                    """,
                    "icon": "🥫"
                },
                {
                    "title": "Plastic Facts",
                    "content": """
                    - Only about 9% of all plastic ever produced has been recycled
                    - It takes up to 1,000 years for plastic to decompose in landfills
                    - 8 million metric tons of plastic end up in our oceans annually
                    """,
                    "icon": "🧴"
                },
                {
                    "title": "Paper Facts",
                    "content": """
                    - Recycling one ton of paper saves 17 trees, 7,000 gallons of water, and 463 gallons of oil
                    - The average American uses 680 pounds of paper per year
                    - Paper can be recycled 5-7 times before fibers become too short
                    """,
                    "icon": "📄"
                },
                {
                    "title": "Glass Facts",
                    "content": """
                    - Glass can be recycled endlessly without losing quality or purity
                    - One ton of recycled glass saves 42 kWh of energy and 1.5 cubic yards of landfill space
                    - It takes 4,000 years for a glass bottle to decompose naturally
                    """,
                    "icon": "🍾"
                },
                {
                    "title": "E-Waste Facts",
                    "content": """
                    - Only 12.5% of e-waste is recycled globally
                    - E-waste contains valuable materials like gold, silver, copper, and palladium
                    - Each American generates about 20 kg of e-waste annually
                    """,
                    "icon": "💻"
                }
            ]
            
            # Display facts in a grid
            cols = st.columns(3)
            for i, fact in enumerate(facts):
                with cols[i % 3]:
                    st.markdown(f'<div class="waste-card">', unsafe_allow_html=True)
                    st.markdown(f"## {fact['icon']} {fact['title']}")
                    st.markdown(fact['content'])
                    st.markdown('</div>', unsafe_allow_html=True)
        
        # Tab 3: Educational Videos
        with education_tabs[2]:
            st.markdown("### Educational Videos")
            st.warning("Note: Video embed functionality would be implemented here. Below are placeholders for video content.")
            
            videos = [
                {
                    "title": "Recycling Process Explained",
                    "description": "Learn about what happens to your recycling after it leaves your home.",
                    "thumbnail": "https://via.placeholder.com/640x360?text=Recycling+Process+Video",
                    "duration": "7:22"
                },
                {
                    "title": "Why Plastic Recycling Is So Complex",
                    "description": "Understanding the challenges and complexities of plastic recycling.",
                    "thumbnail": "https://via.placeholder.com/640x360?text=Plastic+Recycling+Video",
                    "duration": "12:08"
                },
                {
                    "title": "Composting 101",
                    "description": "A beginner's guide to starting and maintaining a compost bin.",
                    "thumbnail": "https://via.placeholder.com/640x360?text=Composting+Video",
                    "duration": "8:55"
                },
                {
                    "title": "Zero Waste Living Tips",
                    "description": "Simple ways to reduce waste in your everyday life.",
                    "thumbnail": "https://via.placeholder.com/640x360?text=Zero+Waste+Video",
                    "duration": "15:30"
                }
            ]
            
            # Display video thumbnails
            col1, col2 = st.columns(2)
            
            with col1:
                for i in range(0, len(videos), 2):
                    if i < len(videos):
                        st.image(videos[i]["thumbnail"])
                        st.markdown(f"**{videos[i]['title']}** ({videos[i]['duration']})")
                        st.markdown(videos[i]["description"])
                        st.button("Watch Video", key=f"video_{i}")
                        st.markdown("---")
            
            with col2:
                for i in range(1, len(videos), 2):
                    if i < len(videos):
                        st.image(videos[i]["thumbnail"])
                        st.markdown(f"**{videos[i]['title']}** ({videos[i]['duration']})")
                        st.markdown(videos[i]["description"])
                        st.button("Watch Video", key=f"video_{i}")
                        st.markdown("---")
        
        # Tab 4: Quizzes
        with education_tabs[3]:
            st.markdown("### Test Your Knowledge")
            st.markdown("Take our interactive quizzes to test and improve your waste management knowledge.")
            
            quizzes = [
                "Recycling Basics", 
                "Plastic Types", 
                "Composting Knowledge", 
                "E-Waste Challenge", 
                "Sustainable Living"
            ]
            
            selected_quiz = st.selectbox("Select a quiz:", quizzes)
            
            # Example quiz questions for Recycling Basics
            if selected_quiz == "Recycling Basics":
                st.markdown("## Recycling Basics Quiz")
                st.markdown("Answer the following questions to test your recycling knowledge:")
                
                questions = [
                    {
                        "question": "Which of these items generally CANNOT be recycled in standard programs?",
                        "options": ["Aluminum cans", "Plastic utensils", "Cardboard boxes", "Glass bottles"],
                        "answer": 1,
                        "explanation": "Plastic utensils are typically not recyclable in standard programs due to their small size and the type of plastic used."
                    },
                    {
                        "question": "What should you do with recyclables before placing them in the recycling bin?",
                        "options": ["Rinse them clean", "Break them into small pieces", "Leave food residue to help composting", "Wrap them in plastic bags"],
                        "answer": 0,
                        "explanation": "Rinsing recyclables helps prevent contamination and odors in the recycling stream."
                    },
                    {
                        "question": "Which color glass is most difficult to recycle?",
                        "options": ["Clear", "Green", "Blue", "Brown"],
                        "answer": 2,
                        "explanation": "Blue glass is less common and often more difficult to recycle as it can't be easily mixed with other colors."
                    }
                ]
                
                # Track score
                if "quiz_score" not in st.session_state:
                    st.session_state.quiz_score = 0
                    st.session_state.questions_answered = 0
                
                # Display questions one by one
                if st.session_state.questions_answered < len(questions):
                    current_q = questions[st.session_state.questions_answered]
                    st.markdown(f"### Question {st.session_state.questions_answered + 1}:")
                    st.markdown(current_q["question"])
                    
                    answer = st.radio("Select your answer:", current_q["options"], key=f"q_{st.session_state.questions_answered}")
                    
                    if st.button("Submit Answer"):
                        if current_q["options"].index(answer) == current_q["answer"]:
                            st.success("✅ Correct!")
                            st.session_state.quiz_score += 1
                        else:
                            st.error("❌ Incorrect")
                            st.info(f"Explanation: {current_q['explanation']}")
                        
                        st.session_state.questions_answered += 1
                        st.session_state['force_rerun'] = not st.session_state.get('force_rerun', False)

                else:
                    # Quiz completed
                    percentage = (st.session_state.quiz_score / len(questions)) * 100
                    st.markdown(f"### Quiz Completed!")
                    st.markdown(f"Your score: **{st.session_state.quiz_score}/{len(questions)}** ({percentage:.0f}%)")
                    
                    if percentage >= 80:
                        st.success("🏆 Excellent! You're a recycling expert!")
                        # Award challenge points
                        st.session_state.challenge_points += 20
                        st.markdown("*+20 challenge points awarded!*")
                    elif percentage >= 60:
                        st.info("👍 Good job! You know the basics but there's room for improvement.")
                        st.session_state.challenge_points += 10
                        st.markdown("*+10 challenge points awarded!*")
                    else:
                        st.warning("📚 Looks like you could use some more recycling knowledge. Check out our educational resources!")
                        st.session_state.challenge_points += 5
                        st.markdown("*+5 challenge points awarded for participation!*")
                    
                    if st.button("Restart Quiz"):
                        st.session_state.quiz_score = 0
                        st.session_state.questions_answered = 0
                        st.session_state['force_rerun'] = not st.session_state.get('force_rerun', False)

            else:
                st.info(f"The {selected_quiz} quiz will be available soon!")
        
        # Tab 5: Resources
        with education_tabs[4]:
            st.markdown("### Additional Resources")
            st.markdown("""
            Explore these additional resources to deepen your understanding of waste management, 
            recycling, and sustainable living practices.
            """)
            
            # Articles
            st.subheader("Articles and Guides")
            articles = [
                {
                    "title": "The Ultimate Guide to Home Composting",
                    "description": "Learn how to start and maintain an effective home composting system.",
                    "url": "#composting-guide",
                    "tags": ["composting", "organic", "beginner"]
                },
                {
                    "title": "Understanding Plastic Recycling Symbols",
                    "description": "A comprehensive breakdown of plastic recycling numbers and what they mean.",
                    "url": "#plastic-symbols",
                    "tags": ["plastic", "recycling", "guide"]
                },
                {
                    "title": "How to Create a Zero-Waste Kitchen",
                    "description": "Practical tips for reducing waste in your kitchen and food preparation.",
                    "url": "#zero-waste-kitchen",
                    "tags": ["zero-waste", "kitchen", "tips"]
                },
                {
                    "title": "E-Waste: How to Properly Dispose of Electronics",
                    "description": "Guidelines for responsible disposal and recycling of electronic devices.",
                    "url": "#ewaste-guide",
                    "tags": ["electronic", "e-waste", "disposal"]
                }
            ]
            
            # Display articles with filtering
            tags = set()
            for article in articles:
                for tag in article["tags"]:
                    tags.add(tag)
            
            selected_tags = st.multiselect("Filter by topic:", sorted(list(tags)))
            
            filtered_articles = articles
            if selected_tags:
                filtered_articles = [
                    article for article in articles 
                    if any(tag in selected_tags for tag in article["tags"])
                ]
            
            for article in filtered_articles:
                st.markdown(f'<div class="waste-card">', unsafe_allow_html=True)
                st.markdown(f"#### [{article['title']}]({article['url']})")
                st.markdown(article["description"])
                st.markdown(f"Tags: {', '.join(article['tags'])}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            # External resources
            st.subheader("External Organizations")
            
            organizations = [
                {
                    "name": "Environmental Protection Agency (EPA)",
                    "description": "Official resources on waste management and environmental protection.",
                    "url": "https://www.epa.gov/recycle"
                },
                {
                    "name": "Earth911",
                    "description": "Comprehensive recycling database to find local recycling options.",
                    "url": "https://earth911.com"
                },
                {
                    "name": "Plastic Pollution Coalition",
                    "description": "Global alliance working to reduce plastic pollution and its impacts.",
                    "url": "https://www.plasticpollutioncoalition.org"
                },
                {
                    "name": "Composting Council",
                    "description": "Educational resources on composting practices and benefits.",
                    "url": "https://www.compostingcouncil.org"
                }
            ]
            
            for org in organizations:
                st.markdown(f"- [{org['name']}]({org['url']}) - {org['description']}")
    
    # Footer
    st.markdown('<div class="footer">', unsafe_allow_html=True)
    st.markdown("Smart Waste Classification System | Developed with ♥ for a cleaner planet")
    st.markdown("© 2023 EcoScan Technologies | Privacy Policy | Terms of Service")
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
