import streamlit as st
from streamlit_cropper import st_cropper
from PIL import Image
import numpy as np
import io
from google.cloud import vision
import os
import json

# -------------------------------
# 🔧 Google Cloud Setup
# -------------------------------
# Use a raw string (r"...") to prevent Python from interpreting 
# backslashes (\) as escape sequences.

try:
    # Load credentials directly from Streamlit Secrets
    creds_json = st.secrets["GOOGLE_APPLICATION_CREDENTIALS"]
    creds_dict = json.loads(creds_json)
    client = vision.ImageAnnotatorClient.from_service_account_info(creds_dict)
    st.session_state['client_ready'] = True
except Exception as e:
    st.error("❌ Could not load Google Vision API credentials from Streamlit Secrets.")
    st.exception(e)
    st.session_state['client_ready'] = False
    client = None

# -------------------------------
#  Streamlit App UI
# -------------------------------
# Set wide layout for best use of space
st.set_page_config(page_title="Vision API Demo", page_icon="🧠", layout="wide")
st.title("Vision API Demo - Object Detection Tool")

# Sidebar
st.sidebar.header("⚙️ Options")
aspect_ratio_option = st.sidebar.selectbox(
    "Aspect Ratio",
    ("Free", "1:1", "16:9", "4:3")
)
aspect_dict = {"Free": None, "1:1": (1, 1), "16:9": (16, 9), "4:3": (4, 3)}
aspect_ratio = aspect_dict[aspect_ratio_option]

# Upload Image
uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Check for image upload success
    try:
        image = Image.open(uploaded_file)
    except Exception as e:
        st.error(f"Error reading image file: {e}")
        image = None
        
    if image:
        # Unified Cropper and Output Section
        # [2 parts for the interactive cropper, 1 part for the output]
        col_cropper, col_output = st.columns([2, 1]) 

        # Column 1: Interactive Cropper
        with col_cropper:
            st.subheader("✂️ Select Area on Image")
            cropped_img = st_cropper(
                image,
                realtime_update=True,
                box_color='#0077B6',
                aspect_ratio=aspect_ratio,
                return_type="image",
                key="canvas"
            )
            st.text("")

        # Column 2: Cropped Result and Action Button
        with col_output:
            st.subheader("✅ Cropped Result & Action")
            # Using width='stretch' to fix Streamlit deprecation warning
            st.image(cropped_img, width='stretch', caption="Selected Region")

            st.markdown("<br>", unsafe_allow_html=True)
            
            # Only allow detection if the client initialized successfully
            if st.session_state.get('client_ready', False):
                if st.button("🚀 Run Detection (Object Localization)", width='stretch'):
                    st.info("Running Object Localization on cropped image...")
                    
                    # --- Image Pre-processing (RGBA to RGB) ---
                    if cropped_img.mode == 'RGBA':
                        # Convert to RGB by pasting onto a white background
                        background = Image.new('RGB', cropped_img.size, (255, 255, 255))
                        background.paste(cropped_img, mask=cropped_img.split()[3])
                        final_img = background
                    else:
                        final_img = cropped_img
                        
                    # Convert final image to bytes for API
                    buf = io.BytesIO()
                    final_img.save(buf, format="JPEG") 
                    content = buf.getvalue()

                    image_for_api = vision.Image(content=content)

                    # --- Run Vision API OBJECT LOCALIZATION ---
                    response = client.object_localization(image=image_for_api)
                    
                    # FIX: Corrected the attribute name based on the traceback.
                    localized_objects = response.localized_object_annotations 

                    if response.error.message:
                        st.error(f"API Error: {response.error.message}")
                    else:
                        st.success("✅ Object Localization Complete")
                        st.markdown("### 🔍 Detected Objects:")
                        
                        if localized_objects:
                            # Display results
                            for obj in localized_objects:
                                # Object Localization returns name and score
                                st.write(f"- **{obj.name}** (Score: {obj.score:.2%})")
                                # Display bounding box info (optional but helpful for visualization)
                                try:
                                    # Normalized vertices give coordinates from 0 to 1
                                    v = obj.bounding_poly.normalized_vertices
                                    st.caption(f"Box: ({v[0].x:.2f}, {v[0].y:.2f}) to ({v[2].x:.2f}, {v[2].y:.2f})")
                                except:
                                    st.caption("Bounding box data unavailable.")
                        else:
                            st.info("No specific objects were localized in the selected region.")
            else:
                st.warning("Cannot run detection: Vision API client failed to initialize.")

else:
    st.info("👆 Upload an image to begin.")


