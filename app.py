import streamlit as st
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
from PIL import Image
import os

# Initialize MediaPipe Image Segmenter
# cache the segmenter to avoid reloading model on every run?
# Streamlit re-runs the script.
# We should probably load it inside the function or use @st.cache_resource

@st.cache_resource
def get_segmenter():
    base_options = python.BaseOptions(model_asset_path='selfie_segmenter.tflite')
    options = vision.ImageSegmenterOptions(base_options=base_options,
                                           output_category_mask=True)
    return vision.ImageSegmenter.create_from_options(options)

def process_image(image, background_image=None, background_color=None):
    """
    Processes the input image to replace the background.
    """
    # Resize if too large to keep latency low (max dimension 1280)
    max_dim = 1280
    if image.width > max_dim or image.height > max_dim:
        image.thumbnail((max_dim, max_dim))

    # Convert PIL Image to NumPy array
    img_np = np.array(image)
    
    # Image is already RGB due to .convert('RGB') in main
    img_rgb = img_np
    
    # Create MP Image
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    
    # Run Segmentation
    segmenter = get_segmenter()
    segmentation_result = segmenter.segment(mp_image)
    category_mask = segmentation_result.category_mask
    
    # category_mask is a mediapipe Image object? No, it's a generic object with .numpy_view()
    mask_np = category_mask.numpy_view() # uint8, 0 or 255
    
    # Convert mask to float 0..1 for blending
    # Based on testing: 0 is Person, 255 is Background
    mask = (mask_np == 0).astype(np.float32)

    # Apply Gaussian Blur to the mask for feathering edges
    mask = cv2.GaussianBlur(mask, (15, 15), 0)
    
    # Create 3-channel mask for broadcasting
    mask_3d = np.stack((mask,) * 3, axis=-1)

    # Prepare Foreground
    foreground = img_rgb

    # Prepare Background
    h, w, _ = foreground.shape
    
    if background_image:
        # Resize custom background to match foreground
        bg_img = background_image.resize((w, h))
        background = np.array(bg_img)
        # Ensure background is RGB
        if background.shape[-1] == 4:
            background = cv2.cvtColor(background, cv2.COLOR_RGBA2RGB)
    elif background_color:
        # Create solid color background
        # background_color is expected to be a hex string like '#RRGGBB'
        hex_color = background_color.lstrip('#')
        r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        background = np.zeros_like(foreground)
        background[:] = (r, g, b)
    else:
        # Default to black
        background = np.zeros_like(foreground)

    # Blend images
    # Output = Foreground * Mask + Background * (1 - Mask)
    output_image = (foreground * mask_3d + background * (1 - mask_3d)).astype(np.uint8)

    return Image.fromarray(output_image)

def main():
    st.set_page_config(page_title="Streamlit Background Remover", layout="wide")
    
    st.title("Streamlit Background Remover")
    st.write("Upload an image and replace the background using AI.")

    # Sidebar for settings
    st.sidebar.header("Settings")
    bg_mode = st.sidebar.radio("Background Mode", ["Solid Color", "Custom Image"])

    bg_color = None
    bg_image_file = None

    if bg_mode == "Solid Color":
        color_choice = st.sidebar.selectbox(
            "Choose a Background Color", 
            ["Green", "Blue", "Red", "Black", "White", "Yellow", "Cyan", "Magenta", "Custom"]
        )
        if color_choice == "Custom":
            bg_color = st.sidebar.color_picker("Pick a Custom Color", "#00FF00")
        else:
            color_map = {
                "Green": "#00FF00",
                "Blue": "#0000FF",
                "Red": "#FF0000",
                "Black": "#000000",
                "White": "#FFFFFF",
                "Yellow": "#FFFF00",
                "Cyan": "#00FFFF",
                "Magenta": "#FF00FF"
            }
            bg_color = color_map[color_choice]
    else:
        bg_image_file = st.sidebar.file_uploader("Upload Background Image", type=["png", "jpg", "jpeg"])

    # Main Area
    uploaded_file = st.file_uploader("Upload Foreground Image", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        
        bg_image = None
        if bg_image_file:
            bg_image = Image.open(bg_image_file).convert('RGB')

        # Columns for side-by-side view
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original")
            st.image(image, use_column_width=True)

        # Process
        with st.spinner("Removing background..."):
            try:
                # Ensure model exists
                if not os.path.exists('selfie_segmenter.tflite'):
                     st.error("Model file 'selfie_segmenter.tflite' not found. Please download it.")
                else:
                    result = process_image(image, background_image=bg_image, background_color=bg_color)
                    
                    with col2:
                        st.subheader("Result")
                        st.image(result, use_column_width=True)
                        
                        # Download button
                        from io import BytesIO
                        buf = BytesIO()
                        result.save(buf, format="PNG")
                        byte_im = buf.getvalue()
                        
                        st.download_button(
                            label="Download Result",
                            data=byte_im,
                            file_name="background_removed.png",
                            mime="image/png"
                        )
            except Exception as e:
                st.error(f"Error processing image: {e}")
                # Print exception helper
                import traceback
                st.text(traceback.format_exc())

if __name__ == "__main__":
    main()
