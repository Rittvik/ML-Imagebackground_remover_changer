import streamlit as st
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
from PIL import Image
import os
from streamlit_cropper import st_cropper

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

def process_image(image, background_image=None, background_color=None, scale=1.0, x_offset=0, y_offset=0):
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

    # Determine canvas dimensions and background
    if background_image:
        bg_img = background_image
        bg_img.thumbnail((max_dim, max_dim)) # avoid huge backgrounds
        out_w, out_h = bg_img.size
        background = np.array(bg_img.convert('RGB'))
    else:
        out_h, out_w, _ = foreground.shape
        background = np.zeros((out_h, out_w, 3), dtype=np.uint8)
        if background_color:
            hex_color = background_color.lstrip('#')
            r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
            background[:] = (r, g, b)

    # Resize foreground based on scale
    new_w = int(foreground.shape[1] * scale)
    new_h = int(foreground.shape[0] * scale)
    if new_w <= 0 or new_h <= 0:
        return Image.fromarray(background)
        
    scaled_fg = cv2.resize(foreground, (new_w, new_h))
    scaled_mask = cv2.resize(mask_3d, (new_w, new_h))
    
    # Blend images
    output_image = background.copy()
    
    # Calculate bounding box on the background
    y1 = y_offset
    y2 = y_offset + new_h
    x1 = x_offset
    x2 = x_offset + new_w
    
    # Clip to background bounds
    bg_y1 = max(0, y1)
    bg_y2 = min(out_h, y2)
    bg_x1 = max(0, x1)
    bg_x2 = min(out_w, x2)
    
    # Calculate corresponding bounding box on the foreground
    fg_y1 = bg_y1 - y1
    fg_y2 = new_h - (y2 - bg_y2)
    fg_x1 = bg_x1 - x1
    fg_x2 = new_w - (x2 - bg_x2)
    
    if bg_y1 < bg_y2 and bg_x1 < bg_x2:
        target_bg = output_image[bg_y1:bg_y2, bg_x1:bg_x2]
        target_fg = scaled_fg[fg_y1:fg_y2, fg_x1:fg_x2]
        target_mask = scaled_mask[fg_y1:fg_y2, fg_x1:fg_x2]
        
        output_image[bg_y1:bg_y2, bg_x1:bg_x2] = (
            target_fg * target_mask + target_bg * (1 - target_mask)
        ).astype(np.uint8)

    return Image.fromarray(output_image)

@st.dialog("Crop Image")
def crop_image_modal(image_obj, key_prefix):
    """
    Shows the image cropper in a modal window.
    """
    st.write("Use the tools below to crop your image.")
    # Initialize the specific session state variable for this cropper
    box_key = f"{key_prefix}_crop_box"
    if box_key not in st.session_state:
         st.session_state[box_key] = None

    # We use return_type="box" to get the coordinates
    crop_box = st_cropper(
         image_obj, 
         realtime_update=True, 
         box_color='#00FF00' if key_prefix == 'bg' else '#FF0000', 
         key=f"{key_prefix}_cropper_widget",
         return_type="box",
         default_coords=st.session_state[box_key]
    )
    
    if st.button("Save Crop", use_container_width=True):
         # Save coordinates to session state and trigger rerun to close modal
         st.session_state[box_key] = crop_box
         st.rerun()

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

    bg_image = None
    if bg_mode == "Custom Image" and bg_image_file is not None:
        raw_bg = Image.open(bg_image_file).convert('RGB')
        
        # Determine if we have a saved crop box
        bg_box = st.session_state.get('bg_crop_box')
        
        if st.sidebar.button("Crop Background Image"):
             crop_image_modal(raw_bg, "bg")
             
        # Manually crop the image if coordinates exist
        if bg_box:
             left, top, width, height = (bg_box['left'], bg_box['top'], bg_box['width'], bg_box['height'])
             bg_image = raw_bg.crop((left, top, left + width, top + height))
        else:
             # Fallback if no box is returned yet
             bg_image = raw_bg

    # Main Area
    uploaded_file = st.file_uploader("Upload Foreground Image", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        raw_image = Image.open(uploaded_file).convert('RGB')
        
        # Determine if we have a saved crop box
        fg_box = st.session_state.get('fg_crop_box')

        if st.button("Crop Foreground Image"):
             crop_image_modal(raw_image, "fg")
             
        if fg_box:
             left, top, width, height = (fg_box['left'], fg_box['top'], fg_box['width'], fg_box['height'])
             image = raw_image.crop((left, top, left + width, top + height))
        else:
             image = raw_image

        st.sidebar.header("Foreground Adjustments")
        fg_scale = st.sidebar.slider("Foreground Scale", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
        fg_x_offset = st.sidebar.slider("X Offset", min_value=-2000, max_value=2000, value=0, step=10)
        fg_y_offset = st.sidebar.slider("Y Offset", min_value=-2000, max_value=2000, value=0, step=10)

        # Columns for side-by-side view
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original (Cropped)")
            st.image(image, use_column_width=True)

        # Process
        with st.spinner("Removing background..."):
            try:
                # Ensure model exists
                if not os.path.exists('selfie_segmenter.tflite'):
                     st.error("Model file 'selfie_segmenter.tflite' not found. Please download it.")
                else:
                    result = process_image(
                        image, 
                        background_image=bg_image, 
                        background_color=bg_color,
                        scale=fg_scale,
                        x_offset=fg_x_offset,
                        y_offset=fg_y_offset
                    )
                    
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
