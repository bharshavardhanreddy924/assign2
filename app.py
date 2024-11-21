import streamlit as st
import numpy as np
import cv2
import os
import torch
import pandas as pd

from rich.console import Console
from rich.table import Table

# Direct import since files are in same directory
from pipe import ISPPipeline, DenoiseSharpenPipeline
def _safe_dncnn_denoise(pipeline, image):
    """
    Safely apply DnCNN denoising with fallback
    """
    try:
        # Check if DnCNN model is loaded
        if pipeline.dncnn is None:
            print("DnCNN model not loaded. Falling back to original image.")
            return image
        
        # Ensure image is in correct format for DnCNN
        if len(image.shape) == 2:  # Grayscale
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        # Preprocess image for DnCNN
        img_tensor = torch.from_numpy(image.transpose((2, 0, 1))).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(pipeline.device)
        
        # Apply denoising
        with torch.no_grad():
            denoised_tensor = pipeline.dncnn(img_tensor)
        
        # Convert back to numpy
        denoised = denoised_tensor.squeeze().cpu().numpy().transpose((1, 2, 0))
        denoised = (np.clip(denoised, 0, 1) * 255).astype(np.uint8)
        
        return denoised
    
    except Exception as e:
        print(f"Error in DnCNN denoising: {e}")
        return image

def convert_image_for_display(image):
    """Convert image for Streamlit display"""
    if image.dtype == np.uint16:
        image = (image / 256).astype(np.uint8)
    
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 1:
        image = cv2.cvtColor(image.squeeze(), cv2.COLOR_GRAY2RGB)
    
    return image

def main():
    st.set_page_config(page_title="Advanced Image Signal Processing", layout="wide")
    
    st.title("🖼️ Advanced Image Signal Processing: Denoising and Sharpness Techniques")
    
    # Sidebar for configuration
    st.sidebar.header("Image Processing Options")
    
    # File uploader
    uploaded_file = st.sidebar.file_uploader(
        "Upload RAW Image", 
        type=['raw'], 
        help="Upload a 12-bit RAW image in GRBG Bayer pattern"
    )
    
    # Processing methods selection
    processing_options = st.sidebar.multiselect(
        "Select Processing Methods",
        [
            "Demosaic", 
            "White Balance", 
            "Gamma Correction", 
            # Denoising Methods
            "Gaussian Denoising", 
            "Median Denoising", 
            "Bilateral Denoising", 
            "DnCNN Denoising",
            # Sharpening Methods
            "Unsharp Mask", 
            "Laplacian Sharpening"
        ],
        default=["Demosaic", "White Balance", "Gamma Correction", "Gaussian Denoising"]
    )
    
    # Denoising Method Specific Parameters
    st.sidebar.subheader("Denoising Parameters")
    gaussian_kernel = st.sidebar.slider("Gaussian Kernel Size", 3, 11, 5, step=2)
    bilateral_params = st.sidebar.slider("Bilateral Filter Params", 1, 100, 75)
    
    # ROI selection
    st.sidebar.subheader("Region of Interest")
    roi_x = st.sidebar.slider("ROI X Position", 0, 1920, 200)
    roi_y = st.sidebar.slider("ROI Y Position", 0, 1280, 200)
    roi_width = st.sidebar.slider("ROI Width", 100, 800, 400)
    roi_height = st.sidebar.slider("ROI Height", 100, 800, 400)
    
    if uploaded_file is not None:
        # Initialize pipelines
        isp_pipeline = ISPPipeline()
        denoiser_pipeline = DenoiseSharpenPipeline()
        
        # Save uploaded file temporarily
        with open("temp_upload.raw", "wb") as f:
            f.write(uploaded_file.getvalue())
        
        # Read RAW image
        raw_image = isp_pipeline.read_raw("temp_upload.raw")
        
        # Image processing steps
        processed_images = {}
        
        if "Demosaic" in processing_options:
            demosaiced = isp_pipeline.demosaic(raw_image)
            processed_images["Demosaic"] = convert_image_for_display(demosaiced)
        
        if "White Balance" in processing_options:
            wb_image = isp_pipeline.white_balance(demosaiced)
            processed_images["White Balance"] = convert_image_for_display(wb_image)
        
        if "Gamma Correction" in processing_options:
            gamma_image = isp_pipeline.apply_gamma(wb_image)
            processed_images["Gamma Correction"] = convert_image_for_display(gamma_image)
        
        # Denoising methods with custom parameters
        # Denoising methods with custom parameters
        # Modify the denoising methods definition
        denoising_methods = {
            "Gaussian Denoising": lambda img: cv2.GaussianBlur(img, (gaussian_kernel, gaussian_kernel), 1.0),
            "Median Denoising": lambda img: cv2.medianBlur(img, 5),
            "Bilateral Denoising": lambda img: cv2.bilateralFilter(img, 9, bilateral_params, bilateral_params),
            "DnCNN Denoising": lambda img: _safe_dncnn_denoise(denoiser_pipeline, img)
        }
        
        # Apply selected denoising methods
        for method_name, method_func in denoising_methods.items():
            if method_name in processing_options:
                if method_name == "DnCNN Denoising":
                    # Special handling for DnCNN
                    processed_images[method_name] = method_func(gamma_image)
                else:
                    denoised_img = method_func(gamma_image)
                    processed_images[method_name] = convert_image_for_display(denoised_img)
        
        # Sharpening methods
        sharpening_methods = {
            "Unsharp Mask": lambda img: cv2.addWeighted(img, 1.5, cv2.GaussianBlur(img, (5, 5), 1.0), -0.5, 0),
            "Laplacian Sharpening": lambda img: cv2.filter2D(img, -1, np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]))
        }
        
        # Apply selected sharpening methods
        for method_name, method_func in sharpening_methods.items():
            if method_name in processing_options:
                sharpened_img = method_func(gamma_image)
                processed_images[method_name] = convert_image_for_display(sharpened_img)
        
        # Metrics computation
        roi = (roi_x, roi_y, roi_width, roi_height)
        metrics_results = {}
        
        for name, img in processed_images.items():
            snr, edge_strength = denoiser_pipeline.compute_metrics(img, roi)
            metrics_results[name] = {
                'SNR': snr,
                'Edge Strength': edge_strength
            }
        
        # Display results
        col1, col2 = st.columns(2)
        
        with col1:
            st.header("Processed Images")
            for name, img in processed_images.items():
                st.subheader(name)
                st.image(img, channels="RGB")
        
        with col2:
            st.header("Image Quality Metrics")
            metrics_df = pd.DataFrame.from_dict(metrics_results, orient='index')
            st.dataframe(metrics_df)
        
        # ROI visualization
        st.header("Regions of Interest")
        for name, img in processed_images.items():
            img_with_roi = img.copy()
            cv2.rectangle(
                img_with_roi, 
                (roi_x, roi_y), 
                (roi_x + roi_width, roi_y + roi_height), 
                (0, 255, 0), 
                2
            )
            st.image(img_with_roi, caption=f"ROI for {name}", channels="RGB")
        
        # Download results
        st.header("Download Results")
        results_csv = metrics_df.to_csv(index=True)
        st.download_button(
            label="Download Metrics CSV",
            data=results_csv,
            file_name="image_processing_metrics.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()