import cv2
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# Applies Gaussian Blur filter
def gaussian_blur_filter(frame):
    return cv2.GaussianBlur(frame, (15, 15), 0)

# Applies sharpening filter
def sharpening_filter(frame):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(frame, -1, kernel)

# Applies edge detection filter
def edge_detection_filter(frame):
    return cv2.Canny(frame, 100, 200)

# Inverts the image
def invert_filter(frame):
    return cv2.bitwise_not(frame)

# Adjusts brightness and contrast
def adjust_brightness_contrast_filter(frame, alpha=1.0, beta=50):
    return cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)

# Converts the image to grayscale
def grayscale_filter(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Converts the image to HSV color space
def hsvscale_filter(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

# Applies median blur filter
def median_blur_filter(frame):
    return cv2.medianBlur(frame, 5)

# Applies bilateral filter
def bilateral_filter(frame):
    return cv2.bilateralFilter(frame, 9, 75, 75)

# Applies box filter
def box_filter(frame):
    return cv2.boxFilter(frame, -1, (5, 5))

# Applies Laplacian filter
def laplacian_filter(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.convertScaleAbs(cv2.Laplacian(gray_frame, cv2.CV_64F))

# Applies Sobel filter
def sobel_filter(frame):
    sobelx = cv2.Sobel(frame, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(frame, cv2.CV_64F, 0, 1, ksize=5)
    return cv2.bitwise_or(sobelx, sobely)

# Applies Scharr filter
def scharr_filter(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    scharr_x = cv2.Scharr(gray_frame, cv2.CV_64F, 1, 0)
    scharr_y = cv2.Scharr(gray_frame, cv2.CV_64F, 0, 1)
    return cv2.convertScaleAbs(cv2.sqrt(scharr_x**2 + scharr_y**2))

# Applies erosion operation
def erosion_filter(frame):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.erode(frame, kernel, iterations=1)

# Applies dilation operation
def dilation_filter(frame):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.dilate(frame, kernel, iterations=1)

# Applies morphological gradient operation
def morph_gradient_filter(frame):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.morphologyEx(frame, cv2.MORPH_GRADIENT, kernel)

# Applies morphological top hat operation
def morph_top_hat_filter(frame):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.morphologyEx(frame, cv2.MORPH_TOPHAT, kernel)

# Applies morphological black hat operation
def morph_black_hat_filter(frame):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.morphologyEx(frame, cv2.MORPH_BLACKHAT, kernel)

# Applies Gaussian pyramid operation
def gaussian_pyramid_filter(frame):
    return cv2.pyrDown(frame)

# Applies Laplacian pyramid operation
def laplacian_pyramid_filter(frame):
    return cv2.pyrUp(frame)

# Applies CLAHE (Contrast Limited Adaptive Histogram Equalization)
def clahe_filter(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray_frame)

# Applies sepia effect
def sepia_filter(frame):
    sepia_filter = np.array([[0.272, 0.534, 0.131],
                             [0.349, 0.686, 0.168],
                             [0.393, 0.769, 0.189]])
    return cv2.transform(frame, sepia_filter)

# Applies fall effect
def fall_filter(frame):
    fall_filter = np.array([[0.393, 0.769, 0.189],
                            [0.349, 0.686, 0.168],
                            [0.272, 0.534, 0.131]])
    return cv2.transform(frame, fall_filter)

# Applies snow effect
def snow_filter(frame):
    snow_filter = np.array([[0.5, 0.5, 0.5],
                            [0.5, 0.5, 0.5],
                            [0.5, 0.5, 0.5]])
    return cv2.transform(frame, snow_filter)

# Applies rain effect
def rain_filter(frame):
    rain_filter = np.array([[0.3, 0.59, 0.11],
                            [0.3, 0.59, 0.11],
                            [0.3, 0.59, 0.11]])
    return cv2.transform(frame, rain_filter)

# Applies the selected filter
def apply_filter(filter_type, input_image=None):
    if input_image is not None:
        frame = input_image

    # Store filter functions in a dictionary
    call_func = {
        "Gaussian Blur": gaussian_blur_filter,
        "Sharpen": sharpening_filter,
        "Edge Detection": edge_detection_filter,
        "Invert": invert_filter,
        "Brightness & Contrast": adjust_brightness_contrast_filter,
        "Grayscale": grayscale_filter,
        "HSV Scale": hsvscale_filter,
        "Median Blur": median_blur_filter,
        "Bilateral": bilateral_filter,
        "Box": box_filter,
        "Laplacian": laplacian_filter,
        "Sobel": sobel_filter,
        "Scharr": scharr_filter,
        "Erosion": erosion_filter,
        "Dilation": dilation_filter,
        "Morph Gradient": morph_gradient_filter,
        "Morph Top Hat": morph_top_hat_filter,
        "Morph Black Hat": morph_black_hat_filter,
        "Gaussian Pyramid": gaussian_pyramid_filter,
        "Laplacian Pyramid": laplacian_pyramid_filter,
        "CLAHE": clahe_filter,
        "Sepia": sepia_filter,
        "Snow": snow_filter,
        "Fall": fall_filter,
        "Rain": rain_filter
    }
    
    # Call the selected filter
    for i in call_func:
        if i == filter_type:
            return call_func[i](frame)

# Streamlit application title
st.title("Image Filters App")

# User selects a filter
filters = ["Gaussian Blur", "Sharpen", "Edge Detection", "Invert", "Brightness & Contrast", 
           "Grayscale", "HSV Scale", "Median Blur", "Bilateral", "Box", "Laplacian", 
           "Sobel", "Scharr", "Erosion", "Dilation", "Morph Gradient", "Morph Top Hat", 
           "Morph Black Hat", "Gaussian Pyramid", "Fall", "Laplacian Pyramid", 
           "CLAHE", "Sepia", "Snow", "Rain"]

# Allows user to select a filter
selected_filter = st.selectbox("Select a filter", filters)

# Asks the user to upload an image
input_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# Button to apply the filter
if st.button("Apply"):
    if input_image is not None:
        # Reads the uploaded image
        image = plt.imread(input_image)
        frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Change color format

        # Applies the selected filter
        filtered_image = apply_filter(selected_filter, frame)

        # If the filter result is single channel, convert it to three channels
        if len(filtered_image.shape) == 2:  # Single channel case
            filtered_image = cv2.cvtColor(filtered_image, cv2.COLOR_GRAY2BGR)

        # Correct invalid values
        filtered_image = np.nan_to_num(filtered_image, nan=0.0, posinf=255.0, neginf=0.0)
        filtered_image = np.clip(filtered_image, 0, 255).astype(np.uint8)

        # Display the filtered image
        st.image(filtered_image, channels="BGR")
    else:
        st.error("Please upload an image.")
