import cv2
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

def gaussian_blur_filter(frame):
    return cv2.GaussianBlur(frame, (15, 15), 0)

def sharpening_filter(frame):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(frame, -1, kernel)

def edge_detection_filter(frame):
    return cv2.Canny(frame, 100, 200)

def invert_filter(frame):
    return cv2.bitwise_not(frame)

def adjust_brightness_contrast_filter(frame, alpha=1.0, beta=50):
    return cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)

def grayscale_filter(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

def hsvscale_filter(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

def median_blur_filter(frame):
    return cv2.medianBlur(frame, 5)

def bilateral_filter(frame):
    return cv2.bilateralFilter(frame, 9, 75, 75)

def box_filter(frame):
    return cv2.boxFilter(frame, -1, (5, 5))

def laplacian_filter(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.convertScaleAbs(cv2.Laplacian(gray_frame, cv2.CV_64F))

def sobel_filter(frame):
    sobelx = cv2.Sobel(frame, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(frame, cv2.CV_64F, 0, 1, ksize=5)
    return cv2.bitwise_or(sobelx, sobely)

def scharr_filter(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    scharr_x = cv2.Scharr(gray_frame, cv2.CV_64F, 1, 0)
    scharr_y = cv2.Scharr(gray_frame, cv2.CV_64F, 0, 1)
    return cv2.convertScaleAbs(cv2.sqrt(scharr_x**2 + scharr_y**2))

def erosion_filter(frame):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.erode(frame, kernel, iterations=1)

def dilation_filter(frame):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.dilate(frame, kernel, iterations=1)

def morph_gradient_filter(frame):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.morphologyEx(frame, cv2.MORPH_GRADIENT, kernel)

def morph_top_hat_filter(frame):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.morphologyEx(frame, cv2.MORPH_TOPHAT, kernel)

def morph_black_hat_filter(frame):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.morphologyEx(frame, cv2.MORPH_BLACKHAT, kernel)

def gaussian_pyramid_filter(frame):
    return cv2.pyrDown(frame)

def laplacian_pyramid_filter(frame):
    return cv2.pyrUp(frame)

def clahe_filter(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray_frame)

def sepia_filter(frame):
    sepia_filter = np.array([[0.272, 0.534, 0.131],
                             [0.349, 0.686, 0.168],
                             [0.393, 0.769, 0.189]])
    return cv2.transform(frame, sepia_filter)

def fall_filter(frame):
    fall_filter = np.array([[0.393, 0.769, 0.189],
                            [0.349, 0.686, 0.168],
                            [0.272, 0.534, 0.131]])
    return cv2.transform(frame, fall_filter)

def snow_filter(frame):
    snow_filter = np.array([[0.5, 0.5, 0.5],
                            [0.5, 0.5, 0.5],
                            [0.5, 0.5, 0.5]])
    return cv2.transform(frame, snow_filter)

def rain_filter(frame):
    rain_filter = np.array([[0.3, 0.59, 0.11],
                            [0.3, 0.59, 0.11],
                            [0.3, 0.59, 0.11]])
    return cv2.transform(frame, rain_filter)

def apply_filter(filter_type, input_image = None):
    if input_image is not None:
        frame = input_image

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
    
    for i in call_func:
        if i == filter_type:
            return call_func[i](frame)

    # if filter_type == "Gaussian Blur":
    #     return gaussian_blur_filter(frame)
    # elif filter_type == "Sharpen":
    #     return sharpening_filter(frame)
    # elif filter_type == "Edge Detection":
    #     return edge_detection_filter(frame)
    # elif filter_type == "Invert":
    #     return invert_filter(frame)
    # elif filter_type == "Brightness & Contrast":
    #     return adjust_brightness_contrast_filter(frame)
    # elif filter_type == "Grayscale":
    #     return grayscale_filter(frame)
    # elif filter_type == "HSV Scale":
    #     return hsvscale_filter(frame)
    # elif filter_type == "Median Blur":
    #     return median_blur_filter(frame)
    # elif filter_type == "Bilateral":
    #     return bilateral_filter(frame)
    # elif filter_type == "Box":
    #     return box_filter(frame)
    # elif filter_type == "Laplacian":
    #     return laplacian_filter(frame)
    # elif filter_type == "Sobel":
    #     return sobel_filter(frame)
    # elif filter_type == "Scharr":
    #     return scharr_filter(frame)
    # elif filter_type == "Erosion":
    #     return erosion_filter(frame)
    # elif filter_type == "Dilation":
    #     return dilation_filter(frame)
    # elif filter_type == "Morph Gradient":
    #     return morph_gradient_filter(frame)
    # elif filter_type == "Morph Top Hat":
    #     return morph_top_hat_filter(frame)
    # elif filter_type == "Morph Black Hat":
    #     return morph_black_hat_filter(frame)
    # elif filter_type == "Gaussian Pyramid":
    #     return gaussian_pyramid_filter(frame)
    # elif filter_type == "Laplacian Pyramid":
    #     return laplacian_pyramid_filter(frame)
    # elif filter_type == "CLAHE":
    #     return clahe_filter(frame)
    # elif filter_type == "Sepia":
    #     return sepia_filter(frame)
    # elif filter_type == "Fall":
    #     return fall_filter(frame)
    # elif filter_type == "Snow":
    #     return snow_filter(frame)
    # elif filter_type == "Rain":
    #     return rain_filter(frame)
    

st.title("Image Filters App")

filters = ["Gaussian Blur", "Sharpen", "Edge Detection", "Invert", "Brightness & Contrast", "Grayscale",
           "HSV Scale", "Median Blur", "Bilateral", "Box", "Laplacian", "Sobel", "Scharr", "Erosion",
           "Dilation", "Morph Gradient", "Morph Top Hat", "Morph Black Hat", "Gaussian Pyramid", "Fall",
           "Laplacian Pyramid", "CLAHE", "Sepia", "Snow", "Rain" ]

selected_filter = st.selectbox("Select a filter", filters)

input_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if st.button("Apply"):
    if input_image is not None:
        image = plt.imread(input_image)
        frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        filtered_image = apply_filter(selected_filter, frame)

        # Eğer filtre sonucu tek kanallı ise, bunu 3 kanallı hale getir
        if len(filtered_image.shape) == 2:  # Tek kanal durumu
            filtered_image = cv2.cvtColor(filtered_image, cv2.COLOR_GRAY2BGR)

        # Geçersiz değerleri düzelt
        filtered_image = np.nan_to_num(filtered_image, nan=0.0, posinf=255.0, neginf=0.0)
        filtered_image = np.clip(filtered_image, 0, 255).astype(np.uint8)

        st.image(filtered_image, channels="BGR")
    else:
        st.error("Please upload an image.")