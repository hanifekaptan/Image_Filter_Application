import cv2
import numpy as np
import gradio as gr

# Farklı filtre fonksiyonları
def apply_gaussian_blur_filter(frame):
    return cv2.GaussianBlur(frame, (15, 15), 0)

def apply_sharpening_filter(frame):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(frame, -1, kernel)

def apply_edge_detection_filter(frame):
    return cv2.Canny(frame, 100, 200)

def apply_invert_filter(frame):
    return cv2.bitwise_not(frame)

def adjust_brightness_contrast_filter(frame, alpha=1.0, beta=50):
    return cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)

def apply_grayscale_filter(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

def apply_hsvscale_filter(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

def apply_median_blur_filter(frame):
    return cv2.medianBlur(frame, 5)

def apply_bilateral_filter(frame):
    return cv2.bilateralFilter(frame, 9, 75, 75)

def apply_box_filter(frame):
    return cv2.boxFilter(frame, -1, (5, 5))

def apply_laplacian_filter(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.convertScaleAbs(cv2.Laplacian(gray_frame, cv2.CV_64F))

def apply_sobel_filter(frame):
    sobelx = cv2.Sobel(frame, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(frame, cv2.CV_64F, 0, 1, ksize=5)
    return cv2.bitwise_or(sobelx, sobely)

def apply_scharr_filter(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    scharr_x = cv2.Scharr(gray_frame, cv2.CV_64F, 1, 0)
    scharr_y = cv2.Scharr(gray_frame, cv2.CV_64F, 0, 1)
    return cv2.convertScaleAbs(cv2.sqrt(scharr_x**2 + scharr_y**2))

def apply_erosion_filter(frame):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.erode(frame, kernel, iterations=1)

def apply_dilation_filter(frame):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.dilate(frame, kernel, iterations=1)

def apply_morph_gradient_filter(frame):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.morphologyEx(frame, cv2.MORPH_GRADIENT, kernel)

def apply_morph_top_hat_filter(frame):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.morphologyEx(frame, cv2.MORPH_TOPHAT, kernel)

def apply_morph_black_hat_filter(frame):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.morphologyEx(frame, cv2.MORPH_BLACKHAT, kernel)

def apply_gaussian_pyramid_filter(frame):
    return cv2.pyrDown(frame)

def apply_laplacian_pyramid_filter(frame):
    return cv2.pyrUp(frame)

def apply_clahe_filter(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray_frame)

def apply_sepia_filter(frame):
    sepia_filter = np.array([[0.272, 0.534, 0.131],
                             [0.349, 0.686, 0.168],
                             [0.393, 0.769, 0.189]])
    return cv2.transform(frame, sepia_filter)

def apply_fall_filter(frame):
    fall_filter = np.array([[0.393, 0.769, 0.189],
                            [0.349, 0.686, 0.168],
                            [0.272, 0.534, 0.131]])
    return cv2.transform(frame, fall_filter)

# Filtre uygulama fonksiyonu
def apply_filter(filter_type, input_image=None):
    if input_image is not None:
        frame = input_image
    else:
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            return "Web kameradan görüntü alınamadı"

    if filter_type == "Gaussian Blur":
        return apply_gaussian_blur_filter(frame)
    elif filter_type == "Sharpen":
        return apply_sharpening_filter(frame)
    elif filter_type == "Edge Detection":
        return apply_edge_detection_filter(frame)
    elif filter_type == "Invert":
        return apply_invert_filter(frame)
    elif filter_type == "Brightness":
        return adjust_brightness_contrast_filter(frame, alpha=1.0, beta=50)
    elif filter_type == "Grayscale":
        return apply_grayscale_filter(frame)
    elif filter_type == "HSV Scale":
        return apply_hsvscale_filter(frame)
    elif filter_type == "Median Blur":
        return apply_median_blur_filter(frame)
    elif filter_type == "Bilateral":
        return apply_bilateral_filter(frame)
    elif filter_type == "Box":
        return apply_box_filter(frame)
    elif filter_type == "Laplacian":
        return apply_laplacian_filter(frame)
    elif filter_type == "Sobel":
        return apply_sobel_filter(frame)
    elif filter_type == "Scharr":
        return apply_scharr_filter(frame)
    elif filter_type == "Erosion":
        return apply_erosion_filter(frame)
    elif filter_type == "Dilation":
        return apply_dilation_filter(frame)
    elif filter_type == "Morphological Gradient":
        return apply_morph_gradient_filter(frame)
    elif filter_type == "Morphological Top Hat":
        return apply_morph_top_hat_filter(frame)
    elif filter_type == "Morphological Black Hat":
        return apply_morph_black_hat_filter(frame)
    elif filter_type == "Gaussian Pyramid":
        return apply_gaussian_pyramid_filter(frame)
    elif filter_type == "Laplacian Pyramid":
        return apply_laplacian_pyramid_filter(frame)
    elif filter_type == "CLAHE":
        return apply_clahe_filter(frame)
    elif filter_type == "Sepia":
        return apply_sepia_filter(frame)
    elif filter_type == "Fall":
        return apply_fall_filter(frame)

# Gradio arayüzü
with gr.Blocks() as demo:
    gr.Markdown("# Web Kameradan Canlı Filtreleme")

    # Filtre seçenekleri
    filter_type = gr.Dropdown(
        label="Filtre Seçin",
        choices=["Gaussian Blur", "Sharpen", "Edge Detection",
                 "Invert", "Brightness", "Grayscale", "HSV Scale",
                 "Median Blur", "Bilateral", "Box", "Laplacian",
                 "Sobel", "Scharr", "Erosion", "Dilation",
                 "Morphological Gradient", "Morphological Top Hat",
                 "Morphological Black Hat", "Gaussian Pyramid",
                 "Laplacian Pyramid", "CLAHE", "Sepia", "Fall"],
        value="Gaussian Blur"
    )

    # Görüntü yükleme alanı
    input_image = gr.Image(label="Resim Yükle", type="numpy")

    # Çıktı için görüntü
    output_image = gr.Image(label="Filtre Uygulandı")

    # Filtre uygula butonu
    apply_button = gr.Button("Filtreyi Uygula")

    # Butona tıklanınca filtre uygulama fonksiyonu
    apply_button.click(fn=apply_filter, inputs=[filter_type, input_image], outputs=output_image)

# Gradio arayüzünü başlat
demo.launch()
