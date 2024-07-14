import cv2
import numpy as np

def preprocess_image(image):
    #Convert the image to grayscale
    image = cv2.resize(image, (640, 640))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Apply GaussianBlur to reduce noise
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)

    # Apply adaptive thresholding to create a binary image
    binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)

    # Use morphological operations to slightly close gaps and remove noise
    kernel = np.ones((5, 5), np.uint8)
    morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel, iterations=1)

    # Find contours in the morphologically processed image
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a mask for the detected contours
    mask = np.zeros_like(gray)

    # Filter out small contours to reduce noise
    min_contour_area = 1000  # Adjust this value based on your image
    filtered_contours = [c for c in contours if cv2.contourArea(c) > min_contour_area]

    # Draw the filtered contours on the mask
    cv2.drawContours(mask, filtered_contours, -1, (255), thickness=cv2.FILLED)

    # Apply the mask to the original grayscale image
    result = cv2.bitwise_and(enhanced, enhanced, mask=mask)

    return result


