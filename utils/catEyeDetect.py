import cv2
import numpy as np

def detect_cat_eyes(image_path):
    # Read the image
    img = cv2.imread(image_path)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply threshold to isolate dark regions
    _, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours based on aspect ratio and area
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = h / w
        area = cv2.contourArea(contour)
        
        # Check if the contour is vertical (height > width) and has appropriate area
        if aspect_ratio > 2 and 50 < area < 1000:
            # Draw rectangle around potential cat eye
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # Display the result
    cv2.imshow('Cat Eye Detection', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
image_path = '../misc/cat3.jpg'  # Update this path to your cat image
detect_cat_eyes(image_path)

