import cv2
import numpy as np
import mediapipe as mp

def segment_hand(image, bbox=None):
    """
    Segment a hand from an image given a bounding box. The hand may extend beyond the bounding box.
    Uses MediaPipe Hands to refine the segmentation.
    
    :param image: Input image (BGR format)
    :param bbox: Optional tuple of (x, y, w, h) representing the bounding box where part of the hand is located
    :return: Segmented hand image
    """
    if bbox is None:
        # Use the entire image if no bbox is provided
        h, w = image.shape[:2]
        bbox = (0, 0, w, h)
    
    # We'll use the entire image for processing, but we'll use the bbox to focus our initial search
    x, y, w, h = bbox
    
    # The ROI is now the entire image
    roi = image.copy()
    # Convert ROI to YCrCb color space
    ycrcb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCrCb)
    
    # Define the range for skin color in YCrCb space
    lower = np.array([0, 135, 85], dtype=np.uint8)
    upper = np.array([255, 180, 135], dtype=np.uint8)
    
    # Create a binary mask of the skin color
    mask = cv2.inRange(ycrcb, lower, upper)
    
    # Apply morphological operations to clean up the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Use MediaPipe Hands to refine the segmentation
    mp_hands = mp.solutions.hands
    with mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5) as hands:
        # Convert the image to RGB for MediaPipe
        rgb_image = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_image)
        
        if results.multi_hand_landmarks:
            # Create a new mask based on MediaPipe hand landmarks
            refined_mask = np.zeros(mask.shape, dtype=np.uint8)
            h, w = mask.shape[:2]
            
            # Draw a rectangle that contains all hand landmarks on the refined mask, with 10% extra area
            for hand_landmarks in results.multi_hand_landmarks:
                landmarks = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks.landmark]
                x_min = min(landmark[0] for landmark in landmarks)
                y_min = min(landmark[1] for landmark in landmarks)
                x_max = max(landmark[0] for landmark in landmarks)
                y_max = max(landmark[1] for landmark in landmarks)
                
                # Calculate the width and height of the original rectangle
                width = x_max - x_min
                height = y_max - y_min
                
                # Calculate the amount to expand (5% on each side)
                expand_x = int(width * 0.2)
                expand_y = int(height * 0.2)
                
                # Expand the rectangle by 10% (5% on each side)
                x_min = max(0, x_min - expand_x)
                y_min = max(0, y_min - expand_y)
                x_max = min(w, x_max + expand_x)
                y_max = min(h, y_max + expand_y)
                
                cv2.rectangle(refined_mask, (x_min, y_min), (x_max, y_max), 255, -1)
            # Combine the original mask with the refined mask
            final_mask = cv2.bitwise_and(mask, refined_mask)
            
            # Apply the final mask to the original ROI
            segmented_hand = cv2.bitwise_and(roi, roi, mask=final_mask)
            
            return segmented_hand
    
    # If MediaPipe didn't detect a hand, fall back to the original method
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Find the largest contour (assumed to be the hand)
        hand_contour = max(contours, key=cv2.contourArea)
        
        # Create a refined mask for the hand
        refined_mask = np.zeros(mask.shape, dtype=np.uint8)
        cv2.drawContours(refined_mask, [hand_contour], 0, 255, -1)
        
        # Apply the refined mask to the original ROI
        segmented_hand = cv2.bitwise_and(roi, roi, mask=refined_mask)
        
        return segmented_hand
    else:
        return None

# Display the segmented hand
def display_segmented_hand(image, bbox=None):
    segmented_hand = segment_hand(image, bbox)
    if segmented_hand is not None:
        cv2.imshow('Segmented Hand', segmented_hand)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No hand detected in the given region.")

# Example usage:
image = cv2.imread('../misc/finger4.jpg')
display_segmented_hand(image)  


