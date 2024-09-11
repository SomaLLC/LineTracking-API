import cv2
import numpy as np
import mediapipe as mp
import urllib.request
import os
from collections import deque

import firebase_admin
from firebase_admin import credentials, storage

import tensorflow as tf
from tensorflow import keras

from ultralytics import YOLO


# Initialize Firebase Admin SDK
cred = credentials.Certificate("../../credentials.json")
firebase_admin.initialize_app(cred, {
    'storageBucket': 'test-421b9.appspot.com'
})

model = YOLO("../runs/detect/cat-yolo2/weights/best.pt") 

def segment_lips_and_teeth(frame):
    """
    Segment lips and teeth from a frame and make the rest of the frame transparent.
    
    :param frame: Input frame (BGR format)
    :return: Frame with only lips and teeth visible, rest is transparent
    """
    mp_face_mesh = mp.solutions.face_mesh
    
    # Enable GPU acceleration for MediaPipe
    with tf.device('/GPU:0'):
        with mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5) as face_mesh:
            # Convert the BGR image to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process the frame and find facial landmarks
            results = face_mesh.process(rgb_frame)
            
            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]
                
                # Get lip and mouth landmarks
                mouth_landmarks = [
                    # Outer lip contour
                    61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146,
                    # Inner lip contour
                    78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191,
                    # Additional points for better coverage
                    76, 77, 90, 180, 85, 16, 315, 404, 320, 307, 306, 184, 74, 73, 72, 11, 302, 303, 304, 408,
                    # Teeth area
                    62, 96, 89, 179, 86, 15, 316, 403, 319, 325, 292, 407, 272, 271, 268, 12, 38, 41, 42, 183
                ]
                
                # Extract mouth coordinates
                h, w = frame.shape[:2]
                mouth_points = [(int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)) for i in mouth_landmarks]
                
                # Find the outermost points to create a single polygon
                hull = cv2.convexHull(np.array(mouth_points))
                
                # Get the bounding rectangle of the hull
                x, y, w, h = cv2.boundingRect(hull)
                
                # Create a mask for the mouth region, sized to the bounding rectangle
                mask = np.zeros((h, w), dtype=np.uint8)
                
                # Adjust hull points to the new coordinate system
                hull_adjusted = hull - np.array([x, y])
                
                # Fill the polygon completely
                cv2.fillPoly(mask, [hull_adjusted], 255)
                
                # Apply Gaussian blur to soften the mask edges
                mask = cv2.GaussianBlur(mask, (5, 5), 0)
                
                # Create the output frame (transparent background)
                output_frame = np.zeros((h, w, 4), dtype=np.uint8)
                
                # Apply the mask to the original frame and combine with the transparent background
                output_frame[:,:,0:3] = cv2.bitwise_and(frame[y:y+h, x:x+w], frame[y:y+h, x:x+w], mask=mask)
                output_frame[:,:,3] = mask
                
                return output_frame, hull, (x, y)
            else:
                # If no face is detected, return a transparent frame and None for hull and coordinates
                return np.zeros((frame.shape[0], frame.shape[1], 4), dtype=np.uint8), None, None

def detect_cat_face(frame):
    """
    Detect the cat's face in the frame, including side views.
    
    :param frame: Input frame (BGR format)
    :return: Bounding box of the cat's face (x, y, w, h)
    """
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    original_height, original_width = frame.shape[:2]
    img_resized = cv2.resize(img, (224, 224))  # Resize for prediction
    img_normalized = img_resized / 255.0  # Normalize pixel values
    img_batch = np.expand_dims(img_normalized, axis=0)  # Add batch dimension

    # Make prediction
    prediction = catKerasModel.predict(img_batch)

    # Extract and scale the predicted points
    points = []
    for i in range(0, len(prediction[0]), 2):
        x = int((prediction[0][i] * original_width))
        y = int((prediction[0][i+1] * original_height))
        points.append((x, y))

    # Find bounding box
    x_coords, y_coords = zip(*points)

    left = min(x_coords)
    right = max(x_coords)
    top = min(y_coords)
    bottom = max(y_coords)


    print("\n\n\n\n\n\n")
    print(left, top, right, bottom)
    print("\n\n\n\n\n\n")
    return (left, top, max(1,right - left), max(1, bottom - top), points)

def detect_cat_face_using_yolo(frame):
    # Run YOLO model on the image
    results = model(frame)
    # Process the results
    if len(results) > 0 and len(results[0].boxes) > 0:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        areas = (boxes[:, 2]) * (boxes[:, 3])
        biggest_box_index = np.argmax(areas)
        biggest_box = boxes[biggest_box_index]
        return tuple(map(int, biggest_box))
    else:
        return None, None, None, None


def detect_cat_nose(frame, face_rect):
    """
    Detect the cat's nose within the face region using contour detection.
    
    :param frame: Input frame (BGR format)
    :param face_rect: Bounding box of the cat's face (x, y, w, h)
    :return: Coordinates of the cat's nose (x, y)
    """
    x, y, w, h = face_rect
    face_region = frame[y:y+h, x:x+w]
    
    # Convert to grayscale and apply threshold
    gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        # Approximate the contour to a polygon
        epsilon = 0.04 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # Check if the polygon has 3 vertices (triangle)
        if len(approx) == 3:
            # Calculate the orientation of the triangle
            orientation = np.arctan2(approx[1][0][1] - approx[0][0][1], 
                                     approx[1][0][0] - approx[0][0][0])
            orientation = np.degrees(orientation)
            
            # Check if the triangle is pointing downwards (orientation between 45 and 135 degrees)
            if 45 < orientation < 135:
                # Calculate centroid of the triangle
                M = cv2.moments(approx)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    return (x + cx, y + cy)
    
    # If no suitable nose is found, return the center of the upper half of the face
    return (x + w // 2, y + h // 3)

def process_video(human_video_path, cat_video_path, output_path):
    """
    Process videos to segment lips and teeth from the human video,
    and overlay them on the cat's mouth in the cat video.
    
    :param human_video_path: Path to the human video file
    :param cat_video_path: Path to the cat video file
    :param output_path: Path to save the output video file
    """
    human_cap = cv2.VideoCapture(human_video_path)
    cat_cap = cv2.VideoCapture(cat_video_path)
    
    # Get video properties (assuming both videos have the same properties)
    width = int(cat_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cat_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cat_cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cat_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    
    
    last_cat_face = None  # Store the last detected cat face
    face_history = deque(maxlen=30)  # Store last 3 face detections
    
    frame_count = 0
    while human_cap.isOpened() and cat_cap.isOpened():
        human_ret, human_frame = human_cap.read()
        cat_ret, cat_frame = cat_cap.read()
        
        if not human_ret or not cat_ret:
            break
        
        # Process the human frame to get segmented lips
        segmented_lips, hull, coords = segment_lips_and_teeth(human_frame)
        
        if hull is not None:
            # Detect cat's face
            cat_face = detect_cat_face_using_yolo(cat_frame)

            """print("\n\n\n\n\n\n")
            print(cat_face)
            print("\n\n\n\n\n\n")"""

            if cat_face is not None:
                x, y, w, h = cat_face
                cv2.rectangle(cat_frame, (x, y), (x+w, y+h), (0, 0, 255), 2)  # Red bounding box
                face_history.append((x,y,w,h))
            
            # Use the median of recent face detections to stabilize the bounding box
            if face_history:
                median_face = np.median(face_history, axis=0).astype(int)
                x, y, w, h = median_face
            elif last_cat_face is not None:
                x, y, w, h = last_cat_face
            else:
                x, y, w, h = None, None, None, None, None
            
            if x is not None:
                # Draw bounding box around cat's face
                cv2.rectangle(cat_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Estimate cat's mouth region (lower third of the face)
                mouth_y = y + int(2*h/3)
                mouth_h = max(1,int(h/3))
                
                # Adjust the size of the lip mask to cover 70% of the cat's mouth region width
                human_mouth_w = np.max(hull[:, 0, 0]) - np.min(hull[:, 0, 0])
                human_mouth_h = np.max(hull[:, 0, 1]) - np.min(hull[:, 0, 1])
                cat_mouth_w = int(0.7 * w)  # 70% of the face width
                scale_x = cat_mouth_w / human_mouth_w
                scale_y = mouth_h / human_mouth_h
                
                # Resize segmented lips to match 70% of cat's mouth width
                resized_lips = cv2.resize(segmented_lips, (cat_mouth_w, mouth_h))
                
                # Calculate the position to place the lips within the cat's face bounding box
                start_x = x + w - cat_mouth_w  # Start from the right side
                start_y = mouth_y
                end_x = x + w
                end_y = mouth_y + mouth_h
                
                # Ensure the dimensions match
                resized_lips = cv2.resize(resized_lips, (end_x - start_x, end_y - start_y))
                
                # Create a mask for the resized lips
                mask = resized_lips[:, :, 3] / 255.0
                
                # Overlay resized lips on the cat frame
                for c in range(0, 3):
                    cat_frame[start_y:end_y, start_x:end_x, c] = (1 - mask) * cat_frame[start_y:end_y, start_x:end_x, c] + \
                                                                   mask * resized_lips[:, :, c]
        # Write the frame
        out.write(cat_frame)
        
        # Update frame count and print progress
        frame_count += 1
        progress = (frame_count / total_frames) * 100
        print(f"\n\n\n\n\n\nProgress: {progress:.2f}%", end="")
    
    print("\nProcessing complete!")
    
    # Release everything
    human_cap.release()
    cat_cap.release()
    out.release()
    cv2.destroyAllWindows()

    # Upload the output video to Firebase Storage
    bucket = storage.bucket()
    blob = bucket.blob(f'processed_videos/{os.path.basename(output_path)}')
    blob.upload_from_filename(output_path)

    # Make the blob publicly accessible
    blob.make_public()

    # Get the public URL
    output_url = blob.public_url

    # Remove the local output video file
    os.remove(output_path)

    return output_url

#cat_video_url = 'https://drive.google.com/uc?export=download&id=1qBEZixvQgECAN3JjYcLAoCTlRbGnSPXJ'
cat_video_url = 'https://drive.google.com/uc?export=download&id=1_l5mowDH5wXWB2pTSBriujpxM2CjrW2s'
#cat_video_url = 'https://drive.google.com/uc?export=download&id=1-rN9db7xeYKDDEK8PUDZGiucF3EYfq5j'
human_video_url = 'https://drive.google.com/uc?export=download&id=1zPNI_dwRa53NfniDhQc3sGa0YYaY-kQj'
cat_video_path = 'temp_cat_video.mp4'
human_video_path = 'temp_human_video.mp4'
output_video = 'output_video_cat_human.mp4'

# Download videos
urllib.request.urlretrieve(cat_video_url, cat_video_path)
urllib.request.urlretrieve(human_video_url, human_video_path)

# Process videos
output_url = process_video(human_video_path, cat_video_path, output_video)

# Clean up downloaded videos
os.remove(cat_video_path)
os.remove(human_video_path)

print(f"Processed video URL: {output_url}")
