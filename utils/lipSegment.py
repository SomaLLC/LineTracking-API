import cv2
import numpy as np
import mediapipe as mp
import urllib.request
import os

import firebase_admin
from firebase_admin import credentials, storage

# Initialize Firebase Admin SDK
cred = credentials.Certificate("../../credentials.json")
firebase_admin.initialize_app(cred, {
    'storageBucket': 'test-421b9.appspot.com'
})


def segment_lips(frame):
    """
    Segment lips from a frame and make the rest of the frame black.
    
    :param frame: Input frame (BGR format)
    :return: Frame with only lips visible, rest is black
    """
    mp_face_mesh = mp.solutions.face_mesh
    
    with mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5) as face_mesh:
        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame and find facial landmarks
        results = face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            
            # Get lip landmarks (upper and lower lip)
            lip_landmarks = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
            
            # Extract lip coordinates
            h, w = frame.shape[:2]
            lip_points = [(int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)) for i in lip_landmarks]
            
            # Create a mask for the lips
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [np.array(lip_points)], 255)
            
            # Apply Gaussian blur to soften the mask edges
            mask = cv2.GaussianBlur(mask, (5, 5), 0)
            
            # Create the output frame (black background)
            output_frame = np.zeros_like(frame)
            
            # Apply the mask to the original frame and combine with the black background
            output_frame = cv2.bitwise_and(frame, frame, mask=mask)
            
            return output_frame
        else:
            # If no face is detected, return a black frame
            return np.zeros_like(frame)

def process_video(video_path, video_url, output_path):
    """
    Process a video to segment lips and make the rest black.
    
    :param video_path: Path to store the downloaded video file
    :param video_url: URL of the input video
    :param output_path: Path to save the output video file
    """
    # Download the video from the URL
    urllib.request.urlretrieve(video_url, video_path)
    
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process the frame
        processed_frame = segment_lips(frame)
        
        # Write the frame
        out.write(processed_frame)
    
    # Release everything
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    # Remove the downloaded video file
    os.remove(video_path)

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

# Example usage:
video_path = 'temp_input_video.mp4'
video_url = 'https://drive.google.com/uc?export=download&id=1zPNI_dwRa53NfniDhQc3sGa0YYaY-kQj'
output_video = 'output_video.mp4'  # Changed to a simple filename in the current directory
process_video(video_path, video_url, output_video)
