import cv2
import mediapipe as mp
from PIL import Image
import firebase_admin
from firebase_admin import credentials, storage
import numpy as np
from PIL import ImageDraw, ImageChops, ImageFilter
import os
from ultralytics import SAM
import torch

# Initialize Firebase Admin SDK
cred = credentials.Certificate("../../credentials.json")
firebase_admin.initialize_app(cred, {
    'storageBucket': 'test-421b9.appspot.com'
})

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
# Load the hand image and Domino's logo
hand_image_path = '../misc/finger2.jpg'
dominos_logo_path = '../misc/dominos.png'  # Add path to the Domino's logo

model = SAM("../models/sam2_t.pt")

# Load hand image and convert it to RGB
hand_img = cv2.imread(hand_image_path)
hand_img_rgb = cv2.cvtColor(hand_img, cv2.COLOR_BGR2RGB)

height, width, _ = hand_img.shape 

results = hands.process(hand_img_rgb)

if results.multi_hand_landmarks:
    for hand_landmarks in results.multi_hand_landmarks:
        # Get all hand landmarks
        landmarks = [(lm.x, lm.y) for lm in hand_landmarks.landmark]
        
        # Calculate the center of the hand
        center_x = sum(lm[0] for lm in landmarks) / len(landmarks)
        center_y = sum(lm[1] for lm in landmarks) / len(landmarks)
        
        # Convert normalized coordinates to pixel coordinates
        h, w, _ = hand_img.shape
        center_x_px = int(center_x * w)
        center_y_px = int(center_y * h)
        
        print(f"Center of the hand: ({center_x_px}, {center_y_px})")
        
        # Run SAM on the center of the hand
        results = model(hand_img_rgb, points=[[center_x_px, center_y_px]])
        
        # Get the mask from the results
        mask = results[0].masks.data[0].cpu().numpy()
        
        # Convert the mask to uint8 format
        mask = (mask * 255).astype(np.uint8)
        
        # Create an image showing the mask
        mask_image = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        
        # Save the mask image locally
        mask_image_path = 'hand_mask.png'
        cv2.imwrite(mask_image_path, mask_image)
        
        # Upload the mask image to Firebase Storage
        bucket = storage.bucket()
        blob = bucket.blob("hand_mask.png")
        blob.upload_from_filename(mask_image_path)
        
        # Make the file public
        blob.make_public()
        
        # Get the public URL
        firebase_url = blob.public_url
        
        print(f"Mask image uploaded to Firebase Storage. Public URL: {firebase_url}")
        

print(f"Done!")
