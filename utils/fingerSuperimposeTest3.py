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
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0)
mp_drawing = mp.solutions.drawing_utils

# Load the Domino's logo
dominos_logo_path = '../misc/dominos.png'
dominos_logo = Image.open(dominos_logo_path)

model = SAM("../models/sam2_t.pt")

# Function to process a single image
def process_image(hand_image_path):
    # Load hand image and convert it to RGB
    hand_img = cv2.imread(hand_image_path)
    hand_img_rgb = cv2.cvtColor(hand_img, cv2.COLOR_BGR2RGB)

    height, width, _ = hand_img.shape 

    results = hands.process(hand_img_rgb)
    # Calculate the center of the image
    h, w, _ = hand_img.shape
    center_x_px = w // 2
    center_y_px = h // 2

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Get all hand landmarks
            landmarks = [(lm.x, lm.y) for lm in hand_landmarks.landmark]

            pinky_tip = hand_landmarks.landmark[20]
            pinky_base = hand_landmarks.landmark[19]
            h, w, _ = hand_img.shape
            pinky_tip_x, pinky_tip_y = int(pinky_tip.x * w), int(pinky_tip.y * h)
            pinky_base_x, pinky_base_y = int(pinky_base.x * w), int(pinky_base.y * h)

            # Calculate pinky length
            pinky_length = ((pinky_tip_x - pinky_base_x)**2 + (pinky_tip_y - pinky_base_y)**2)**0.5
            
            # Calculate logo size (1/3 of pinky length)
            logo_size = int(pinky_length / 3)
            
            # Resize the logo
            resized_logo = dominos_logo.resize((logo_size, logo_size))

            # Calculate angle of rotation
            angle = np.degrees(np.arctan2(pinky_tip_y - pinky_base_y, pinky_tip_x - pinky_base_x))

            # Rotate the logo
            rotated_logo = resized_logo.rotate(-(angle + 90), expand=True)

            # Calculate position to paste the rotated logo
            paste_x = pinky_tip_x - rotated_logo.width // 2
            paste_y = pinky_tip_y - rotated_logo.height // 2

            # Convert the OpenCV image (hand_img) to PIL
            hand_img_pil = Image.fromarray(cv2.cvtColor(hand_img, cv2.COLOR_BGR2RGB))

            # Calculate the center of the hand
            hand_center_x = sum(lm[0] for lm in landmarks) / len(landmarks)
            hand_center_y = sum(lm[1] for lm in landmarks) / len(landmarks)
            
            # Convert normalized coordinates to pixel coordinates
            center_x_px = int(hand_center_x * w)
            center_y_px = int(hand_center_y * h)
            
            print(f"Center of the hand: ({center_x_px}, {center_y_px})")

    print(f"Center of the image: ({center_x_px}, {center_y_px})")

    # Run SAM on the center of the image
    results = model(hand_img_rgb, points=[[center_x_px, center_y_px]])

    # Get the mask from the results
    mask = results[0].masks.data[0].cpu().numpy()

    # Convert the mask to uint8 format
    mask = (mask * 255).astype(np.uint8)

    # Apply Gaussian blur to smooth the mask
    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    # Create a PIL Image from the mask
    mask_pil = Image.fromarray(mask)

    # Resize the mask to match the hand image size
    mask_pil = mask_pil.resize((w, h))

    # Create a new image for the masked logo
    masked_logo = Image.new('RGBA', (w, h), (0, 0, 0, 0))

    # Calculate new position to paste the rotated logo (moved down the finger)
    offset = int(pinky_length * 0.2)  # Move logo down by 20% of finger length
    new_paste_x = paste_x - int(offset * 0.7)
    new_paste_y = paste_y + offset

    # Paste the rotated logo onto the new image at the new position
    masked_logo.paste(rotated_logo, (new_paste_x, new_paste_y), rotated_logo)

    # Apply the mask to the logo
    masked_logo = Image.composite(masked_logo, Image.new('RGBA', (w, h), (0, 0, 0, 0)), mask_pil)

    # Paste the masked logo onto the hand image
    hand_img_pil.paste(masked_logo, (0, 0), masked_logo)

    # Convert the result back to OpenCV format
    hand_img = cv2.cvtColor(np.array(hand_img_pil), cv2.COLOR_RGB2BGR)

    # Create an image showing the mask
    mask_image = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    return hand_img_pil, mask_image

# Process all images
for i in range(1, 5):
    hand_image_path = f'../misc/finger{i}.jpg'
    hand_img_pil, mask_image = process_image(hand_image_path)

    # Save the mask image locally
    mask_image_path = f'hand_mask_{i}.png'
    cv2.imwrite(mask_image_path, mask_image)

    # Save the hand image with logo locally
    hand_image_path = f'hand_with_logo_{i}.png'
    hand_img_pil.save(hand_image_path)

    # Upload the mask image to Firebase Storage
    bucket = storage.bucket()
    mask_blob = bucket.blob(f"hand_mask_{i}.png")
    mask_blob.upload_from_filename(mask_image_path)

    # Upload the hand image with logo to Firebase Storage
    hand_blob = bucket.blob(f"hand_with_logo_{i}.png")
    hand_blob.upload_from_filename(hand_image_path)

    # Make the files public
    mask_blob.make_public()
    hand_blob.make_public()

    # Get the public URLs
    mask_firebase_url = mask_blob.public_url
    hand_firebase_url = hand_blob.public_url

    print(f"Mask image {i} uploaded to Firebase Storage. Public URL: {mask_firebase_url}")
    print(f"Hand image {i} with logo uploaded to Firebase Storage. Public URL: {hand_firebase_url}")

print("All images processed and uploaded successfully!")
