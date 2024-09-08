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

#TODO: try pretrained YOLO instead of mediapipe
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
            pinky_mcp = hand_landmarks.landmark[17]  # Pinky metacarpophalangeal joint
            h, w, _ = hand_img.shape
            pinky_tip_x, pinky_tip_y = int(pinky_tip.x * w), int(pinky_tip.y * h)
            pinky_base_x, pinky_base_y = int(pinky_base.x * w), int(pinky_base.y * h)
            pinky_mcp_x, pinky_mcp_y = int(pinky_mcp.x * w), int(pinky_mcp.y * h)

            # Calculate pinky length
            pinky_length = ((pinky_tip_x - pinky_base_x)**2 + (pinky_tip_y - pinky_base_y)**2)**0.5
            
            # Calculate finger width (distance between pinky base and MCP joint)
            finger_width = ((pinky_base_x - pinky_mcp_x)**2 + (pinky_base_y - pinky_mcp_y)**2)**0.5
            
            # Calculate logo size (1/3 of pinky length)
            logo_size = int(pinky_length / 3)
            
            # Resize the logo, making it 2% wider than the finger width
            logo_width = int(finger_width * 1.02)  # 2% wider
            logo_height = int(logo_width * (dominos_logo.height / dominos_logo.width))  # Maintain aspect ratio
            resized_logo = dominos_logo.resize((logo_width, logo_height))

            # Calculate angle of rotation
            angle = np.degrees(np.arctan2(pinky_tip_y - pinky_base_y, pinky_tip_x - pinky_base_x))

            # Rotate the logo
            rotated_logo = resized_logo.rotate(-(angle + 90), expand=True)

            print("ANGLE: ", angle)

            # Calculate position to paste the rotated logo
            # Adjust the position based on the angle of the pinky
            offset_factor = 0.3  # Adjust this value to control how far down the finger the logo is placed
            offset_x = int(np.cos(np.radians(angle)) * pinky_length * offset_factor)
            offset_y = int(np.sin(np.radians(angle)) * pinky_length * offset_factor)

            paste_x = pinky_tip_x - rotated_logo.width // 2 - offset_x
            paste_y = pinky_tip_y - rotated_logo.height // 2 - offset_y

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
    mask = cv2.GaussianBlur(mask, (15, 15), 0)

    # Apply morphological operations to further smooth the mask
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Apply edge-preserving filter
    mask = cv2.edgePreservingFilter(mask, flags=1, sigma_s=60, sigma_r=0.4)

    # Apply bilateral filter to smooth while preserving edges
    mask = cv2.bilateralFilter(mask, 9, 75, 75)

    # Create a PIL Image from the mask
    mask_pil = Image.fromarray(mask)

    # Resize the mask to match the hand image size
    mask_pil = mask_pil.resize((w, h))

    # Create a new image for the masked logo
    masked_logo = Image.new('RGBA', (w, h), (0, 0, 0, 0))

    # Paste the rotated logo onto the new image at the calculated position
    masked_logo.paste(rotated_logo, (paste_x, paste_y), rotated_logo)

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
for i in range(9, 11):
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
