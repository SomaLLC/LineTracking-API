import cv2
import mediapipe as mp
from PIL import Image
import firebase_admin
from firebase_admin import credentials, storage

# Initialize Firebase Admin SDK
cred = credentials.Certificate("../../credentials.json")
firebase_admin.initialize_app(cred, {
    'storageBucket': 'test-421b9.appspot.com'
})

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

# Load the hand image and Domino's logo
hand_image_path = '../misc/finger4.jpg'
dominos_logo_path = '../misc/dominos.png'  # Add path to the Domino's logo

# Load hand image and convert it to RGB
hand_img = cv2.imread(hand_image_path)
hand_img_rgb = cv2.cvtColor(hand_img, cv2.COLOR_BGR2RGB)

# Detect hands in the image
results = hands.process(hand_img_rgb)

# Check if hand landmarks were detected
if results.multi_hand_landmarks:
    for hand_landmarks in results.multi_hand_landmarks:
        # Get the pinky finger coordinates (landmark 20 is the tip of the pinky, 19 is the base)
        pinky_tip = hand_landmarks.landmark[20]
        pinky_base = hand_landmarks.landmark[19]
        h, w, _ = hand_img.shape
        pinky_tip_x, pinky_tip_y = int(pinky_tip.x * w), int(pinky_tip.y * h)
        pinky_base_x, pinky_base_y = int(pinky_base.x * w), int(pinky_base.y * h)

        # Calculate angle of rotation
        angle = np.degrees(np.arctan2(pinky_tip_y - pinky_base_y, pinky_tip_x - pinky_base_x))

        # Load Domino's logo as PIL image
        dominos_logo = Image.open(dominos_logo_path)
        dominos_logo = dominos_logo.resize((50, 50))  # Resize the logo to fit the pinky

        # Rotate the logo
        rotated_logo = dominos_logo.rotate(-angle, expand=True)

        # Convert the OpenCV image (hand_img) to PIL
        hand_img_pil = Image.fromarray(cv2.cvtColor(hand_img, cv2.COLOR_BGR2RGB))

        # Calculate position to paste the rotated logo
        paste_x = pinky_tip_x - rotated_logo.width // 2
        paste_y = pinky_tip_y - rotated_logo.height // 2

        # Superimpose the rotated logo at the pinky location
        hand_img_pil.paste(rotated_logo, (paste_x, paste_y), rotated_logo)

        # Save or show the final image
        hand_img_pil.show()
        hand_img_pil.save("hand_with_dominos.png")

        # Upload the image to Firebase Storage
        bucket = storage.bucket()
        blob = bucket.blob("hand_with_dominos.png")
        blob.upload_from_filename("hand_with_dominos.png")

        # Make the file public
        blob.make_public()

        # Get the public URL
        firebase_url = blob.public_url
        print(f"Image uploaded to Firebase Storage. Public URL: {firebase_url}")

print(f"Done!: ", results)

print("Attributes of results:")
for attr in dir(results):
    if not attr.startswith('__'):
        print(attr)

print(results.multi_hand_world_landmarks)
print(results.multi_handedness)
print(results.multi_handedness[0])
print(results.multi_handedness[0].classification)
print(results.multi_handedness[0].classification[0])
print(results.multi_handedness[0].classification[0].label)
print(results.multi_handedness[0].classification[0].label)