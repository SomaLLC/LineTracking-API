import firebase_admin
from firebase_admin import credentials, storage
import cv2
import os
from ultralytics import YOLO
import torch
import numpy as np

def get_polygon_masks(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply adaptive thresholding with a smaller block size and lower C value
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 7, 1)

    # Apply edge detection to capture fine lines
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3)

    # Combine thresholded image and edges
    combined = cv2.bitwise_or(thresh, edges)

    # Apply morphological operations to clean up the image
    kernel = np.ones((3,3), np.uint8)
    morph = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Find contours in the image
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a copy of the original image to draw on
    result = image.copy()
    # Find the largest contour (assumed to be the rectangle)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Approximate the contour to a polygon
        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
        
        # Create a mask for the area inside the rectangle
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        
        if len(approx) == 4:
            cv2.drawContours(mask, [approx], 0, 255, -1)
        else:
            x, y, w, h = cv2.boundingRect(largest_contour)
            cv2.rectangle(mask, (x, y), (x+w, y+h), 255, -1)
        
        # Apply the mask to the original image
        result = cv2.bitwise_and(image, image, mask=mask)

    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

    # Apply edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Apply stronger dilation to connect nearby edges
    kernel = np.ones((5,5), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=2)

    # Apply Hough Line Transform with relaxed parameters
    lines = cv2.HoughLinesP(dilated_edges, 1, np.pi/180, threshold=30, minLineLength=30, maxLineGap=20)

    # Create a blank mask to draw the lines
    line_mask = np.zeros_like(gray)

    # Draw the detected lines on the mask with thicker lines
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_mask, (x1, y1), (x2, y2), 255, 4)  # Increased thickness to 4

    kernel = np.ones((1,1), np.uint8)

    # Apply stronger dilation to further connect the lines
    connected_lines = cv2.dilate(line_mask, kernel, iterations=3)

    # Apply closing operation to fill small gaps
    connected_lines = cv2.morphologyEx(connected_lines, cv2.MORPH_CLOSE, kernel)

    # Apply morphological operations to remove small "nugget" shapes
    kernel = np.ones((2,2), np.uint8)
    cleaned_lines = cv2.morphologyEx(connected_lines, cv2.MORPH_OPEN, kernel)

    # Apply skeletonization to thin the lines
    def skeletonize(img):
        skel = np.zeros(img.shape, np.uint8)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
        while True:
            open = cv2.morphologyEx(img, cv2.MORPH_OPEN, element)
            temp = cv2.subtract(img, open)
            eroded = cv2.erode(img, element)
            skel = cv2.bitwise_or(skel, temp)
            img = eroded.copy()
            if cv2.countNonZero(img) == 0:
                break
        return skel

    thinned_lines = skeletonize(cleaned_lines)

    # Make lines even thicker
    thicker_lines = cv2.dilate(thinned_lines, np.ones((12,12), np.uint8), iterations=2)
    # Invert the image
    inverted_lines = cv2.bitwise_not(thicker_lines)
    # Create a copy of the inverted lines image for flood filling
    flood_fill_image = inverted_lines.copy()

    # Convert to 3-channel image for colored flood fill
    flood_fill_image = cv2.cvtColor(flood_fill_image, cv2.COLOR_GRAY2BGR)

    # Get image dimensions
    h, w = inverted_lines.shape[:2]

    # Create a mask for flood fill, slightly larger than the image
    mask = np.zeros((h+2, w+2), np.uint8)

    # Flood fill from multiple seed points
    for y in range(0, h, 20):  # Adjust step size as needed
        for x in range(0, w, 20):  # Adjust step size as needed
            if inverted_lines[y, x] == 255:  # If the pixel is white (background)
                # Generate a random color
                color = tuple(np.random.randint(0, 255, 3).tolist())
                # Perform flood fill
                cv2.floodFill(flood_fill_image, mask, (x, y), color)

    # Create separate polygonal masks for each colored section
    h, w = flood_fill_image.shape[:2]
    all_masks = np.zeros((h, w, 3), dtype=np.uint8)

    # Get unique colors in the flood-filled image
    unique_colors = np.unique(flood_fill_image.reshape(-1, flood_fill_image.shape[-1]), axis=0)

    # Remove the color of the polygon containing (0,0)
    color_at_origin = flood_fill_image[0, 0]
    unique_colors = unique_colors[~np.all(unique_colors == color_at_origin, axis=1)]

    # Define minimum area threshold for polygons
    min_area_threshold = 4000  # Adjust this value as needed

    valid_masks = []
    for i, color in enumerate(unique_colors, start=1):
        # Skip black color (background)
        if np.all(color == [0, 0, 0]):
            continue
        
        # Create a binary mask for the current color
        color_mask = cv2.inRange(flood_fill_image, color, color)
        
        # Find contours of the color region
        contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find the largest contour (main section)
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Check if the contour area is above the threshold
            if cv2.contourArea(largest_contour) < min_area_threshold:
                continue # Skip this polygon
            
            # Create a polygonal approximation of the contour
            epsilon = 0.01 * cv2.arcLength(largest_contour, True)
            approx = cv2.approxPolyDP(largest_contour, epsilon, True)
            
            # Create a mask for this polygon
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(mask, [approx], 0, 255, -1)
            
            valid_masks.append(mask)

    # Stack the valid masks into a 3D array
    if valid_masks:
        all_masks = np.stack(valid_masks, axis=-1)
    else:
        all_masks = np.zeros((h, w, 1), dtype=np.uint8)

    return all_masks

# Initialize Firebase Admin SDK
cred = credentials.Certificate("../../credentials.json")
firebase_admin.initialize_app(cred, {
    'storageBucket': 'test-421b9.appspot.com'
})

torch.cuda.set_device(0)

# Load the YOLO model
model = YOLO("../runs/detect/paper-yolo3/weights/best.pt") 

# Ask user for video path
#video_path = input("Enter the path to the video file: ")

video_path = "https://drive.google.com/uc?export=download&id=1ZB1zJRVeKbd_kv77acpJ72EqzJckJwBJ"

# Create an output directory to save the images
output_dir = "output_frames"
os.makedirs(output_dir, exist_ok=True)

# Open the video file
cap = cv2.VideoCapture(video_path)

# Check if the video file was opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

frame_count = 0
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define the codec and create a VideoWriter object to save the video
output_video_path = "output_video_alt2.avi"
fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # Using H.264 codec for better compatibility
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

polygon_masks = []
# Process the video frame by frame
frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count == 0:
        polygon_masks = get_polygon_masks(frame)
        print(f"Shape of polygon_masks: {polygon_masks.shape}")
        print(f"Number of polygons: {polygon_masks.shape[2]}")
    
    # Overlay the polygon masks on the frame
    """overlay = frame.copy()
    for i in range(polygon_masks.shape[2]):
        mask = polygon_masks[:,:,i]
        color = np.random.randint(0, 255, 3).tolist()
        overlay[mask > 0] = color
    
    # Blend the overlay with the original frame
    alpha = 0.2  # Adjust this value to change the transparency (0.0 - 1.0)
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)"""
        
    
    # Process every 5th frame
    if frame_count % 5 == 0:
        # Run YOLO model on the frame
        results = model(frame)
        
        # Dictionary to store bounding boxes with unique IDs
        if 'bounding_boxes' not in locals():
            bounding_boxes = {}
        
        # Draw detections with confidence > 0.5
        for result in results:
            for obj, label, bbox, confidence in zip(result.boxes.data, result.boxes.cls, result.boxes.xyxy, result.boxes.conf):
                if confidence > 0.5:
                    x1, y1, x2, y2 = map(int, bbox)
                    new_bb = {'bbox': (x1, y1, x2, y2), 'label': label, 'confidence': confidence}
                    
                    # Check if the new bounding box matches any existing one
                    matched = False
                    for bb_id, existing_bb in bounding_boxes.items():
                        existing_area = (existing_bb['bbox'][2] - existing_bb['bbox'][0]) * (existing_bb['bbox'][3] - existing_bb['bbox'][1])
                        intersection_area = max(0, min(x2, existing_bb['bbox'][2]) - max(x1, existing_bb['bbox'][0])) * \
                                            max(0, min(y2, existing_bb['bbox'][3]) - max(y1, existing_bb['bbox'][1]))
                        if intersection_area / existing_area > 0.6:
                            bounding_boxes[bb_id] = new_bb
                            matched = True
                            break
                    
                    # If no match found, add as a new bounding box
                    if not matched:
                        new_id = max(bounding_boxes.keys(), default=0) + 1
                        bounding_boxes[new_id] = new_bb
                        bb_id = new_id
                    
                    # Draw the bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Find the polygon with the most overlap
                    bbox_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                    cv2.rectangle(bbox_mask, (x1, y1), (x2, y2), 255, -1)
                    max_overlap = 0
                    max_overlap_polygon = 0
                    for i in range(polygon_masks.shape[2]):
                        overlap = cv2.bitwise_and(bbox_mask, polygon_masks[:,:,i])
                        overlap_area = np.sum(overlap > 0)
                        if overlap_area > max_overlap:
                            max_overlap = overlap_area
                            max_overlap_polygon = i + 1
                    
                    # Calculate the center of the bounding box
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    
                    # Display bounding box ID and polygon number in the center
                    bb_text = f"BB {bb_id} in region {max_overlap_polygon}"
                    cv2.putText(frame, "BB " + str(bb_id), (center_x, center_y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 3)
                    cv2.putText(frame, "in", (center_x, center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3)
                    cv2.putText(frame, "region " + str(max_overlap_polygon), (center_x, center_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 3)

                    # Add text with label and confidence
                    """ label_text = f"{label} {confidence:.2f}"
                    cv2.putText(frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    # Create a mask for the bounding box
                    bbox_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                    cv2.rectangle(bbox_mask, (x1, y1), (x2, y2), 255, -1)
                    
                    # Find the polygon with the most overlap
                    max_overlap = 0
                    max_overlap_polygon = 0
                    for i in range(polygon_masks.shape[2]):
                        overlap = cv2.bitwise_and(bbox_mask, polygon_masks[:,:,i])
                        overlap_area = np.sum(overlap > 0)
                        #print(f"Overlap area for polygon {i+1}: {overlap_area}")
                        if overlap_area > max_overlap:
                            max_overlap = overlap_area
                            max_overlap_polygon = i + 1
                    
                    #print(f"Selected polygon: {max_overlap_polygon}")
                    
                    # Calculate the center of the bounding box
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    
                    # Add text with label, confidence, and polygon number
                    label_text = f"{label} {confidence:.2f} P{max_overlap_polygon}"
                    cv2.putText(frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    # Add polygon number in the center of the bounding box
                    cv2.putText(frame, str(max_overlap_polygon), (center_x, center_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2) """
     
        # Save the frame as an image
        output_path = os.path.join(output_dir, f"frame_{frame_count:04d}.jpg")
        cv2.imwrite(output_path, frame)
        
        # Write the frame to the video file
        out.write(frame)
        
        percentage_processed = (frame_count / total_frames) * 100
        print(f"Processed {percentage_processed:.2f}% of frames.")
    
    frame_count += 1

# Release the video capture and writer
cap.release()
out.release()

print(f"Processed {frame_count} frames. Video saved as '{output_video_path}'.")

# Upload the video to Firebase Storage
bucket = storage.bucket()
blob = bucket.blob(output_video_path)
blob.upload_from_filename(output_video_path)

# Make the file public
blob.make_public()

# Get the public URL
firebase_url = blob.public_url
print(f"Video uploaded to Firebase Storage. Public URL: {firebase_url}")

