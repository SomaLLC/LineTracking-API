import cv2
import numpy as np
import os

# Correct file path
file_path = '../misc/areaSegments2.jpg'

# Check if the file exists
if not os.path.isfile(file_path):
    raise FileNotFoundError(f"File does not exist at path: {file_path}")

# Read the image
image = cv2.imread(file_path)

# Check if the image is loaded successfully
if image is None:
    raise ValueError(f"Unable to load image at path: {file_path}. The file might be corrupted or in an unsupported format.")

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

# Display the result
cv2.imshow("Image Inside Detected Rectangle", result)
cv2.waitKey(0)

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

cv2.imshow("Line mask: ", line_mask)

cv2.waitKey(0)

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

# Display the strongly connected lines
# Apply morphological operations to remove small "nugget" shapes
kernel = np.ones((2,2), np.uint8)
cleaned_lines = cv2.morphologyEx(connected_lines, cv2.MORPH_OPEN, kernel)

cv2.imshow("Cleaned lines: ", cleaned_lines)

cv2.waitKey(0)

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

cv2.imshow("Skeleton: ", thinned_lines)

cv2.waitKey(0)

# Double the thickness of the white lines
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

cv2.imshow("Flood Filled Sections", flood_fill_image)

cv2.waitKey(0)




# Create separate polygonal masks for each colored section
h, w = flood_fill_image.shape[:2]
all_masks = np.zeros((h, w, 3), dtype=np.uint8)

# Get unique colors in the flood-filled image
# Get unique colors in the flood-filled image
unique_colors = np.unique(flood_fill_image.reshape(-1, flood_fill_image.shape[-1]), axis=0)

# Remove the color of the polygon containing (0,0)
color_at_origin = flood_fill_image[0, 0]
unique_colors = unique_colors[~np.all(unique_colors == color_at_origin, axis=1)]

# Define minimum area threshold for polygons
min_area_threshold = 1000  # Adjust this value as needed

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
        
        # Draw the polygonal mask
        section_mask = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.drawContours(section_mask, [approx], 0, color.tolist(), -1)
        
        # Add the section mask to the combined mask
        all_masks = cv2.add(all_masks, section_mask)
        
        # Calculate the center of the contour
        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            
            # Put the number in the center of the polygon
            cv2.putText(all_masks, str(i), (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, (0, 0, 0), 2, cv2.LINE_AA)

# Display all masks together
cv2.imshow("All Polygonal Masks with Numbers", all_masks)
cv2.waitKey(0)
cv2.destroyAllWindows()


bbs = [[806,175,1082,348], [1338,662,1605,933]]

# Create a copy of the original image to draw on
result_image = all_masks.copy()

# Function to calculate overlap area between a bounding box and a polygon
def calculate_overlap(bb, polygon):
    bb_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(bb_mask, (bb[0], bb[1]), (bb[2], bb[3]), 255, -1)
    polygon_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(polygon_mask, [polygon], 0, 255, -1)
    overlap = cv2.bitwise_and(bb_mask, polygon_mask)
    return np.sum(overlap > 0)

# Iterate through the bounding boxes
for bb in bbs:
    # Draw the bounding box on the result image
    cv2.rectangle(result_image, (bb[0], bb[1]), (bb[2], bb[3]), (0, 255, 0), 2)
    
    max_overlap = 0
    max_overlap_polygon = None
    
    # Find the polygon with the highest overlap
    for i, color in enumerate(unique_colors, start=1):
        if np.all(color == [0, 0, 0]):
            continue
        
        color_mask = cv2.inRange(flood_fill_image, color, color)
        contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest_contour) >= min_area_threshold:
                overlap = calculate_overlap(bb, largest_contour)
                if overlap > max_overlap:
                    max_overlap = overlap
                    max_overlap_polygon = i
    
    # Add text with the polygon number inside the bounding box
    if max_overlap_polygon is not None:
        text = f"Polygon {max_overlap_polygon}"
        cv2.putText(result_image, text, (bb[0]+5, bb[1]+20), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, (255, 255, 255), 2, cv2.LINE_AA)

# Display the result
cv2.imshow("Bounding Boxes with Polygon Numbers", result_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
